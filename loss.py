import torchvision
import torch
import config

from torchvision.ops._box_convert import _box_cxcywh_to_xyxy as xywh_to_xyxy
from scipy.optimize import linear_sum_assignment as lsa
from torch.nn import functional as F

def box_iou_union(boxes1, boxes2):
	area1 = torchvision.ops.boxes.box_area(boxes1)
	area2 = torchvision.ops.boxes.box_area(boxes2)
	lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
	rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
	wh = (rb - lt).clamp(min=0)
	inter = wh[..., 0] * wh[..., 1]
	union = area1[:, None] + area2 - inter
	iou = inter / union
	return iou, union

def generalized_box_iou(boxes1, boxes2):
	iou, union = box_iou_union(boxes1, boxes2)
	lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
	rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
	wh = (rb - lt).clamp(min=0)
	area = wh[..., 0] * wh[..., 1]
	return iou - (area - union) / area

class HungarianMatcher(torch.nn.Module):
	def __init__(self) -> None:
		super(HungarianMatcher, self).__init__()

	@torch.no_grad()
	def forward(self, outputs: torch.Tensor, targets: list) -> list:
		B, N, L = outputs.shape
		out_prob = outputs[..., 5:].view(-1, L - 5)
		out_bbox = outputs[..., :4].view(-1, 4)
		tgt_classes = torch.cat([x[:, -1] for x in targets]).long()
		tgt_bbox = torch.cat([x[:, :4] for x in targets])

		cost_class = -out_prob[:, tgt_classes]
		cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1.0)
		cost_giou = -generalized_box_iou(xywh_to_xyxy(out_bbox), tgt_bbox)

		costs = ((cost_bbox * 5) + (cost_giou * 2) + (cost_class * 1)).view(B, N, -1).cpu()
		indices = [lsa(item[idx]) for idx, item in enumerate(costs.split([x.size(0) for x in targets], -1))]
		return [(torch.LongTensor(i), torch.LongTensor(j)) for i, j in indices]

class DETRCriterion(torch.nn.Module):
	def __init__(self, num_classes, eos_coef = 0.01) -> None:
		super(DETRCriterion, self).__init__()
		self.num_classes = num_classes
		self.matcher = HungarianMatcher().to(config.DEVICE)
		self.losses = dict({'boxes': 0, 'classes': 0, 'cardinality': 0})
		empty_weight = torch.ones(self.num_classes + 1)
		empty_weight[self.num_classes] = eos_coef
		self.register_buffer('empty_weight', empty_weight)

	def zero_losses(self) -> None:
		self.losses.update({'boxes': 0, 'classes': 0, 'cardinality': 0})

	@torch.no_grad()
	def loss_cardinality(self, outputs, targets):
		num_targets = torch.Tensor([item.size(0) for item in targets], device=config.DEVICE)
		predictions = torch.sum(outputs[..., 4:].argmax(-1) != self.num_classes, dim=1).float()
		return F.l1_loss(predictions, num_targets)

	def loss_labels(self, outputs, targets, indices):
		idx = self._get_src_permutation_idx(indices)
		out_classes = outputs[..., 4:]
		tgt_classes_obj = torch.cat([item[:, -1][i] for item, (_, i) in zip(targets, indices)]).long()
		tgt_classes = torch.full(out_classes.shape[:2], self.num_classes, device=config.DEVICE).long()
		tgt_classes[idx] = tgt_classes_obj
		return F.cross_entropy(out_classes.transpose(1, 2), tgt_classes, self.empty_weight)

	def loss_boxes(self, outputs, targets, indices, num_boxes):
		idx = self._get_src_permutation_idx(indices)
		bbox_out = outputs[..., :4][idx]
		bbox_tar = torch.cat([item[i, :4] for item, (_, i) in zip(targets, indices)], dim=0)
		loss_bbox = F.l1_loss(bbox_out, bbox_tar, reduction='none')
		loss_giou = torchvision.ops.generalized_box_iou_loss(xywh_to_xyxy(bbox_out), bbox_tar)
		return (loss_bbox.sum().div(num_boxes) * 5) + (loss_giou.sum().div(num_boxes) * 2)

	def _get_src_permutation_idx(self, indices):
		batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
		src_idx = torch.cat([src for (src, _) in indices])
		return batch_idx, src_idx

	def _get_tgt_permutation_idx(self, indices):
		batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
		tgt_idx = torch.cat([tgt for (_, tgt) in indices])
		return batch_idx, tgt_idx

	def forward(self, outputs: torch.Tensor, targets: list):
		targets = [item.to(config.DEVICE) for item in targets]
		num_boxes = sum([item.size(0) for item in targets])
		indices = self.matcher(outputs, targets)
		loss_boxes = self.loss_boxes(outputs, targets, indices, num_boxes)
		loss_classes = self.loss_labels(outputs, targets, indices)
		loss_cardinality = self.loss_cardinality(outputs, targets)
		self.losses['boxes'] += loss_boxes
		self.losses['classes'] += loss_classes
		self.losses['cardinality'] += loss_cardinality
		return loss_boxes + loss_classes + loss_cardinality

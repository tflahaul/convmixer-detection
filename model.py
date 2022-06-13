import torch

from torch import Tensor
from typing import OrderedDict
from torchvision.ops import FeaturePyramidNetwork as FPN

class Residual(torch.nn.Module):
	def __init__(self, func) -> None:
		super(Residual, self).__init__()
		self.func = func

	def forward(self, inputs: Tensor) -> Tensor:
		return self.func(inputs) + inputs

class DetectionHead(torch.nn.Module):
	def __init__(self, dim: int, num_classes: int, filters: int = 96) -> None:
		super(DetectionHead, self).__init__()
		self.fpn = FPN([filters * 2, filters * 3], (5 + num_classes))
		self.conv0 = torch.nn.Sequential(
			torch.nn.Conv2d(dim, filters, 9, 3),
			torch.nn.GELU(),
			torch.nn.GroupNorm(1, filters))
		self.conv1 = torch.nn.Sequential(
			torch.nn.Conv2d(filters, filters * 2, 5),
			torch.nn.GELU(),
			torch.nn.GroupNorm(1, filters * 2))
		self.conv2 = torch.nn.Sequential(
			torch.nn.Conv2d(filters * 2, filters * 3, 3),
			torch.nn.GELU(),
			torch.nn.GroupNorm(1, filters * 3))

	def forward(self, inputs: Tensor) -> Tensor:
		f0 = self.conv0(inputs) # downsample to 16x16
		f1 = self.conv1(f0)
		f2 = self.conv2(f1)
		out = self.fpn(OrderedDict({'feat0': f1, 'feat1': f2}))
		out = torch.cat([x.flatten(2, 3).transpose(1, 2) for x in out.values()], dim=1)
		out[..., :4] = out[..., :4].sigmoid()
		return out

def ConvMixer(
	in_channels: int,
	dim: int,
	depth: int,
	kernel_size: int = 9,
	patch_size: int = 7,
	num_classes: int = 1000
):
	return torch.nn.Sequential(
		torch.nn.Conv2d(in_channels, dim, patch_size, stride=patch_size),
		torch.nn.SiLU(),
		torch.nn.GroupNorm(1, dim),
		*(torch.nn.Sequential(
			Residual(torch.nn.Sequential(
				torch.nn.Conv2d(dim, dim, kernel_size, groups=dim, padding='same'),
				torch.nn.SiLU(),
				torch.nn.GroupNorm(1, dim))),
			torch.nn.Conv2d(dim, dim, kernel_size=1),
			torch.nn.SiLU(),
			torch.nn.GroupNorm(1, dim)) for d in range(depth)),
		DetectionHead(dim, num_classes))

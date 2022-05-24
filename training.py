import torchvision
import torch
import json
import time
import config

from torchvision import transforms as trsfm
from loss import ObjectDetectionCriterion
from model import ConvMixer
from PIL import Image

class PascalVOC(torch.utils.data.Dataset):
	def __init__(self, location: str) -> None:
		super(PascalVOC, self).__init__()
		self.location = location
		self.classes = open(f'{location}/classes.txt', mode='r').read().splitlines()
		self.items = json.load(open(f'{location}/targets.json', mode='r'))
		self.transforms = trsfm.Compose((
			trsfm.Resize((416, 416)),
			trsfm.ToTensor(),
			trsfm.ConvertImageDtype(torch.float32)))
#			trsfm.Normalize((0.4564, 0.4370, 0.4081), (0.2717, 0.2680, 0.2810))))

	@classmethod
	def post_process_batch(self, items: list):
		images = torch.stack(tuple(zip(*items))[0])
		targets = tuple(zip(*items))[1]
		return images, targets

	def __getitem__(self, index: int):
		image = Image.open(f"{self.location}/images/{self.items[index]['filename']}").convert('RGB')
		targets = [x['bbox'] + [self.classes.index(x['class'])] for x in self.items[index]['targets']]
		return self.transforms(image), torch.Tensor(targets)

	def __len__(self) -> int:
		return len(self.items)

def fit(model, optimizer, criterion, train_set) -> None:
	for epoch in range(1, config.MAX_ITER + 1):
		optimizer.zero_grad()
		criterion.zero_losses()
		running_loss, start = 0, time.time()
		for minibatch, (images, targets) in enumerate(train_set, 1):
			out = model(images.to(config.DEVICE))
			loss = criterion(out, targets)
			running_loss += loss.item()
			loss.backward()
			if minibatch % config.SUBDIVISIONS == 0:
				optimizer.step()
				optimizer.zero_grad()
		print(f"epoch {epoch:>3d}/{config.MAX_ITER:<3d}| loss:{running_loss:.5f}, loss-boxes:{criterion.losses['boxes']:.4f}, loss-cls:{criterion.losses['classes']:.4f}, card:{criterion.losses['cardinality']:.4f}, time:{time.time() - start:.2f}")
	torch.save(model.state_dict(), 'convmixer-1152-3.pth')

def main() -> None:
	model = ConvMixer(3, 1152, 3, 9, 7, config.NUM_CLASSES).to(config.DEVICE) # 1536, 20
	trainval_set = PascalVOC(config.DATASET_DIR)
	train_set, test_set = torch.utils.data.random_split(
		dataset=trainval_set,
		lengths=[round(len(trainval_set) * 0.85), round(len(trainval_set) * 0.15)])
	train_set = torch.utils.data.DataLoader(
		dataset=train_set,
		batch_size=(config.BATCH_SIZE // config.SUBDIVISIONS),
		collate_fn=PascalVOC.post_process_batch,
		pin_memory=True,
		shuffle=True,
		drop_last=True)
	test_set = torch.utils.data.DataLoader(
		dataset=test_set,
		collate_fn=PascalVOC.post_process_batch,
		batch_size=len(test_set))
	optimizer = torch.optim.AdamW(
		params=model.parameters(),
		weight_decay=5e-4,
		lr=1e-4)
	criterion = ObjectDetectionCriterion(num_classes=config.NUM_CLASSES)
	print(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
	fit(model, optimizer, criterion, train_set)

if __name__ == '__main__':
	main()

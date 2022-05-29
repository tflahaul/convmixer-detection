import torch
import json
import time
import config

from torchvision import transforms as trsfm
from matplotlib import pyplot as plt
from loss import DETRCriterion
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
			trsfm.ConvertImageDtype(torch.float32),
			trsfm.Normalize((0.4564, 0.4370, 0.4081), (0.2717, 0.2680, 0.2810))))

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

def plot_metrics(metrics: dict) -> None:
	_, axes = plt.subplots(1, 3, figsize=(12, 4))
	for index, (key, values) in enumerate(metrics.items()):
		axes[index].plot(values)
		axes[index].set_title(key)
	plt.tight_layout()
	plt.show()

def main() -> None:
	model = ConvMixer(3, 1536, 20, 9, 7, config.NUM_CLASSES).to(config.DEVICE)
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
		weight_decay=config.DECAY,
		lr=config.LEARNING_RATE)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
		optimizer=optimizer,
		T_max=config.MAX_ITER,
		eta_min=5e-5)
	criterion = DETRCriterion(num_classes=config.NUM_CLASSES)
	metrics = {'boxes': list(), 'classes': list(), 'cardinality': list()}

	for epoch in range(1, config.MAX_ITER + 1):
		criterion.zero_losses()
		optimizer.zero_grad()
		running_loss, start = 0, time.time()
		for minibatch, (images, targets) in enumerate(train_set, 1):
			out = model(images.to(config.DEVICE))
			loss = criterion(out, targets)
			running_loss += loss.item()
			loss.backward()
			if minibatch % config.SUBDIVISIONS == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_NORM)
				optimizer.step()
				optimizer.zero_grad()
		scheduler.step()
		for key, value in criterion.losses.items():
			metrics[key].append(value.item())
		print(f"epoch {epoch:>2d}/{config.MAX_ITER:<2d}| loss:{running_loss:.3f}, {(', ').join([f'{k}:{v[-1]:.2f}' for k, v in metrics.items()])}, time:{time.time() - start:.0f}")

	torch.save(model.state_dict(), 'convmixer-1536-20.pth')
	plot_metrics(metrics)

if __name__ == '__main__':
	main()

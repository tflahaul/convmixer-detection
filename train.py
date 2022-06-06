import config
import torch

from matplotlib import pyplot as plt
from dataloader import PascalVOC
from model import ConvMixer
from loss import DETRCriterion

def plot_metrics(metrics: dict) -> None:
	_, axes = plt.subplots(1, 3, figsize=(12, 4))
	for index, (key, values) in enumerate(metrics.items()):
		axes[index].plot(values)
		axes[index].set_title(key)
	plt.tight_layout()
	plt.show()

def main() -> None:
	model = ConvMixer(3, 768, 32, 9, 7, config.NUM_CLASSES).to(config.DEVICE)
	trainval_set = PascalVOC(config.DATASET_DIR)
	train_set, test_set = torch.utils.data.random_split(
		dataset=trainval_set,
		lengths=[round(len(trainval_set) * 0.85), round(len(trainval_set) * 0.15)])
	train_set = torch.utils.data.DataLoader(
		dataset=train_set,
		batch_size=(config.BATCH_SIZE // config.SUBDIVISIONS),
		collate_fn=PascalVOC.post_process_batch,
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
	criterion = DETRCriterion(config.NUM_CLASSES).to(config.DEVICE)
	metrics = {'boxes': list(), 'classes': list(), 'cardinality': list()}
	print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

	for epoch in range(1, config.MAX_ITER + 1):
		criterion.zero_losses()
		optimizer.zero_grad()
		running_loss = 0
		for minibatch, (images, targets) in enumerate(train_set, 1):
			out = model(images.to(config.DEVICE))
			loss = criterion(out, targets)
			running_loss += loss.item()
			loss.backward()
			if minibatch % config.SUBDIVISIONS == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_NORM)
				optimizer.step()
				optimizer.zero_grad()
		for key, value in criterion.losses.items():
			metrics[key].append(value.item())
		print(f"epoch {epoch:>2d}/{config.MAX_ITER:<2d}| loss:{running_loss:.3f}, {(', ').join([f'{k}:{v[-1]:.2f}' for k, v in metrics.items()])}")

	torch.save(model.state_dict(), 'convmixer-1536-20.pth')
	plot_metrics(metrics)

if __name__ == '__main__':
	main()

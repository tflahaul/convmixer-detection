import json
import torch
import config

from torchvision import transforms as trsfm
from torch import Tensor
from PIL import Image

def _letterbox_resize(image: Tensor, targets: Tensor, size: int):
	r = size / max(image.size)
	shape = int(image.size[0] * r), int(image.size[1] * r)
	img = Image.new('RGB', (size, size))
	img.paste(image.resize(shape), ((size - shape[0]) // 2, (size - shape[1]) // 2))
	start = 1 if image.size[0] > image.size[1] else 0
	new_r = (shape[start] / size)
	targets[..., start:4:2] = (targets[..., start:4:2] * new_r) + ((1 - new_r) / 2)
	return img, targets

class PascalVOC(torch.utils.data.Dataset):
	def __init__(self, location: str) -> None:
		super(PascalVOC, self).__init__()
		self.root = location
		self.classes = open(f'{location}/classes.txt', mode='r').read().splitlines()
		with open(f'{location}/targets.json', mode='r') as fd:
			self.items = json.load(fd)
		self.transforms = trsfm.Compose((
			trsfm.ToTensor(),
			trsfm.ConvertImageDtype(torch.float32)))
		#	trsfm.Normalize((0.4564, 0.4370, 0.4081), (0.2717, 0.2680, 0.2810))))

	@classmethod
	def post_process_batch(self, items: list):
		images = torch.stack(tuple(zip(*items))[0])
		targets = tuple(zip(*items))[1]
		return images, targets

	def __getitem__(self, index: int):
		image = Image.open(f"{self.root}/images/{self.items[index]['filename']}").convert('RGB')
		targets = [x['bbox'] + [self.classes.index(x['class'])] for x in self.items[index]['targets']]
		image, targets = _letterbox_resize(image, Tensor(targets), config.IMG_SIZE)
		return self.transforms(image), targets

	def __len__(self) -> int:
		return len(self.items)

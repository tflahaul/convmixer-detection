import json
import torch
import config

from torchvision import transforms as trsfm
from PIL import Image

class PascalVOC(torch.utils.data.Dataset):
	def __init__(self, location: str) -> None:
		super(PascalVOC, self).__init__()
		self.location = location
		self.classes = open(f'{location}/classes.txt', mode='r').read().splitlines()
		with open(f'{location}/targets.json', mode='r') as fd:
			self.items = json.load(fd)
		self.transforms = trsfm.Compose((
			trsfm.Resize((config.IMG_SIZE, config.IMG_SIZE)),
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

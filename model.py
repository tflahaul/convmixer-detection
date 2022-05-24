import torch

class Residual(torch.nn.Module):
	def __init__(self, func) -> None:
		super(Residual, self).__init__()
		self.func = func

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.func(inputs) + inputs

class DetectionHead(torch.nn.Module):
	def __init__(self, dim: int, num_classes: int, depth: int = 3, bbox_attrs: int = 4) -> None:
		super(DetectionHead, self).__init__()
		self.func = torch.nn.Sequential(
			torch.nn.Conv2d(dim, 96, kernel_size=5, stride=5),
			torch.nn.GroupNorm(1, 96),
			torch.nn.GELU(),
			*(torch.nn.Sequential(
				torch.nn.Conv2d(96, 96, kernel_size=3),
				torch.nn.GroupNorm(1, 96),
				torch.nn.GELU()) for d in range(depth)),
			torch.nn.Conv2d(96, (bbox_attrs + num_classes + 1), kernel_size=1))

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		out = self.func(inputs)
		B, C, H, W = out.shape
		out = out.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
		out[..., :4] = out[..., :4].sigmoid()
		out[..., 4:] = out[..., 4:].softmax(-1)
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
		torch.nn.GELU(),
		torch.nn.GroupNorm(1, dim),
		*(torch.nn.Sequential(
			Residual(torch.nn.Sequential(
				torch.nn.Conv2d(dim, dim, kernel_size, groups=dim, padding='same'),
				torch.nn.GELU(),
				torch.nn.GroupNorm(1, dim))),
			torch.nn.Conv2d(dim, dim, kernel_size=1),
			torch.nn.GELU(),
			torch.nn.GroupNorm(1, dim)) for d in range(depth)),
		DetectionHead(dim, num_classes))

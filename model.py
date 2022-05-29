import torch

class Residual(torch.nn.Module):
	def __init__(self, func) -> None:
		super(Residual, self).__init__()
		self.func = func

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.func(inputs) + inputs

class DetectionHead(torch.nn.Module):
	def __init__(self, dim: int, num_classes: int) -> None:
		super(DetectionHead, self).__init__()
		self.func = torch.nn.Sequential(
			torch.nn.Conv2d(dim, 96, kernel_size=3, stride=3),
			torch.nn.GELU(),
			torch.nn.GroupNorm(1, 96),
			torch.nn.Conv2d(96, (5 + num_classes), kernel_size=1))

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		out = self.func(inputs)
		B, C, H, W = out.shape
		out = out.reshape(B, C, H * W).permute(0, 2, 1).contiguous()
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

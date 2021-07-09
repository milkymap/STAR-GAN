import torch as th 
import torch.nn as nn 

class RBlock(nn.Module):
	def __init__(self):
		super(RBlock, self).__init__()
		self.body = nn.Sequential(
			nn.Conv2d(256, 256, 3, 1, 1),
			nn.InstanceNorm2d(256), 
			nn.ReLU(),
			nn.Conv2d(256, 256, 3, 1, 1),
			nn.InstanceNorm2d(256) 	
		)

	def forward(self, X0):
		return X0 + self.body(X0)

class Generator(nn.Module):
	def __init__(self, i_channels, n_domains, n_blocks):
		super(Generator, self).__init__()
		self.head = nn.Sequential(
			nn.Conv2d(i_channels + n_domains, 64, 7, 1, 3),
			nn.InstanceNorm2d(64),
			nn.ReLU(), 
			nn.Conv2d( 64, 128, 4, 2, 1),
			nn.InstanceNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 256, 4, 2, 1),
			nn.InstanceNorm2d(256),
			nn.ReLU()
		) 
		self.body = nn.Sequential(*[ RBlock() for _ in range(n_blocks)])
		self.tail = nn.Sequential(
			nn.ConvTranspose2d(256, 128, 4, 2, 1), 
			nn.InstanceNorm2d(128),
			nn.ReLU(),
			nn.ConvTranspose2d(128,  64, 4, 2, 1), 
			nn.InstanceNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 3, 7, 1, 3),
			nn.Tanh()

		) 

	def forward(self, X0, L0):
		L0 = L0[:, :, None, None]
		L0 = L0.repeat((1, 1, X0.size(2), X0.size(3)))
		A0 = th.cat([X0, L0], dim=1)
		return self.tail(self.body(self.head(A0)))

if __name__ == '__main__':
	G = Generator(3, 5, 6)
	print(G)
	X = th.randn((1, 3, 128, 128))
	print(G(X, th.randn((1, 5))).shape)
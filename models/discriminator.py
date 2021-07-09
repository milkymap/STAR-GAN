import torch as th 
import torch.nn as nn 

class Discriminator(nn.Module):
	def __init__(self, n_domains):
		super(Discriminator, self).__init__()
		self.head = nn.Sequential(
			nn.Conv2d(3, 64, 4, 2, 1),
			nn.LeakyReLU(0.2)
		)
		self.body = []
		for idx in range(5):
			i_channels = 64 * 2 ** idx
			o_channels = i_channels * 2 
			self.body.append(nn.Conv2d(i_channels, o_channels, 4, 2, 1))
			self.body.append(nn.LeakyReLU())
			
		self.body = nn.Sequential(*self.body)
		self.tail = nn.Conv2d(o_channels, 1, 3, 1, 1)
		self.term = nn.Sequential(
			nn.Conv2d(o_channels, n_domains, 2, 1, 0),
			nn.Sigmoid()
		)

	def forward(self, X0):
		X1 = self.body(self.head(X0))
		return th.squeeze(self.tail(X1)), th.squeeze(self.term(X1))

if __name__ == '__main__':
	D = Discriminator(3)
	print(D)
	X0 = th.randn((3, 3, 128, 128))
	X1, X2 = D(X0)
	print(X1.shape)
	print(X2.shape)

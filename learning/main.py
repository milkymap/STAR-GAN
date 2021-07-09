import os 
import cv2
import click 

import torch as th 
import torch.nn as nn 
import torch.optim as optim 

from torch.utils.data import DataLoader 

from models.generator import Generator
from models.discriminator import Discriminator

from libraries.source import Data 
from libraries.strategies import * 


@click.command()
@click.option('--device', help='gpu index if gpu is present', type=str)
@click.option('--source', help='path to source image data', type=click.Path(True))
@click.option('--attributes', help='path to attributes data', type=click.Path(True))
@click.option('--nb_epochs', help='number of epochs for training', type=int)
@click.option('--bt_size', help='size of batch', type=int)
@click.option('--n_domains', help='number of domains', type=int)
@click.option('--path_to_dump', help='path where sample image were stored', default='storage')
def train(device, source, attributes ,nb_epochs, bt_size, n_domains, path_to_dump):
	 dataset = Data(source, attributes)
	 dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=bt_size)

	 G = Generator(i_channels=3, n_domains=n_domains, n_blocks=6).to(device)
	 D = Discriminator(n_domains=n_domains).to(device)

	 O_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.9, 0.999))
	 O_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.9, 0.999))

	 lab_loss = nn.BCELoss().to(device)
	 dis_loss = nn.MSELoss().to(device)
	 rec_loss = nn.L1Loss().to(device)

	 epoch_counter = 0
	 while epoch_counter < nb_epochs:
	 	for I, (X, S, T) in enumerate(dataloader):
	 		print(S, T)
	 		X = X.to(device)
	 		S = S.to(device)
	 		T = T.to(device)

	 		R = th.ones((X.shape[0], 2, 2)).float().to(device)
	 		F = th.zeros((X.shape[0], 2, 2)).float().to(device)
	 		# Generator 
	 		G_XY = G(X, T)
	 		G_YX = G(G_XY, S)

	 		P_XY, C_XY = D(G_XY)
	 		
	 		L01 = dis_loss(P_XY, R)
	 		L02 = rec_loss(G_YX, X)
	 		L03 = lab_loss(C_XY, T)
	 		L04 = L01 + 10 * L02 + L03

	 		O_G.zero_grad()
	 		L04.backward()
	 		O_G.step()

	 		# Discriminator 
	 		P_X, C_X = D(X)
	 		P_XY, _ = D(G_XY.detach())

	 		L11 = (dis_loss(P_X, R) + dis_loss(P_XY, F)) / 2
	 		L12 = lab_loss(C_X, S) 
	 		L13 = L11 + L12  
	 		O_D.zero_grad()
	 		L13.backward()
	 		O_D.step()

	 		print(L04.item(), L13.item())
	 		if I % 2 == 0:
	 			X = X.cpu()
	 			G_XY = G_XY.cpu()
	 			I_SR = to_grid(X, nb_rows=1)
	 			I_HR = to_grid(G_XY, nb_rows=1)
	 			I_LS = th2cv(th.cat((I_SR, I_HR), -1)) * 255
	 			cv2.imwrite(f'{path_to_dump}/img_{epoch_counter:02d}_{I:03d}.jpg', I_LS)

	 	epoch_counter += 1

if __name__ == '__main__':
	train()
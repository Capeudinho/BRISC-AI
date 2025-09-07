import torch
import torch.nn as nn

class DoubleConvolution(nn.Module):

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_convolution = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True), nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True))

	def forward(self, element):
		return self.double_convolution(element)

class UNet(nn.Module):

	def __init__(self, in_channels, classes):
		super().__init__()
		self.down_1 = DoubleConvolution(in_channels, 64)
		self.pool_1 = nn.MaxPool2d(2)
		self.down_2 = DoubleConvolution(64, 128)
		self.pool_2 = nn.MaxPool2d(2)
		self.down_3 = DoubleConvolution(128, 256)
		self.pool_3 = nn.MaxPool2d(2)
		self.down_4 = DoubleConvolution(256, 512)
		self.pool_4 = nn.MaxPool2d(2)
		self.bottleneck = DoubleConvolution(512, 1024)
		self.up_4 = nn.ConvTranspose2d(1024, 512, kernel_size = 2, stride = 2)
		self.convolution_4 = DoubleConvolution(1024, 512)
		self.up_3 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2)
		self.convolution_3 = DoubleConvolution(512, 256)
		self.up_2 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2)
		self.convolution_2 = DoubleConvolution(256, 128)
		self.up_1 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2)
		self.convolution_1 = DoubleConvolution(128, 64)
		self.out_class = nn.Conv2d(64, classes, kernel_size = 1)

	def forward(self, x):
		new_down_1 = self.down_1(x)
		new_down_2 = self.down_2(self.pool_1(new_down_1))
		new_down_3 = self.down_3(self.pool_2(new_down_2))
		new_down_4 = self.down_4(self.pool_3(new_down_3))
		new_botteneck = self.bottleneck(self.pool_4(new_down_4))
		new_up_4 = self.up_4(new_botteneck)
		new_up_4 = self.convolution_4(torch.cat([new_up_4, new_down_4], dim = 1))
		new_up_3 = self.up_3(new_up_4)
		new_up_3 = self.convolution_3(torch.cat([new_up_3, new_down_3], dim = 1))
		new_up_2 = self.up_2(new_up_3)
		new_up_2 = self.convolution_2(torch.cat([new_up_2, new_down_2], dim = 1))
		new_up_1 = self.up_1(new_up_2)
		new_up_1 = self.convolution_1(torch.cat([new_up_1, new_down_1], dim = 1))
		result = self.out_class(new_up_1)
		return result
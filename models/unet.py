import torch
import torch.nn as nn

class DoubleConvolution(nn.Module):

	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.double_convolution = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True), nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace = True))

	def forward(self, element):
		return self.double_convolution(element)

class UNet(nn.Module):

	def __init__(self, in_channels, out_channels, base_channels):
		super().__init__()
		self.down_1 = DoubleConvolution(in_channels, base_channels)
		self.pool_1 = nn.MaxPool2d(2)
		self.down_2 = DoubleConvolution(base_channels, base_channels*2)
		self.pool_2 = nn.MaxPool2d(2)
		self.down_3 = DoubleConvolution(base_channels*2, base_channels*4)
		self.pool_3 = nn.MaxPool2d(2)
		self.down_4 = DoubleConvolution(base_channels*4, base_channels*8)
		self.pool_4 = nn.MaxPool2d(2)
		self.bottleneck = DoubleConvolution(base_channels*8, base_channels*16)
		self.up_4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, kernel_size = 2, stride = 2)
		self.convolution_4 = DoubleConvolution(base_channels*16, base_channels*8)
		self.up_3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, kernel_size = 2, stride = 2)
		self.convolution_3 = DoubleConvolution(base_channels*8, base_channels*4)
		self.up_2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, kernel_size = 2, stride = 2)
		self.convolution_2 = DoubleConvolution(base_channels*4, base_channels*2)
		self.up_1 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size = 2, stride = 2)
		self.convolution_1 = DoubleConvolution(base_channels*2, base_channels)
		self.output = nn.Conv2d(base_channels, out_channels, kernel_size = 1)

	def forward(self, element):
		down_1 = self.down_1(element)
		down_2 = self.down_2(self.pool_1(down_1))
		down_3 = self.down_3(self.pool_2(down_2))
		down_4 = self.down_4(self.pool_3(down_3))
		bottleneck = self.bottleneck(self.pool_4(down_4))
		up_4 = self.up_4(bottleneck)
		up_4 = self.convolution_4(torch.cat([up_4, down_4], dim = 1))
		up_3 = self.up_3(up_4)
		up_3 = self.convolution_3(torch.cat([up_3, down_3], dim = 1))
		up_2 = self.up_2(up_3)
		up_2 = self.convolution_2(torch.cat([up_2, down_2], dim = 1))
		up_1 = self.up_1(up_2)
		up_1 = self.convolution_1(torch.cat([up_1, down_1], dim = 1))
		result = self.output(up_1)
		return result
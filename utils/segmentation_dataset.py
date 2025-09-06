import os
import torch
import random as rd
from PIL import Image as img
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):

	def __init__(self, dataset_directory, quantity):
		self.dataset_directory = dataset_directory
		self.dataset_names = rd.sample(os.listdir(dataset_directory), quantity)

	def __len__(self):
		length = len(self.dataset_names)
		return length

	def __getitem__(self, index):
		data = torch.load(f"{self.dataset_directory}/{self.dataset_names[index]}")
		image_tensor = data["image_tensor"]
		mask_tensor = data["mask_tensor"]
		return image_tensor, mask_tensor
import os
from PIL import Image as img
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):

	def __init__(self, image_directory, mask_directory, transform_image, transform_mask, quantity):
		self.image_directory = image_directory
		self.mask_directory = mask_directory
		self.image_names = os.listdir(image_directory)[:quantity]
		self.mask_names = os.listdir(mask_directory)[:quantity]
		self.transform_image = transform_image
		self.transform_mask = transform_mask

	def __len__(self):
		length = len(self.image_names)
		return length

	def __getitem__(self, index):
		image_path = f"{self.image_directory}/{self.image_names[index]}"
		mask_path = f"{self.mask_directory}/{self.mask_names[index]}"
		with img.open(image_path) as image, img.open(mask_path) as mask:
			if self.transform_image:
				new_image = self.transform_image(image)
			if self.transform_mask:
				new_mask = self.transform_mask(mask)
			return new_image, new_mask
		return None, None
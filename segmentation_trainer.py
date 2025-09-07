import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from utils.segmentation_dataset import SegmentationDataset

base_channels = 64
training_quantity = 10
validating_quantity = 10
batch_size = 10
learning_rate = 1e-3
epochs = 10
title = "small"

os.chdir(os.path.dirname(os.path.abspath(__file__)))
training_dataset = SegmentationDataset("data/segmentation/training", training_quantity)
validating_dataset = SegmentationDataset("data/segmentation/validating", validating_quantity)
training_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
validating_loader = DataLoader(validating_dataset, batch_size = batch_size, shuffle = False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels = 1, out_channels = 1, base_channels = base_channels).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
for epoch in range(epochs):
	model.train()
	training_loss = 0.0
	for image_tensors, mask_tensors in training_loader:
		image_tensors = image_tensors.to(device)
		mask_tensors = mask_tensors.to(device)
		preds = model(image_tensors)
		loss = criterion(preds, mask_tensors)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		training_loss = training_loss+loss.item()
	# model.eval()
	# validating_loss = 0.0
	# with torch.no_grad():
	# 	for image_tensors, mask_tensors in validating_loader:
	# 		image_tensors = image_tensors.to(device)
	# 		mask_tensors = mask_tensors.to(device)
	# 		preds = model(image_tensors)
	# 		loss = criterion(preds, mask_tensors)
	# 		validating_loss = validating_loss+loss.item()
	# print(f"Epoch {epoch+1} of {epochs}, training loss of {training_loss}, validating loss of {validating_loss}.")
	print(f"Epoch {epoch+1} of {epochs}, training loss of {training_loss}.")
os.makedirs("exports", exist_ok = True)
torch.save(model.state_dict(), f"exports/unet_segmentation_{title}.pt")
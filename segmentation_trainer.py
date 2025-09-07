import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from utils.segmentation_dataset import SegmentationDataset

os.chdir(os.path.dirname(os.path.abspath(__file__)))
training_dataset = SegmentationDataset("data/segmentation/training", 10)
validating_dataset = SegmentationDataset("data/segmentation/validating", 10)
training_loader = DataLoader(training_dataset, batch_size = 10, shuffle = True)
validating_loader = DataLoader(validating_dataset, batch_size = 10, shuffle = False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels = 1, classes = 1).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
epochs = 10
for epoch in range(epochs):
	print(f"Epoch {epoch+1} of {epochs}.")
	model.train()
	for image_tensors, mask_tensors in training_loader:
		image_tensors = image_tensors.to(device)
		mask_tensors = mask_tensors.to(device)
		preds = model(image_tensors)
		loss = criterion(preds, mask_tensors)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	model.eval()
	with torch.no_grad():
		for image_tensors, mask_tensors in validating_loader:
			image_tensors = image_tensors.to(device)
			mask_tensors = mask_tensors.to(device)
			preds = model(image_tensors)
			loss = criterion(preds, mask_tensors)
os.makedirs("exports", exist_ok = True)
torch.save(model.state_dict(), "exports/unet_segmentation.pt")
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.unet import UNet
from utils.segmentation_dataset import SegmentationDataset

os.chdir(os.path.dirname(os.path.abspath(__file__)))
training_dataset = SegmentationDataset("data/segmentation/training/images", "data/segmentation/training/masks", transforms.Compose([transforms.ToTensor()]), transforms.Compose([transforms.ToTensor()]), 10)
validating_dataset = SegmentationDataset("data/segmentation/validating/images", "data/segmentation/validating/masks", transforms.Compose([transforms.ToTensor()]), transforms.Compose([transforms.ToTensor()]), 10)
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
	for images, masks in training_loader:
		images = images.to(device)
		masks = masks.to(device)
		preds = model(images)
		loss = criterion(preds, masks)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	model.eval()
	with torch.no_grad():
		for images, masks in validating_loader:
			images = images.to(device)
			masks = masks.to(device)
			preds = model(images)
			loss = criterion(preds, masks)
os.makedirs("exports", exist_ok = True)
torch.save(model.state_dict(), "exports/unet_segmentation.pth")
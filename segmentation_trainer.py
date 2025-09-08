import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.segmentation_dataset import SegmentationDataset
from losses.bce_dice_loss import BCEDiceLoss

base_channels = 32
training_quantity = 512
validating_quantity = 256
batch_size = 32
learning_rate = 2e-3
epochs = 32

os.chdir(os.path.dirname(os.path.abspath(__file__)))
model = UNet(in_channels = 1, out_channels = 1, base_channels = base_channels)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
training_dataset = SegmentationDataset("data/segmentation/training", training_quantity)
validating_dataset = SegmentationDataset("data/segmentation/validating", validating_quantity)
training_loader = DataLoader(training_dataset, batch_size = batch_size, shuffle = True)
validating_loader = DataLoader(validating_dataset, batch_size = batch_size, shuffle = False)
criterion = BCEDiceLoss(dice_weight = 0.75)
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
for epoch in range(epochs):
	model.train()
	training_loss = 0.0
	for image_tensors, mask_tensors in training_loader:
		image_tensors = image_tensors.to(device)
		mask_tensors = mask_tensors.to(device)
		prediction_tensors = model(image_tensors)
		loss = criterion(prediction_tensors, mask_tensors)
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
	# 		prediction_tensors = model(image_tensors)
	# 		loss = criterion(prediction_tensors, mask_tensors)
	# 		validating_loss = validating_loss+loss.item()
os.makedirs("weights", exist_ok = True)
torch.save(model.state_dict(), f"weights/unet_segmentation_{base_channels}.pt")
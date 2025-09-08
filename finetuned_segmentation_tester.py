import torch
from torch.utils.data import DataLoader
from models.unet import UNet
from datasets.segmentation_dataset import SegmentationDataset

testing_quantity = 120

def dice_coefficient(prediction_tensors, mask_tensors):
	intersection = (prediction_tensors*mask_tensors).sum(dim = (1, 2, 3))
	union = prediction_tensors.sum(dim = (1, 2, 3))+mask_tensors.sum(dim = (1, 2, 3))
	dice = (2.0*intersection+1e-4)/(union+1e-4)
	result = dice.mean().item()
	return result

model = UNet(in_channels = 1, out_channels = 1, base_channels = 32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("weights/unet_finetuned_segmentation_32.pt", map_location = device))
model.to(device)
model.eval()
testing_dataset = SegmentationDataset("data/segmentation/testing", testing_quantity)
testing_loader = DataLoader(testing_dataset, batch_size = 32, shuffle = False)
dice_accuracies = []
with torch.no_grad():
	for image_tensors, mask_tensors in testing_loader:
		image_tensors = image_tensors.repeat(1, 3, 1, 1)
		image_tensors = image_tensors.to(device)
		mask_tensors = mask_tensors.to(device)
		prediction_tensors = model(image_tensors)
		prediction_tensors = torch.sigmoid(prediction_tensors)
		prediction_tensors = (prediction_tensors > 0.5).float()
		dice_accuracy = dice_coefficient(prediction_tensors, mask_tensors)
		dice_accuracies.append(dice_accuracy)
dice_accuracy = sum(dice_accuracies)/len(dice_accuracies)
print(f"Dice coefficient accuracy of {dice_accuracy:.8f}.")
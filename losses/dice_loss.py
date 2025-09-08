import torch
import torch.nn as nn

class DiceLoss(nn.Module):

	def __init__(self):
		super(DiceLoss, self).__init__()

	def forward(self, prediction_tensors, mask_tensors):
		probability_tensors = torch.sigmoid(prediction_tensors)
		probability_tensors = probability_tensors.contiguous().view(-1)
		mask_tensors = mask_tensors.contiguous().view(-1)
		intersection = (probability_tensors * mask_tensors).sum()
		union = probability_tensors.sum()+mask_tensors.sum()
		dice = (2.0*intersection+1e-4)/(union+1e-4)
		result = 1-dice
		return result
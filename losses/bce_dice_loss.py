import torch.nn as nn
from losses.dice_loss import DiceLoss

class BCEDiceLoss(nn.Module):

	def __init__(self, dice_weight = 0.5):
		super(BCEDiceLoss, self).__init__()
		self.bce_criterion = nn.BCEWithLogitsLoss()
		self.dice_criterion = DiceLoss()
		self.dice_weight = dice_weight

	def forward(self, prediction_tensors, mask_tensors):
		bce_loss = self.bce_criterion(prediction_tensors, mask_tensors)
		dice_loss = self.dice_criterion(prediction_tensors, mask_tensors)
		result = (self.dice_weight*dice_loss)+((1-self.dice_weight)*bce_loss)
		return result
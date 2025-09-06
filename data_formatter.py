import os
import torch
import torchvision.transforms as transforms
from PIL import Image as img

os.chdir(os.path.dirname(os.path.abspath(__file__)))
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels = 1), transforms.ToTensor()])
for split in ["training", "validating", "testing"]:
	os.makedirs(f"data/segmentation/{split}", exist_ok = True)
	os.makedirs(f"data/classification/{split}", exist_ok = True)
	for image_name in os.listdir(f"archive/classification/{split}"):
		with img.open(f"archive/classification/{split}/{image_name}") as image:
			new_image = image.convert("L")
			image_tensor = transform(new_image)
			torch.save(image_tensor, f"data/classification/{split}/{os.path.splitext(image_name)[0]}.pt")
	for image_name in os.listdir(f"archive/segmentation/{split}/images"):
		with img.open(f"archive/segmentation/{split}/images/{image_name}") as image, img.open(f"archive/segmentation/{split}/masks/{os.path.splitext(image_name)[0]}.png") as mask:
			new_image = image.convert("L")
			image_tensor = transform(new_image)
			new_mask = mask.convert("L")
			mask_tensor = transform(new_mask)
			torch.save({"image_tensor": image_tensor, "mask_tensor": mask_tensor}, f"data/segmentation/{split}/{os.path.splitext(image_name)[0]}.pt")
import os
from PIL import Image as img
import torchvision.transforms as transforms

os.chdir(os.path.dirname(os.path.abspath(__file__)))
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels = 1), transforms.ToTensor(), transforms.ToPILImage()])
for root, _, images in os.walk("archive"):
	path = f"data/{os.path.relpath(root, "archive")}"
	os.makedirs(path, exist_ok = True)
	for image in images:
		with img.open(f"{root}/{image}") as image_file:
			new_image_file = image_file.convert("L")
			new_image_file = transform(new_image_file)
			new_image_file.save(f"{path}/{image}")
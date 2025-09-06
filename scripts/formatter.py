import os
from PIL import Image as img
import torchvision.transforms as transforms

os.chdir(os.path.dirname(os.path.abspath(__file__)))
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.Grayscale(num_output_channels = 1), transforms.ToTensor(), transforms.ToPILImage()])
for root, _, images in os.walk("../archive"):
	path = f"../data/{os.path.relpath(root, "../archive")}"
	os.makedirs(path, exist_ok = True)
	for image in images:
		image_file = img.open(f"{root}/{image}").convert("L")
		image_file = transform(image_file)
		image_file.save(f"{path}/{image}")
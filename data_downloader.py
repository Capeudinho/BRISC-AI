import os
import re
import shutil
import kagglehub
import torch
import torchvision.transforms as transforms
from PIL import Image as img

os.chdir(os.path.dirname(os.path.abspath(__file__)))

shutil.rmtree("temporary", ignore_errors = True)
shutil.rmtree("archive", ignore_errors = True)
shutil.rmtree("data", ignore_errors = True)
os.makedirs("archive", exist_ok = True)
kagglehub.dataset_download("briscdataset/brisc2025", output_dir = "./temporary", force_download = True)
for task in ["classification", "segmentation"]:
	shutil.move(f"temporary/brisc2025/{task}_task", f"archive/{task}")
	shutil.move(f"archive/{task}/train", f"archive/{task}/training")
	shutil.move(f"archive/{task}/test", f"archive/{task}/validating")
	os.makedirs(f"archive/{task}/testing", exist_ok = True)
os.makedirs("archive/segmentation/testing/images", exist_ok = True)
os.makedirs("archive/segmentation/testing/masks", exist_ok = True)
shutil.rmtree("temporary")

for split in ["training", "validating"]:
	for tumor in ["glioma", "meningioma", "pituitary", "no_tumor"]:
		for image_name in os.listdir(f"archive/classification/{split}/{tumor}"):
			shutil.move(f"archive/classification/{split}/{tumor}/{image_name}", f"archive/classification/{split}/{image_name}")
		os.rmdir(f"archive/classification/{split}/{tumor}")
for split in ["training", "validating"]:
	for image_name in os.listdir(f"archive/classification/{split}"):
		if "_no_" in image_name:
			with open(f"archive/classification/{split}/{image_name}", "rb") as image, open(f"archive/segmentation/{split}/images/{image_name}", "wb") as new_image:
				new_image.write(image.read())
			with img.open(f"archive/classification/{split}/{image_name}") as image:
				new_mask = img.new("L", image.size, 0)
				new_mask.save(f"archive/segmentation/{split}/masks/{os.path.splitext(image_name)[0]}.png")
pattern = re.compile(r"brisc2025_(\w+)_(\d+)_(\w+)_(\w+)_t1\.jpg")
for task in ["segmentation", "classification"]:
	groups = {tumor: {view: [] for view in ["ax", "co", "sa"]} for tumor in ["gl", "me", "pi", "no"]}
	for image_name in os.listdir(f"archive/{task}/validating/images") if task == "segmentation" else os.listdir(f"archive/{task}/validating"):
		match = pattern.match(image_name)
		_, _, tumor, view = match.groups()
		groups[tumor][view].append(image_name)
	for group in groups.values():
		for image_names in group.values():
			image_names.sort()
			for image_name in image_names[:10]:
				if task == "segmentation":
					image_title = os.path.splitext(image_name)[0]
					shutil.move(f"archive/{task}/validating/images/{image_name}", f"archive/{task}/testing/images/{image_name}")
					shutil.move(f"archive/{task}/validating/masks/{image_title}.png", f"archive/{task}/testing/masks/{image_title}.png")
				elif task == "classification":
					shutil.move(f"archive/{task}/validating/{image_name}", f"archive/{task}/testing/{image_name}")

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
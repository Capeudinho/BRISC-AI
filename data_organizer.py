import os
import re
from PIL import Image as img

os.chdir(os.path.dirname(os.path.abspath(__file__)))
for task in ["classification", "segmentation"]:
	os.rename(f"archive/brisc2025/{task}_task", f"archive/{task}")
	os.rename(f"archive/{task}/train", f"archive/{task}/training")
	os.rename(f"archive/{task}/test", f"archive/{task}/validating")
	os.makedirs(f"archive/{task}/testing", exist_ok = True)
os.makedirs("archive/segmentation/testing/images", exist_ok = True)
os.makedirs("archive/segmentation/testing/masks", exist_ok = True)
os.rmdir("archive/brisc2025")
for split in ["training", "validating"]:
	for tumor in ["glioma", "meningioma", "pituitary", "no_tumor"]:
		for image_name in os.listdir(f"archive/classification/{split}/{tumor}"):
			os.rename(f"archive/classification/{split}/{tumor}/{image_name}", f"archive/classification/{split}/{image_name}")
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
					os.rename(f"archive/{task}/validating/images/{image_name}", f"archive/{task}/testing/images/{image_name}")
					os.rename(f"archive/{task}/validating/masks/{image_title}.png", f"archive/{task}/testing/masks/{image_title}.png")
				elif task == "classification":
					os.rename(f"archive/{task}/validating/{image_name}", f"archive/{task}/testing/{image_name}")
import os
import re
from PIL import Image as img

os.chdir(os.path.dirname(os.path.abspath(__file__)))
for task in ["classification", "segmentation"]:
	os.rename(f"../archive/brisc2025/{task}_task", f"../archive/{task}")
	os.rename(f"../archive/{task}/train", f"../archive/{task}/training")
	os.rename(f"../archive/{task}/test", f"../archive/{task}/validating")
	os.makedirs(f"../archive/{task}/testing", exist_ok = True)
os.makedirs("../archive/segmentation/testing/images", exist_ok = True)
os.makedirs("../archive/segmentation/testing/masks", exist_ok = True)
os.rmdir("../archive/brisc2025")
for split in ["training", "validating"]:
	for tumor in ["glioma", "meningioma", "pituitary", "no_tumor"]:
		for image in os.listdir(f"../archive/classification/{split}/{tumor}"):
			os.rename(f"../archive/classification/{split}/{tumor}/{image}", f"../archive/classification/{split}/{image}")
		os.rmdir(f"../archive/classification/{split}/{tumor}")
for split in ["training", "validating"]:
	for image in os.listdir(f"../archive/classification/{split}"):
		if "_no_" in image:
			with open(f"../archive/classification/{split}/{image}", "rb") as source, open(f"../archive/segmentation/{split}/images/{image}", "wb") as target:
				target.write(source.read())
			with img.open(f"../archive/classification/{split}/{image}") as image_file:
				new_image_file = img.new("L", image_file.size, 0)
				new_image_file.save(f"../archive/segmentation/{split}/masks/{os.path.splitext(image)[0]}.png")
pattern = re.compile(r"brisc2025_(\w+)_(\d+)_(\w+)_(\w+)_t1\.jpg")
for task in ["segmentation", "classification"]:
	groups = {tumor: {view: [] for view in ["ax", "co", "sa"]} for tumor in ["gl", "me", "pi", "no"]}
	for image in os.listdir(f"../archive/{task}/validating/images") if task == "segmentation" else os.listdir(f"../archive/{task}/validating"):
		match = pattern.match(image)
		_, _, tumor, view = match.groups()
		groups[tumor][view].append(image)
	for group in groups.values():
		for images in group.values():
			images.sort()
			for image in images[:10]:
				if task == "segmentation":
					name = os.path.splitext(image)[0]
					os.rename(f"../archive/{task}/validating/images/{image}", f"../archive/{task}/testing/images/{image}")
					os.rename(f"../archive/{task}/validating/masks/{name}.png", f"../archive/{task}/testing/masks/{name}.png")
				elif task == "classification":
					os.rename(f"../archive/{task}/validating/{image}", f"../archive/{task}/testing/{image}")
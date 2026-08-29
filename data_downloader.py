import os
import shutil
import kagglehub

os.chdir(os.path.dirname(os.path.abspath(__file__)))
kagglehub.dataset_download("briscdataset/brisc2025", output_dir = "./temporary", force_download = True)
os.makedirs("archive", exist_ok = True)
for task in ["classification", "segmentation"]:
	os.rename(f"temporary/brisc2025/{task}_task", f"archive/{task}")
	os.rename(f"archive/{task}/train", f"archive/{task}/training")
	os.rename(f"archive/{task}/test", f"archive/{task}/validating")
	os.makedirs(f"archive/{task}/testing", exist_ok = True)
os.makedirs("archive/segmentation/testing/images", exist_ok = True)
os.makedirs("archive/segmentation/testing/masks", exist_ok = True)
shutil.rmtree("temporary")
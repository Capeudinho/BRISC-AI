# BRISC-AI

This project trains, and evaluates U-Net models for brain tumor image segmentation. It downloads the BRISC 2025 dataset, prepares the images, and masks, trains a custom U-Net, fine-tunes a pretrained U-Net, and compares their segmentation performance using the Dice coefficient accuracy.

## Project Workflow

Run the scripts in the following order.

### `data_downloader.py`

Downloads the BRISC 2025 dataset from `briscdataset/brisc2025` in Kaggle, reorganizes it into the `archive` directory, and formats it into the `data` directory.

### `segmentation_trainer.py`

Trains the project's custom U-Net model using the segmentation dataset. The resulting model weights are saved in the `weights/` directory.

### `segmentation_tester.py`

Evaluates the trained custom U-Net model using the testing dataset. The measured macs, parameters, and Dice coefficient accuracy are saved in the `logs/` directory.

### `finetuned_segmentation_trainer.py`

Fine-tunes a pretrained U-Net from `mateuszbuda/brain-segmentation-pytorch` in GitHub using the segmentation dataset. The resulting fine-tuned model weights are saved in the `weights/` directory.

### `finetuned_segmentation_tester.py`

Evaluates the fine-tuned U-Net model using the testing dataset. The measured macs, parameters, and Dice coefficient accuracy are saved in the `logs/` directory.

### `logs_plotter.py`

Creates a scatter plot comparing the GMacs, and Dice coefficient accuracy of the tested models.
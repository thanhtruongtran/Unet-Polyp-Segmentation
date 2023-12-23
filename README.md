# Unet-Polyp-Segmentation

Link to tournament: https://www.kaggle.com/competitions/bkai-igh-neopolyp/overview

# Test model

## Instructions

1. Add a new code cell in the Kaggle notebook.
2. Copy and paste the following code into the cell:

```python
import requests
import os

# Replace YOUR_DRIVE_URL with the direct link to the Google Drive file
drive_url = 'YOUR_DRIVE_URL'
''''
https://drive.google.com/u/0/uc?id=11Nr0q0HbuXdbLhpoTgXImlvzEuW5pyVA&export=download&confirm=t&uuid=dadd5b8e-5340-4d69-ace8-6c1b6d6ea2fc&at=AB6BwCAyKZQsBPE5gapIHUG6JD_D:1700062251798
'''
# Directory where the downloaded file will be saved
save_dir = '/kaggle/working/'

# Send a GET request to the drive_url
response = requests.get(drive_url)

# Write the content of the response to a file in the save_dir
with open(os.path.join(save_dir, 'submission.pth'), 'wb') as f:
    f.write(response.content)
```

3. Run

```python
!git clone https://github.com/tttruong0812/Unet-Polyp-Segmentation.git # clone my git repo
```

```python
!mkdir output_masks_directory
```

```python
!python /kaggle/working/Unet-Polyp-Segmentation/infer.py --checkpoint '/kaggle/working/submission.pth' --test_images_directory '/kaggle/input/bkai-igh-neopolyp/test/test' --output_masks_directory '/kaggle/working/output_masks_directory'

# parse args checkpoint, test_images_directory(please add data of competition), output_masks_directory
```


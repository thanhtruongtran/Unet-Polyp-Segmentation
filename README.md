# Unet-Polyp-Segmentation

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
[https://drive.google.com/u/0/uc?id=1rE66914xj9HfNXFHGjtMxMq--Hbk3A69&export=download&confirm=t&uuid=2b4102a9\
    -5972-416b-97eb-88ba28ee326d&at=AB6BwCAGuaEHjfdCyfAwGaV0E-O9:1700047389408'](https://drive.google.com/u/0/uc?id=11Nr0q0HbuXdbLhpoTgXImlvzEuW5pyVA&export=download&confirm=t&uuid=dadd5b8e-5340-4d69-ace8-6c1b6d6ea2fc&at=AB6BwCAyKZQsBPE5gapIHUG6JD_D:1700062251798)
'''
# Directory where the downloaded file will be saved
save_dir = '/kaggle/working/'

# Send a GET request to the drive_url
response = requests.get(drive_url)

# Write the content of the response to a file in the save_dir
with open(os.path.join(save_dir, 'submissions.pth'), 'wb') as f:
    f.write(response.content)


```python

!git clone https://github.com/hdd0510/BKAI_Polyp.git # clone my git repo

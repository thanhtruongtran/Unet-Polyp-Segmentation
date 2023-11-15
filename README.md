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

# Directory where the downloaded file will be saved
save_dir = '/kaggle/working/'

# Send a GET request to the drive_url
response = requests.get(drive_url)

# Write the content of the response to a file in the save_dir
with open(os.path.join(save_dir, 'model.pth'), 'wb') as f:
    f.write(response.content)

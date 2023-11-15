import numpy as np
import pandas as pd
import cv2
import torch
import os
import segmentation_models_pytorch as smp

from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
import argparse

parser = argparse.ArgumentParser(description='Run inference with the deep learning model.')
parser.add_argument('--test_images_directory', type=str, help='Path to the images test data')
parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint')
parser.add_argument('--output_masks_directory', type=str, help='Path to the predicted masks ')
args = parser.parse_args()

model = smp.UnetPlusPlus(
    encoder_name="resnet50",        
    encoder_weights="imagenet",     
    in_channels=3,                  
    classes=3     
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model'])

ori_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

trainsize = 256
color_dict= {0: (0, 0, 0),
             1: (255, 0, 0),
             2: (0, 255, 0)}

def mask_to_rgb(mask, color_dict):
    output = np.zeros((mask.shape[0], mask.shape[1], 3))
    for k in color_dict.keys():
        output[mask==k] = color_dict[k]

    return np.uint8(output)    

model.eval()
test_images_directory = "/kaggle/input/bkai-igh-neopolyp/test/test"
output_masks_directory = "test_mask/"
output_overlaps_directory = "test_overlapmask/"

for filename in os.listdir(test_images_directory):
    image_path = os.path.join(test_images_directory, filename)
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_width = original_image.shape[0]
    original_height = original_image.shape[1]
    
    resized_image = cv2.resize(original_image, (trainsize, trainsize))
    transformed = ori_transform(image=resized_image)
    model_input = transformed["image"]
    model_input = model_input.unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_mask = model(model_input).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    
    scaled_mask = cv2.resize(predicted_mask, (original_height, original_width))
    class_mask = np.argmax(scaled_mask, axis=2)
    colorized_mask = np.zeros((*class_mask.shape, 3)).astype(np.uint8)
    
    # Assuming mask_to_rgb is a predefined function that converts class masks to RGB images
    rgb_colored_mask = mask_to_rgb(class_mask, color_dict)
    rgb_colored_mask_corrected = cv2.cvtColor(rgb_colored_mask, cv2.COLOR_BGR2RGB)
    
    combined_image = 0.7 * original_image + 0.3 * rgb_colored_mask_corrected
    combined_image = combined_image.astype('uint8')
    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
    
    rgb_colored_mask = cv2.cvtColor(rgb_colored_mask, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_masks_directory, filename), rgb_colored_mask)
    cv2.imwrite(os.path.join(output_overlaps_directory, filename), combined_image)
    
    print("Processed image:", image_path)

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 225] = 255
    pixels[pixels <= 225] = 0
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def rle2mask(mask_rle, shape=(3,3)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
#         print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r


MASK_DIR_PATH = args.output_masks_directory
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']

df.to_csv(r'output.csv', index=False)

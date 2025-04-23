import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# Directories
input_dir = 'dataset_120_real'
output_dir = '3. Segmentation and Detection Models/U-Net/predictions'
droplet_output_dir = '3. Segmentation and Detection Models/U-Net/droplet_unet'
os.makedirs(droplet_output_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Define the U-Net model
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(num_classes=3).to(device)
model_path = '3. Segmentation and Detection models/U-Net/unet_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image preprocessing 
transform = transforms.Compose([
    transforms.Resize((180, 240)),
    transforms.ToTensor()
])

# List image files
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))])

for file in tqdm(image_files, desc="Predicting"):
    image_path = os.path.join(input_dir, file)
    image = Image.open(image_path).convert('RGB')

    # Prepare for inference
    input_tensor = transform(image).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    with torch.no_grad():
        output = model(input_tensor)  # [1, 3, H, W]
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Create a 3-class grayscale mask
    class_to_gray = {0: 0, 1: 127, 2: 255}
    visual_mask = np.vectorize(class_to_gray.get)(prediction).astype(np.uint8)
    mask_image = Image.fromarray(visual_mask)

    # Create droplet-only mask (class 1)
    droplet_mask = (prediction == 1).astype(np.uint8) * 255
    droplet_mask_img = Image.fromarray(droplet_mask)

    # Upsample to 720×540 using nearest neighbor
    mask_image = mask_image.resize((720, 540), resample=Image.NEAREST)
    droplet_mask_img = droplet_mask_img.resize((720, 540), resample=Image.NEAREST)

    # Now save both upscaled masks
    mask_image.save(os.path.join(output_dir, file))
    droplet_mask_img.save(os.path.join(droplet_output_dir, file))

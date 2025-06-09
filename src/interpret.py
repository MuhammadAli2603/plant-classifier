import torch
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam
from torchvision import transforms
from PIL import Image
import numpy as np

# Setup Grad-CAM
gradcam = LayerGradCam(model, target_layer=model.layer4[-1])

# Preprocessing for input
prep = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

def show_cam(img_path, class_idx):
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_tensor = prep(img).unsqueeze(0).to(device)

    # Generate Grad-CAM mask
    mask = gradcam.attribute(img_tensor, target=class_idx).squeeze().cpu().detach().numpy()

    # Plot image with Grad-CAM overlay
    plt.imshow(img)
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()

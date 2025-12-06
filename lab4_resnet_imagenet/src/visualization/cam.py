
import torch
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

class CamGenerator:
    def __init__(self, device):
        self.device = device
        
    def get_target_layer(self, model, model_name):
        model_name = model_name.lower()
        if model_name == 'resnet50':
            return [model.layer4[-1]]
        elif model_name == 'vgg16':
            # Last conv layer in features
            return [model.features[-1]]
        elif model_name == 'alexnet':
            # Last conv layer is at index 10
            return [model.features[10]]
        else:
            raise ValueError(f"Unknown model for CAM: {model_name}")

    def generate_comparison(self, models_dict, dataset, image_idx, output_path):
        """
        models_dict: {'alexnet': model, ...}
        dataset: val_dataset (to get raw image)
        """
        img_tensor, label = dataset[image_idx] # Tensor: 3x224x224
        
        # Denormalize for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        # Prepare input tensor
        input_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Prepare RGB image (H, W, C) range [0, 1]
        rgb_img = img_tensor.permute(1, 2, 0).numpy() * std + mean
        rgb_img = np.clip(rgb_img, 0, 1)
        
        # Setup Plot
        fig, axes = plt.subplots(1, len(models_dict) + 1, figsize=(20, 5))
        fig.suptitle(f"Grad-CAM Comparison (Class: {dataset.classes[label]})", fontsize=16, y=1.05)
        
        # Original
        axes[0].imshow(rgb_img)
        axes[0].set_title("Original Image", fontsize=14)
        axes[0].axis('off')
        
        for i, (name, model) in enumerate(models_dict.items()):
            model = model.to(self.device)
            model.eval()
            target_layers = self.get_target_layer(model, name)
            
            # CRITICAL FIX: Enable gradients for all parameters to ensure backward pass works for CAM
            # (pytorch-grad-cam requires gradients to flow to the target layer)
            for param in model.parameters():
                param.requires_grad = True
            
            cam = GradCAM(model=model, target_layers=target_layers)
            
            # Target: predicted class (or ground truth? prompt says "model attention", usually predicted)
            # Let's target the Highest Scoring Class
            targets = None 
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            axes[i+1].imshow(visualization)
            axes[i+1].set_title(f"{name.upper()}", fontsize=14)
            axes[i+1].axis('off')
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved CAM to {output_path}")



import torch
import torch.nn as nn
from torchvision.models import (
    alexnet, AlexNet_Weights,
    vgg16, VGG16_Weights,
    resnet50, ResNet50_Weights
)

class ModelFactory:
    @staticmethod
    def get_model(model_name: str, num_classes: int = 10, pretrained: bool = True):
        model_name = model_name.lower()
        
        if model_name == 'alexnet':
            weights = AlexNet_Weights.DEFAULT if pretrained else None
            model = alexnet(weights=weights)
            ModelFactory._freeze_all(model)
            # AlexNet classifier: (6): Linear(in_features=4096, out_features=1000, bias=True)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            
        elif model_name == 'vgg16':
            weights = VGG16_Weights.DEFAULT if pretrained else None
            model = vgg16(weights=weights)
            ModelFactory._freeze_all(model)
            # VGG16 classifier: (6): Linear(in_features=4096, out_features=1000, bias=True)
            num_ftrs = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_ftrs, num_classes)
            
        elif model_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
            ModelFactory._freeze_all(model)
            # ResNet fc: Linear(in_features=2048, out_features=1000, bias=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            
        else:
            raise ValueError(f"Model {model_name} not supported.")
            
        return model

    @staticmethod
    def _freeze_all(model):
        for param in model.parameters():
            param.requires_grad = False

if __name__ == "__main__":
    # Test
    m = ModelFactory.get_model('resnet50')
    print(m.fc)

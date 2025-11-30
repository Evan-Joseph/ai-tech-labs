import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class VGGBlock(nn.Module):
    """VGG基本块"""
    
    def __init__(self, in_channels: int, out_channels: int, num_convs: int = 2):
        super(VGGBlock, self).__init__()
        
        layers = []
        for i in range(num_convs):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(2, 2))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGG_MNIST(nn.Module):
    """
    适配MNIST的小型VGG网络
    针对MNIST进行了以下修改：
    1. 输入通道数为1
    2. 减少网络深度，避免28x28输入经过过多池化层后尺寸过小
    3. 调整全连接层大小
    """
    
    def __init__(self, num_classes: int = 10):
        super(VGG_MNIST, self).__init__()
        
        # 特征提取部分 - 适配28x28输入
        self.features = nn.Sequential(
            # 输入: 1x28x28
            VGGBlock(1, 32, 2),      # -> 32x14x14
            VGGBlock(32, 64, 2),     # -> 64x7x7
            VGGBlock(64, 128, 1),    # -> 128x3x3 (最后一次池化)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        # 权重初始化
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Xavier权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)


class BasicBlock(nn.Module):
    """ResNet基本残差块"""
    
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18_MNIST(nn.Module):
    """
    适配MNIST的ResNet-18
    针对MNIST进行了以下修改：
    1. 输入通道数为1
    2. 第一层卷积改为3x3, stride=1，避免28x28输入丢失太多信息
    3. 移除初始的最大池化层
    """
    
    def __init__(self, num_classes: int = 10, block: nn.Module = BasicBlock):
        super(ResNet18_MNIST, self).__init__()
        
        self.in_channels = 64
        
        # 适配MNIST的初始卷积层
        # 标准: 7x7, stride=2 -> 改为 3x3, stride=1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 移除了标准ResNet的初始MaxPool2d，保留更多信息
        
        # ResNet-18有4个layer，每个layer包含[2, 2, 2, 2]个basic blocks
        self.layer1 = self._make_layer(block, 64, 2, stride=1)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 权重初始化
        self._initialize_weights()
    
    def _make_layer(self, block: nn.Module, out_channels: int, num_blocks: int, stride: int) -> nn.Module:
        """构建ResNet层"""
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入: 1x28x28
        out = F.relu(self.bn1(self.conv1(x)))
        # out: 64x28x28 (由于stride=1，没有缩小)
        
        out = self.layer1(out)   # -> 64x28x28
        out = self.layer2(out)   # -> 128x14x14
        out = self.layer3(out)   # -> 256x7x7
        out = self.layer4(out)   # -> 512x4x4
        
        out = self.avgpool(out)  # -> 512x1x1
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """获取中间层特征图用于可视化"""
        feature_maps = []
        
        # 第一层卷积后的特征图
        out = F.relu(self.bn1(self.conv1(x)))
        feature_maps.append(out.clone())
        
        # 各个layer后的特征图
        out = self.layer1(out)
        feature_maps.append(out.clone())
        
        out = self.layer2(out)
        feature_maps.append(out.clone())
        
        out = self.layer3(out)
        feature_maps.append(out.clone())
        
        out = self.layer4(out)
        feature_maps.append(out.clone())
        
        return feature_maps


class ModernCNNTrainer:
    """现代CNN训练器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.loss_history = []
        
    def train_step(self, images: torch.Tensor, labels: torch.Tensor, 
                   optimizer: torch.optim.Optimizer) -> float:
        """单步训练"""
        self.model.train()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        self.loss_history.append(loss.item())
        return loss.item()
    
    def evaluate(self, test_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total
        
        return accuracy / 100.0, avg_loss
    
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """预测"""
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.cpu()


if __name__ == "__main__":
    # 测试模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 测试VGG
    print("\n=== 测试 VGG_MNIST ===")
    vgg_model = VGG_MNIST()
    vgg_model = vgg_model.to(device)
    
    # 创建测试数据
    batch_size = 4
    test_input = torch.randn(batch_size, 1, 28, 28).to(device)
    
    output = vgg_model(test_input)
    print(f"VGG输入形状: {test_input.shape}")
    print(f"VGG输出形状: {output.shape}")
    
    # 计算参数量
    vgg_params = sum(p.numel() for p in vgg_model.parameters())
    print(f"VGG参数量: {vgg_params:,}")
    
    # 测试ResNet
    print("\n=== 测试 ResNet18_MNIST ===")
    resnet_model = ResNet18_MNIST()
    resnet_model = resnet_model.to(device)
    
    output = resnet_model(test_input)
    print(f"ResNet输入形状: {test_input.shape}")
    print(f"ResNet输出形状: {output.shape}")
    
    # 计算参数量
    resnet_params = sum(p.numel() for p in resnet_model.parameters())
    print(f"ResNet参数量: {resnet_params:,}")
    
    # 测试特征图提取
    feature_maps = resnet_model.get_feature_maps(test_input)
    print(f"ResNet特征图层数: {len(feature_maps)}")
    for i, feat in enumerate(feature_maps):
        print(f"  Layer {i+1}: {feat.shape}")
    
    # 测试训练器
    print("\n=== 测试训练器 ===")
    trainer = ModernCNNTrainer(resnet_model, device)
    
    # 模拟训练数据
    labels = torch.randint(0, 10, (batch_size,))
    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=0.001)
    
    loss = trainer.train_step(test_input, labels, optimizer)
    print(f"训练损失: {loss:.4f}")
    
    print("现代CNN模型测试完成！")

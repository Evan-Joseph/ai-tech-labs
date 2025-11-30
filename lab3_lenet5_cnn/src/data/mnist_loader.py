import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from typing import Tuple, Optional
import gzip
import os
import struct


class MNISTLoader:
    """MNIST数据集加载器，同时支持NumPy和PyTorch格式"""
    
    def __init__(self, data_dir: Optional[str] = None, normalize: bool = True):
        """
        初始化MNIST加载器
        
        Args:
            data_dir: 数据目录，如果为None则使用torchvision下载
            normalize: 是否归一化到[0,1]
        """
        self.data_dir = data_dir
        self.normalize = normalize
        
        # 如果指定了数据目录，尝试从本地加载
        if data_dir and os.path.exists(data_dir):
            self._load_local_data()
        else:
            self._download_data()
    
    def _load_local_data(self) -> None:
        """从本地文件加载MNIST数据"""
        def load_mnist_images(filename: str) -> np.ndarray:
            with gzip.open(os.path.join(self.data_dir, filename), 'rb') as f:
                # 读取文件头
                magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
                if magic != 2051:
                    raise ValueError(f"Invalid magic number {magic} in {filename}")
                # 读取图像数据
                data = np.frombuffer(f.read(), dtype=np.uint8)
                data = data.reshape(num, rows, cols)
            return data.astype(np.float32) / 255.0 if self.normalize else data.astype(np.float32)
        
        def load_mnist_labels(filename: str) -> np.ndarray:
            with gzip.open(os.path.join(self.data_dir, filename), 'rb') as f:
                magic, num = struct.unpack('>II', f.read(8))
                if magic != 2049:
                    raise ValueError(f"Invalid magic number {magic} in {filename}")
                data = np.frombuffer(f.read(), dtype=np.uint8)
            return data.astype(np.int64)
        
        # 加载训练数据
        self.train_images = load_mnist_images('train-images-idx3-ubyte.gz')
        self.train_labels = load_mnist_labels('train-labels-idx1-ubyte.gz')
        
        # 加载测试数据
        self.test_images = load_mnist_images('t10k-images-idx3-ubyte.gz')
        self.test_labels = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    
    def _download_data(self) -> None:
        """使用torchvision下载MNIST数据"""
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # 下载数据
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # 转换为numpy数组
        self.train_images = train_dataset.data.numpy().astype(np.float32)
        self.train_labels = train_dataset.targets.numpy()
        
        self.test_images = test_dataset.data.numpy().astype(np.float32)
        self.test_labels = test_dataset.targets.numpy()
        
        # 归一化
        if self.normalize:
            self.train_images /= 255.0
            self.test_images /= 255.0
    
    def get_numpy_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        获取NumPy格式的数据
        
        Returns:
            train_x: (N, 1, 28, 28) 训练图像
            train_y: (N, 10) 训练标签 (one-hot)
            test_x: (N, 1, 28, 28) 测试图像
            test_y: (N, 10) 测试标签 (one-hot)
        """
        # 重塑为 (N, 1, 28, 28)
        train_x = self.train_images.reshape(-1, 1, 28, 28)
        test_x = self.test_images.reshape(-1, 1, 28, 28)
        
        # 转换为one-hot编码
        train_y = self._to_one_hot(self.train_labels, 10)
        test_y = self._to_one_hot(self.test_labels, 10)
        
        return train_x, train_y, test_x, test_y
    
    def get_torch_loaders(self, batch_size: int = 64, shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        获取PyTorch DataLoaders
        
        Args:
            batch_size: 批次大小
            shuffle: 是否打乱训练数据
            
        Returns:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
        """
        # 转换为PyTorch张量
        train_images = torch.FloatTensor(self.train_images).unsqueeze(1)  # 添加通道维度
        train_labels = torch.LongTensor(self.train_labels)
        
        test_images = torch.FloatTensor(self.test_images).unsqueeze(1)
        test_labels = torch.LongTensor(self.test_labels)
        
        # 创建数据集
        train_dataset = TensorDataset(train_images, train_labels)
        test_dataset = TensorDataset(test_images, test_labels)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader    
    def _to_one_hot(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """将标签转换为one-hot编码"""
        one_hot = np.zeros((len(labels), num_classes))
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot
    
    def get_data_info(self) -> dict:
        """获取数据集基本信息"""
        return {
            'train_samples': len(self.train_labels),
            'test_samples': len(self.test_labels),
            'image_shape': self.train_images.shape[1:],
            'num_classes': len(np.unique(self.train_labels)),
            'pixel_range': (self.train_images.min(), self.train_images.max())
        }


if __name__ == "__main__":
    # 测试数据加载器
    loader = MNISTLoader()
    
    # 测试NumPy格式
    train_x, train_y, test_x, test_y = loader.get_numpy_data()
    print(f"NumPy format:")
    print(f"Train X shape: {train_x.shape}")
    print(f"Train Y shape: {train_y.shape}")
    print(f"Test X shape: {test_x.shape}")
    print(f"Test Y shape: {test_y.shape}")
    
    # 测试PyTorch格式
    train_loader, test_loader = loader.get_torch_loaders(batch_size=32)
    print(f"\nPyTorch format:")
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Test loader batches: {len(test_loader)}")
    
    # 获取数据信息
    info = loader.get_data_info()
    print(f"\nData info: {info}")

"""
LSTM 模型

使用 PyTorch 实现 LSTM 时间序列预测。
支持 Apple M1 Pro MPS 加速。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class LSTMConfig:
    """LSTM 模型配置"""
    input_size: int = 1
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 1
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 100
    patience: int = 15  # 早停耐心值


class LSTMNetwork(nn.Module):
    """LSTM 网络结构"""
    
    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(config.hidden_size, config.output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        out = self.fc(lstm_out[:, -1, :])
        return out


class LSTMModel:
    """
    LSTM 时间序列预测模型
    
    支持 CPU, CUDA 和 MPS (Apple Silicon) 加速。
    
    Attributes:
        config: 模型配置
        model: PyTorch 网络
        device: 运行设备
    """
    
    def __init__(self, config: Optional[LSTMConfig] = None) -> None:
        """
        初始化 LSTM 模型
        
        Args:
            config: 模型配置，若为 None 则使用默认值
        """
        self.config = config or LSTMConfig()
        self.model: Optional[LSTMNetwork] = None
        self.device = self._get_device()
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
    def _get_device(self) -> torch.device:
        """获取可用的计算设备"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"✓ 使用 CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("✓ 使用 Apple MPS 加速")
        else:
            device = torch.device("cpu")
            print("✓ 使用 CPU")
        return device
    
    def _create_dataloaders(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """创建 DataLoader"""
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_t = torch.FloatTensor(X_val)
            y_val_t = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_t, y_val_t)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
            
        return train_loader, val_loader
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> "LSTMModel":
        """
        训练模型
        
        Args:
            X_train: 训练特征 (samples, seq_len, features)
            y_train: 训练目标 (samples, output_size)
            X_val: 验证特征
            y_val: 验证目标
            verbose: 是否打印训练信息
            
        Returns:
            self
        """
        # 确保输入形状正确
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
            
        # 更新配置
        self.config.input_size = X_train.shape[2]
        self.config.output_size = y_train.shape[1]
        
        # 创建模型
        self.model = LSTMNetwork(self.config).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        train_loader, val_loader = self._create_dataloaders(
            X_train, y_train, X_val, y_val
        )
        
        best_val_loss = np.inf
        patience_counter = 0
        best_state = None
        
        self.train_losses = []
        self.val_losses = []
        
        for epoch in range(self.config.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
            train_loss /= len(train_loader.dataset)
            self.train_losses.append(train_loss)
            
            # 验证阶段
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        outputs = self.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item() * X_batch.size(0)
                        
                val_loss /= len(val_loader.dataset)
                self.val_losses.append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f"⚠ 早停触发于 Epoch {epoch + 1}")
                    break
                    
            # 打印进度
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.config.epochs} - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                print(msg)
                
        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        print(f"✓ LSTM 训练完成，最佳验证损失: {best_val_loss:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入特征 (samples, seq_len, features)
            
        Returns:
            预测值数组
        """
        if self.model is None:
            raise ValueError("请先调用 fit() 训练模型")
            
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_t).cpu().numpy()
            
        return predictions.flatten()
    
    def get_training_history(self) -> Tuple[List[float], List[float]]:
        """
        获取训练历史
        
        Returns:
            (train_losses, val_losses)
        """
        return self.train_losses, self.val_losses


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 生成测试数据
    n_samples = 100
    seq_len = 12
    
    X = np.random.randn(n_samples, seq_len, 1).astype(np.float32)
    y = np.mean(X, axis=(1, 2)) + np.random.randn(n_samples) * 0.1
    
    # 划分
    split = int(n_samples * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # 训练
    config = LSTMConfig(epochs=50, patience=10)
    model = LSTMModel(config)
    model.fit(X_train, y_train, X_val, y_val)
    
    # 预测
    predictions = model.predict(X_val)
    print(f"\n预测值: {predictions[:5]}")
    print(f"真实值: {y_val[:5]}")

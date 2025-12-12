"""
Transformer 模型

使用 PyTorch 实现基础 Transformer 时间序列预测器。
探索模型，重点观察小样本下的过拟合现象。
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional, List
from dataclasses import dataclass
import math


@dataclass
class TransformerConfig:
    """Transformer 模型配置"""
    input_size: int = 1
    d_model: int = 64         # 模型维度
    nhead: int = 4            # 注意力头数
    num_encoder_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    output_size: int = 1
    max_seq_len: int = 100
    learning_rate: float = 0.001
    batch_size: int = 16
    epochs: int = 150
    patience: int = 20


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer("pe", pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerNetwork(nn.Module):
    """Transformer 网络结构"""
    
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        
        # 输入投影
        self.input_projection = nn.Linear(config.input_size, config.d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(
            config.d_model, 
            config.max_seq_len, 
            config.dropout
        )
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # 多层输出投影（避免模式崩溃）
        self.output_layers = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.output_size)
        )
        
        # 直接从输入的残差连接
        self.input_residual = nn.Linear(config.input_size, config.output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        batch_size = x.size(0)
        
        # 保存最后一个时间步的输入用于残差
        last_input = x[:, -1, :]  # (batch, features)
        
        # 转换为 (seq_len, batch, features)
        x = x.permute(1, 0, 2)
        
        # 投影到 d_model 维度
        x = self.input_projection(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer 编码
        x = self.transformer_encoder(x)
        
        # 取最后一个时间步
        x = x[-1, :, :]
        
        # 多层输出
        transformer_out = self.output_layers(x)
        
        # 加上残差连接（从输入直接预测）
        residual_out = self.input_residual(last_input)
        
        # 融合输出
        out = transformer_out + 0.5 * residual_out
        
        return out


class TransformerModel:
    """
    Transformer 时间序列预测模型
    
    探索模型，用于研究 Transformer 在小样本时间序列上的表现。
    
    Attributes:
        config: 模型配置
        model: PyTorch 网络
        device: 运行设备
    """
    
    def __init__(self, config: Optional[TransformerConfig] = None) -> None:
        """
        初始化 Transformer 模型
        
        Args:
            config: 模型配置
        """
        self.config = config or TransformerConfig()
        self.model: Optional[TransformerNetwork] = None
        self.device = self._get_device()
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
    def _get_device(self) -> torch.device:
        """获取可用的计算设备"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
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
    ) -> "TransformerModel":
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            verbose: 是否打印训练信息
            
        Returns:
            self
        """
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        if y_val is not None and y_val.ndim == 1:
            y_val = y_val.reshape(-1, 1)
            
        self.config.input_size = X_train.shape[2]
        self.config.output_size = y_train.shape[1]
        self.config.max_seq_len = X_train.shape[1]
        
        self.model = TransformerNetwork(self.config).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
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
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
            train_loss /= len(train_loader.dataset)
            self.train_losses.append(train_loss)
            
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
                
                scheduler.step(val_loss)
                
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
                    
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.config.epochs} - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                print(msg)
                
        if best_state is not None:
            self.model.load_state_dict(best_state)
            
        print(f"✓ Transformer 训练完成，最佳验证损失: {best_val_loss:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入特征
            
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
        """获取训练历史"""
        return self.train_losses, self.val_losses
    
    def count_parameters(self) -> int:
        """统计模型参数量"""
        if self.model is None:
            return 0
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    
    n_samples = 100
    seq_len = 12
    
    X = np.random.randn(n_samples, seq_len, 1).astype(np.float32)
    y = np.mean(X, axis=(1, 2)) + np.random.randn(n_samples) * 0.1
    
    split = int(n_samples * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    config = TransformerConfig(epochs=50, patience=10)
    model = TransformerModel(config)
    model.fit(X_train, y_train, X_val, y_val)
    
    print(f"\n模型参数量: {model.count_parameters():,}")
    
    predictions = model.predict(X_val)
    print(f"预测值: {predictions[:5]}")
    print(f"真实值: {y_val[:5]}")

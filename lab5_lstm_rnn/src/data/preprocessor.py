"""
时间序列预处理模块

提供 ADF 平稳性检验、差分、归一化、滑动窗口等功能。
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller


@dataclass
class ADFResult:
    """ADF 检验结果"""
    statistic: float
    p_value: float
    used_lags: int
    n_obs: int
    critical_values: dict
    is_stationary: bool  # p < 0.05 认为平稳
    
    def __str__(self) -> str:
        status = "✓ 平稳" if self.is_stationary else "✗ 非平稳"
        return (
            f"ADF 检验结果 ({status}):\n"
            f"  统计量: {self.statistic:.4f}\n"
            f"  p-value: {self.p_value:.4f}\n"
            f"  临界值: 1%={self.critical_values['1%']:.4f}, "
            f"5%={self.critical_values['5%']:.4f}, "
            f"10%={self.critical_values['10%']:.4f}"
        )


class TimeSeriesPreprocessor:
    """
    时间序列预处理器
    
    提供 ADF 检验、差分、归一化和滑动窗口构建等功能。
    
    Attributes:
        scaler: MinMax 归一化器
        diff_order: 差分阶数
    """
    
    def __init__(self) -> None:
        """初始化预处理器"""
        self.scaler: Optional[MinMaxScaler] = None
        self.diff_order: int = 0
        self._original_first_values: List[float] = []
        
    def adf_test(
        self, 
        series: pd.Series | np.ndarray,
        significance_level: float = 0.05
    ) -> ADFResult:
        """
        ADF 平稳性检验
        
        Args:
            series: 时间序列数据
            significance_level: 显著性水平，默认 0.05
            
        Returns:
            ADFResult 对象，包含检验结果
        """
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
            
        result = adfuller(series.dropna(), autolag="AIC")
        
        return ADFResult(
            statistic=result[0],
            p_value=result[1],
            used_lags=result[2],
            n_obs=result[3],
            critical_values=result[4],
            is_stationary=result[1] < significance_level
        )
    
    def difference(
        self, 
        series: pd.Series | pd.DataFrame,
        order: int = 1
    ) -> pd.DataFrame:
        """
        差分变换
        
        Args:
            series: 原始时间序列
            order: 差分阶数
            
        Returns:
            差分后的序列
        """
        if isinstance(series, pd.DataFrame):
            data = series.copy()
        else:
            data = pd.DataFrame({"value": series})
            
        self.diff_order = order
        self._original_first_values = []
        
        for _ in range(order):
            self._original_first_values.append(data.iloc[0].values[0])
            data = data.diff().dropna()
            
        return data
    
    def auto_difference(
        self, 
        series: pd.Series | pd.DataFrame,
        max_order: int = 2
    ) -> Tuple[pd.DataFrame, int]:
        """
        自动差分直到序列平稳
        
        Args:
            series: 原始时间序列
            max_order: 最大差分阶数
            
        Returns:
            (差分后的序列, 差分阶数)
        """
        if isinstance(series, pd.DataFrame):
            data = series.copy()
        else:
            data = pd.DataFrame({"value": series})
            
        self._original_first_values = []
        
        for order in range(max_order + 1):
            adf_result = self.adf_test(data)
            if adf_result.is_stationary:
                self.diff_order = order
                print(f"✓ 序列在 {order} 阶差分后平稳")
                return data, order
                
            if order < max_order:
                self._original_first_values.append(data.iloc[0].values[0])
                data = data.diff().dropna()
                
        self.diff_order = max_order
        print(f"⚠ 达到最大差分阶数 {max_order}，序列可能仍不平稳")
        return data, max_order
    
    def normalize(
        self, 
        data: np.ndarray | pd.DataFrame,
        feature_range: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        """
        Min-Max 归一化
        
        Args:
            data: 原始数据
            feature_range: 归一化范围
            
        Returns:
            归一化后的数据
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        self.scaler = MinMaxScaler(feature_range=feature_range)
        normalized = self.scaler.fit_transform(data)
        
        return normalized
    
    def inverse_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        反归一化
        
        Args:
            data: 归一化后的数据
            
        Returns:
            原始尺度的数据
        """
        if self.scaler is None:
            raise ValueError("请先调用 normalize() 进行归一化")
            
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        return self.scaler.inverse_transform(data)
    
    def create_sequences(
        self,
        data: np.ndarray,
        window_size: int = 12,
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用滑动窗口创建监督学习数据集
        
        Args:
            data: 归一化后的时间序列数据
            window_size: 输入窗口大小（历史步数）
            horizon: 预测步长
            
        Returns:
            (X, y) 训练数据对
            X shape: (samples, window_size, features)
            y shape: (samples, horizon)
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)
            
        X, y = [], []
        
        for i in range(len(data) - window_size - horizon + 1):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size:i + window_size + horizon, 0])
            
        return np.array(X), np.array(y)
    
    def train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_ratio: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        时间序列顺序划分（严禁随机打乱）
        
        Args:
            X: 输入特征
            y: 目标变量
            test_ratio: 测试集比例
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - test_ratio))
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"✓ 数据集划分完成")
        print(f"  训练集: {len(X_train)} 样本")
        print(f"  测试集: {len(X_test)} 样本")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 模拟时间序列
    t = np.arange(100)
    series = 50 + 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.randn(100) * 5
    series = pd.Series(series)
    
    preprocessor = TimeSeriesPreprocessor()
    
    # ADF 检验
    print("=== 原始序列 ADF 检验 ===")
    result = preprocessor.adf_test(series)
    print(result)
    
    # 自动差分
    print("\n=== 自动差分 ===")
    diff_series, order = preprocessor.auto_difference(series)
    
    # 归一化
    print("\n=== 归一化 ===")
    normalized = preprocessor.normalize(diff_series)
    print(f"归一化后范围: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    # 创建序列
    print("\n=== 滑动窗口 ===")
    X, y = preprocessor.create_sequences(normalized, window_size=12)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # 划分数据
    print("\n=== 数据划分 ===")
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(X, y)

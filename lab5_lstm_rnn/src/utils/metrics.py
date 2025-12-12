"""
评估指标模块

计算 MAE, RMSE, MAPE 等评估指标。
"""

import numpy as np
from typing import Dict


def calculate_metrics(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        包含 MAE, RMSE, MAPE 的字典
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # MAE: Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE: Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE: Mean Absolute Percentage Error
    # 避免除零
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.inf
        
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        model_name: 模型名称
    """
    print(f"\n{'='*40}")
    print(f"  {model_name} 评估指标")
    print(f"{'='*40}")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"{'='*40}")

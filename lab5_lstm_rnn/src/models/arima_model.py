"""
ARIMA 模型

使用 statsmodels 实现自动 ARIMA 参数选择。
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")


class ARIMAModel:
    """
    ARIMA 时间序列预测模型
    
    自动网格搜索最优 (p, d, q) 参数。
    
    Attributes:
        order: ARIMA (p, d, q) 参数
        model: 拟合后的模型对象
    """
    
    def __init__(
        self,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5
    ) -> None:
        """
        初始化 ARIMA 模型
        
        Args:
            max_p: AR 阶数最大值
            max_d: 差分阶数最大值
            max_q: MA 阶数最大值
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.order: Optional[Tuple[int, int, int]] = None
        self.model = None
        self._fitted_model = None
        
    def _find_d(self, series: pd.Series) -> int:
        """
        通过 ADF 检验确定最优差分阶数
        
        Args:
            series: 时间序列
            
        Returns:
            最优差分阶数
        """
        for d in range(self.max_d + 1):
            if d == 0:
                test_series = series
            else:
                test_series = series.diff(d).dropna()
                
            result = adfuller(test_series)
            if result[1] < 0.05:  # p-value < 0.05，平稳
                return d
                
        return self.max_d
    
    def _grid_search(
        self,
        train: pd.Series,
        d: int
    ) -> Tuple[int, int, int]:
        """
        网格搜索最优 (p, q) 参数
        
        Args:
            train: 训练数据
            d: 差分阶数
            
        Returns:
            最优 (p, d, q) 参数
        """
        best_aic = np.inf
        best_order = (1, d, 1)
        
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(train, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
                    
        return best_order
    
    def fit(
        self,
        train: pd.Series | np.ndarray,
        auto_order: bool = True,
        order: Optional[Tuple[int, int, int]] = None
    ) -> "ARIMAModel":
        """
        拟合模型
        
        Args:
            train: 训练数据
            auto_order: 是否自动搜索最优参数
            order: 手动指定 (p, d, q)，当 auto_order=False 时使用
            
        Returns:
            self
        """
        if isinstance(train, np.ndarray):
            train = pd.Series(train)
            
        if auto_order:
            print("正在搜索最优 ARIMA 参数...")
            d = self._find_d(train)
            self.order = self._grid_search(train, d)
            print(f"✓ 最优参数: ARIMA{self.order}")
        else:
            self.order = order if order else (1, 1, 1)
            
        self.model = ARIMA(train, order=self.order)
        self._fitted_model = self.model.fit()
        
        print(f"  AIC: {self._fitted_model.aic:.2f}")
        print(f"  BIC: {self._fitted_model.bic:.2f}")
        
        return self
    
    def predict(self, steps: int) -> np.ndarray:
        """
        预测未来值
        
        Args:
            steps: 预测步数
            
        Returns:
            预测值数组
        """
        if self._fitted_model is None:
            raise ValueError("请先调用 fit() 拟合模型")
            
        forecast = self._fitted_model.forecast(steps=steps)
        return np.array(forecast)
    
    def get_fitted_values(self) -> np.ndarray:
        """
        获取训练集上的拟合值
        
        Returns:
            拟合值数组
        """
        if self._fitted_model is None:
            raise ValueError("请先调用 fit() 拟合模型")
            
        return self._fitted_model.fittedvalues.values
    
    def summary(self) -> str:
        """获取模型摘要"""
        if self._fitted_model is None:
            return "模型未拟合"
        return str(self._fitted_model.summary())


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 生成测试数据
    t = np.arange(100)
    series = 50 + 0.3 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.randn(100) * 3
    series = pd.Series(series)
    
    train = series[:80]
    test = series[80:]
    
    model = ARIMAModel()
    model.fit(train)
    
    predictions = model.predict(len(test))
    print(f"\n预测值: {predictions[:5]}")
    print(f"真实值: {test.values[:5]}")

"""
Prophet 模型

使用 Facebook Prophet 进行时间序列预测。
"""

import numpy as np
import pandas as pd
from typing import Optional
from prophet import Prophet
import logging

# 抑制 Prophet 的日志输出
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetModel:
    """
    Prophet 时间序列预测模型
    
    Facebook 开发的加性模型，适合具有季节性的时间序列。
    
    Attributes:
        model: Prophet 模型对象
        forecast: 预测结果 DataFrame
    """
    
    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05
    ) -> None:
        """
        初始化 Prophet 模型
        
        Args:
            yearly_seasonality: 是否包含年度季节性
            weekly_seasonality: 是否包含周季节性
            daily_seasonality: 是否包含日季节性
            changepoint_prior_scale: 变点先验尺度
        """
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.model: Optional[Prophet] = None
        self.forecast: Optional[pd.DataFrame] = None
        self._train_df: Optional[pd.DataFrame] = None
        
    def _prepare_data(
        self,
        series: pd.Series | pd.DataFrame,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        准备 Prophet 格式的数据
        
        Args:
            series: 时间序列值
            dates: 日期索引
            
        Returns:
            Prophet 格式的 DataFrame (ds, y)
        """
        if isinstance(series, pd.DataFrame):
            values = series.iloc[:, 0].values
            dates = series.index if dates is None else dates
        else:
            values = series.values
            dates = series.index if dates is None else dates
            
        return pd.DataFrame({
            "ds": pd.to_datetime(dates),
            "y": values
        })
        
    def fit(
        self,
        train: pd.Series | pd.DataFrame,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> "ProphetModel":
        """
        拟合模型
        
        Args:
            train: 训练数据
            dates: 日期索引
            
        Returns:
            self
        """
        self._train_df = self._prepare_data(train, dates)
        
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale
        )
        
        self.model.fit(self._train_df)
        print(f"✓ Prophet 模型拟合完成")
        print(f"  训练样本数: {len(self._train_df)}")
        
        return self
    
    def predict(
        self,
        steps: int,
        freq: str = "MS"
    ) -> np.ndarray:
        """
        预测未来值
        
        Args:
            steps: 预测步数
            freq: 预测频率
            
        Returns:
            预测值数组
        """
        if self.model is None:
            raise ValueError("请先调用 fit() 拟合模型")
            
        # 创建未来日期
        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        self.forecast = self.model.predict(future)
        
        # 返回预测部分
        predictions = self.forecast["yhat"].values[-steps:]
        return predictions
    
    def get_fitted_values(self) -> np.ndarray:
        """
        获取训练集上的拟合值
        
        Returns:
            拟合值数组
        """
        if self.model is None:
            raise ValueError("请先调用 fit() 拟合模型")
            
        if self.forecast is None:
            future = self.model.make_future_dataframe(periods=0)
            self.forecast = self.model.predict(future)
            
        train_len = len(self._train_df)
        return self.forecast["yhat"].values[:train_len]
    
    def get_components(self) -> pd.DataFrame:
        """
        获取分解组件（趋势、季节性等）
        
        Returns:
            包含各组件的 DataFrame
        """
        if self.forecast is None:
            raise ValueError("请先调用 predict() 进行预测")
            
        components = ["ds", "trend"]
        if self.yearly_seasonality:
            components.append("yearly")
        if self.weekly_seasonality:
            components.append("weekly")
            
        return self.forecast[components]


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)
    
    # 生成测试数据
    dates = pd.date_range("2018-01", periods=60, freq="MS")
    values = 50 + 0.3 * np.arange(60) + 10 * np.sin(2 * np.pi * np.arange(60) / 12) + np.random.randn(60) * 3
    series = pd.DataFrame({"count": values}, index=dates)
    
    train = series[:48]
    test = series[48:]
    
    model = ProphetModel()
    model.fit(train)
    
    predictions = model.predict(len(test))
    print(f"\n预测值: {predictions}")
    print(f"真实值: {test['count'].values}")

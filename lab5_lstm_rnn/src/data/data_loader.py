"""
数据加载与聚合模块

负责从原始 CSV 文件加载土地发证记录，并按月聚合为时间序列。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


class DataLoader:
    """
    土地发证数据加载器
    
    负责加载原始 CSV 数据并按月聚合为时间序列。
    
    Attributes:
        data_path: 原始数据文件路径
        encoding: 文件编码格式
    """
    
    def __init__(
        self, 
        data_path: str | Path,
        encoding: str = "gb18030"
    ) -> None:
        """
        初始化数据加载器
        
        Args:
            data_path: 原始 CSV 文件路径
            encoding: 文件编码，默认为 gb18030
        """
        self.data_path = Path(data_path)
        self.encoding = encoding
        self._raw_df: Optional[pd.DataFrame] = None
        self._monthly_series: Optional[pd.Series] = None
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        加载原始数据
        
        Returns:
            包含所有原始记录的 DataFrame
            
        Raises:
            FileNotFoundError: 当数据文件不存在时
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
            
        self._raw_df = pd.read_csv(
            self.data_path,
            encoding=self.encoding,
            parse_dates=["发证日期"]
        )
        
        print(f"✓ 成功加载 {len(self._raw_df)} 条原始记录")
        print(f"  时间范围: {self._raw_df['发证日期'].min()} ~ {self._raw_df['发证日期'].max()}")
        
        return self._raw_df
    
    def aggregate_monthly(self) -> pd.DataFrame:
        """
        按月聚合发证数量
        
        Returns:
            包含月度发证数量的 DataFrame，索引为日期（月初）
        """
        if self._raw_df is None:
            self.load_raw_data()
            
        # 提取年月
        df = self._raw_df.copy()
        df["year_month"] = df["发证日期"].dt.to_period("M")
        
        # 按月统计
        monthly_counts = df.groupby("year_month").size()
        
        # 转换为 DataFrame，索引为日期
        self._monthly_series = pd.DataFrame({
            "count": monthly_counts.values
        }, index=monthly_counts.index.to_timestamp())
        
        self._monthly_series.index.name = "date"
        
        print(f"✓ 生成 {len(self._monthly_series)} 个月的时间序列")
        print(f"  月均发证数量: {self._monthly_series['count'].mean():.1f}")
        print(f"  标准差: {self._monthly_series['count'].std():.1f}")
        
        return self._monthly_series
    
    def get_full_monthly_series(self) -> pd.DataFrame:
        """
        获取完整的月度时间序列（填充缺失月份为 0）
        
        Returns:
            完整的月度时间序列 DataFrame
        """
        if self._monthly_series is None:
            self.aggregate_monthly()
            
        # 创建完整的月度索引
        full_range = pd.date_range(
            start=self._monthly_series.index.min(),
            end=self._monthly_series.index.max(),
            freq="MS"  # Month Start
        )
        
        # 重索引，填充缺失值为 0
        full_series = self._monthly_series.reindex(full_range, fill_value=0)
        full_series.index.name = "date"
        
        print(f"✓ 完整时间序列包含 {len(full_series)} 个月")
        
        return full_series
    
    def get_statistics(self) -> dict:
        """
        获取数据统计信息
        
        Returns:
            包含各类统计指标的字典
        """
        if self._raw_df is None:
            self.load_raw_data()
            
        if self._monthly_series is None:
            self.aggregate_monthly()
            
        stats = {
            "total_records": len(self._raw_df),
            "date_range": (
                self._raw_df["发证日期"].min().strftime("%Y-%m-%d"),
                self._raw_df["发证日期"].max().strftime("%Y-%m-%d")
            ),
            "num_months": len(self._monthly_series),
            "monthly_mean": self._monthly_series["count"].mean(),
            "monthly_std": self._monthly_series["count"].std(),
            "monthly_min": self._monthly_series["count"].min(),
            "monthly_max": self._monthly_series["count"].max(),
            "land_use_types": self._raw_df["用途"].value_counts().to_dict()
        }
        
        return stats


if __name__ == "__main__":
    # 测试代码
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "国有土地发证记录.csv"
    loader = DataLoader(data_path)
    
    # 加载并聚合
    raw_df = loader.load_raw_data()
    monthly_df = loader.get_full_monthly_series()
    
    print("\n--- 数据统计 ---")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")

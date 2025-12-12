"""
时间序列可视化模块

提供统一的绘图接口，支持中英文字体混排。
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any


class TimeSeriesPlotter:
    """
    时间序列绑图器
    
    配置中文 (Songti SC) 和英文 (Times New Roman) 字体混排。
    """
    
    # 颜色方案
    COLORS = {
        "primary": "#2E86AB",      # 蓝色
        "secondary": "#A23B72",    # 紫红色
        "success": "#31A354",      # 绿色
        "warning": "#F18F01",      # 橙色
        "danger": "#E63946",       # 红色
        "gray": "#6C757D",         # 灰色
    }
    
    MODEL_COLORS = {
        "ARIMA": "#2E86AB",
        "Prophet": "#A23B72",
        "LSTM": "#31A354",
        "GRU": "#F18F01",
        "Transformer": "#E63946",
        "Actual": "#1D3557",
    }
    
    def __init__(
        self,
        figure_dir: str | Path = "assets/figures",
        dpi: int = 150,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        初始化绑图器
        
        Args:
            figure_dir: 图片保存目录
            dpi: 图片分辨率
            figsize: 默认图片尺寸
        """
        self.figure_dir = Path(figure_dir)
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        
        self._setup_fonts()
        
    def _setup_fonts(self) -> None:
        """配置中英文字体"""
        plt.rcParams.update({
            # 中文字体
            "font.family": ["Songti SC", "Times New Roman", "DejaVu Sans"],
            "font.sans-serif": ["Songti SC", "SimHei", "DejaVu Sans"],
            # 数学字体
            "mathtext.fontset": "stix",
            # 负号显示
            "axes.unicode_minus": False,
            # 默认尺寸
            "figure.figsize": self.figsize,
            "figure.dpi": self.dpi,
            # 线条样式
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            # 网格样式
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            # 图例样式
            "legend.framealpha": 0.9,
            "legend.edgecolor": "gray",
        })
        
    def plot_time_series(
        self,
        series: pd.DataFrame | pd.Series,
        title: str = "时间序列趋势图",
        xlabel: str = "日期",
        ylabel: str = "发证数量",
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        绑制时间序列趋势图
        
        Args:
            series: 时间序列数据
            title: 图片标题
            xlabel: X轴标签
            ylabel: Y轴标签
            filename: 保存文件名（不含扩展名）
            show: 是否显示图片
            
        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if isinstance(series, pd.DataFrame):
            values = series.iloc[:, 0]
            index = series.index
        else:
            values = series.values
            index = series.index
            
        ax.plot(index, values, color=self.COLORS["primary"], linewidth=1.5)
        ax.fill_between(index, values, alpha=0.2, color=self.COLORS["primary"])
        
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        plt.tight_layout()
        
        if filename:
            save_path = self.figure_dir / f"{filename}.pdf"
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"✓ 图片已保存: {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_predictions_comparison(
        self,
        actual: np.ndarray,
        predictions: Dict[str, np.ndarray],
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "模型预测对比",
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        绑制多模型预测对比图
        
        Args:
            actual: 真实值
            predictions: 模型名 -> 预测值 的字典
            dates: 日期索引
            title: 图片标题
            filename: 保存文件名
            show: 是否显示
            
        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = dates if dates is not None else np.arange(len(actual))
        
        # 绘制真实值
        ax.plot(x, actual, label="实际值 (Actual)", 
                color=self.MODEL_COLORS["Actual"],
                linewidth=2.5, zorder=10)
        
        # 绘制各模型预测
        for model_name, pred in predictions.items():
            color = self.MODEL_COLORS.get(model_name, self.COLORS["gray"])
            ax.plot(x, pred, label=model_name, color=color, 
                    linewidth=1.5, linestyle="--", alpha=0.8)
            
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("发证数量", fontsize=12)
        ax.legend(loc="upper left", fontsize=10)
        
        plt.tight_layout()
        
        if filename:
            save_path = self.figure_dir / f"{filename}.pdf"
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"✓ 图片已保存: {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        title: str = "训练损失曲线",
        model_name: str = "Model",
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        绘制训练/验证损失曲线
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            title: 图片标题
            model_name: 模型名称
            filename: 保存文件名
            show: 是否显示
            
        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = np.arange(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, label="训练损失 (Train Loss)",
                color=self.COLORS["primary"], linewidth=1.5)
        
        if val_losses is not None:
            ax.plot(epochs, val_losses, label="验证损失 (Val Loss)",
                    color=self.COLORS["warning"], linewidth=1.5)
            
        ax.set_title(f"{title} - {model_name}", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (MSE)", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)
        
        # 设置y轴从0开始
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if filename:
            save_path = self.figure_dir / f"{filename}.pdf"
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"✓ 图片已保存: {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def plot_acf_pacf(
        self,
        series: pd.Series | np.ndarray,
        lags: int = 40,
        title: str = "自相关与偏自相关分析",
        filename: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        绘制 ACF 和 PACF 图
        
        Args:
            series: 时间序列
            lags: 滞后阶数
            title: 图片标题
            filename: 保存文件名
            show: 是否显示
            
        Returns:
            matplotlib Figure 对象
        """
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        plot_acf(series, lags=lags, ax=ax1, alpha=0.05)
        ax1.set_title("自相关函数 (ACF)", fontsize=12, fontweight="bold")
        
        plot_pacf(series, lags=lags, ax=ax2, alpha=0.05)
        ax2.set_title("偏自相关函数 (PACF)", fontsize=12, fontweight="bold")
        
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
        
        plt.tight_layout()
        
        if filename:
            save_path = self.figure_dir / f"{filename}.pdf"
            fig.savefig(save_path, dpi=self.dpi, bbox_inches="tight")
            print(f"✓ 图片已保存: {save_path}")
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return fig
    
    def create_metrics_table(
        self,
        metrics: Dict[str, Dict[str, float]],
        filename: Optional[str] = None
    ) -> pd.DataFrame:
        """
        创建评估指标表格
        
        Args:
            metrics: 模型名 -> {指标名: 值} 的嵌套字典
            filename: 保存的 LaTeX 文件名
            
        Returns:
            DataFrame 格式的指标表
        """
        df = pd.DataFrame(metrics).T
        df.index.name = "模型"
        
        # 排序：按 RMSE 升序
        if "RMSE" in df.columns:
            df = df.sort_values("RMSE")
            
        if filename:
            save_path = self.figure_dir.parent / "tables" / f"{filename}.tex"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 格式化为 LaTeX 表格
            latex_table = df.to_latex(
                float_format=lambda x: f"{x:.4f}",
                caption="各模型评估指标对比",
                label="tab:metrics",
                position="htbp"
            )
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(latex_table)
                
            print(f"✓ LaTeX 表格已保存: {save_path}")
            
        return df


if __name__ == "__main__":
    # 测试代码
    import numpy as np
    
    plotter = TimeSeriesPlotter()
    
    # 测试时间序列图
    dates = pd.date_range("2020-01", periods=50, freq="MS")
    values = 30 + 10 * np.sin(np.arange(50) * 0.5) + np.random.randn(50) * 3
    series = pd.DataFrame({"count": values}, index=dates)
    
    plotter.plot_time_series(series, title="测试时间序列", show=True)

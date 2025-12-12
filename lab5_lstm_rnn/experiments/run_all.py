"""
实验一键执行脚本

运行所有模型的训练、评估和可视化。
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import warnings
from typing import Dict, Any

warnings.filterwarnings("ignore")

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# 导入自定义模块
from src.data.data_loader import DataLoader
from src.data.preprocessor import TimeSeriesPreprocessor
from src.models.arima_model import ARIMAModel
from src.models.prophet_model import ProphetModel
from src.models.lstm_model import LSTMModel, LSTMConfig
from src.models.gru_model import GRUModel, GRUConfig
from src.models.transformer_model import TransformerModel, TransformerConfig
from src.visualization.plotter import TimeSeriesPlotter
from src.utils.metrics import calculate_metrics, print_metrics


# ==================== 配置 ====================
CONFIG = {
    # 数据配置
    "data_path": PROJECT_ROOT / "data" / "raw" / "国有土地发证记录.csv",
    "output_dir": PROJECT_ROOT / "assets",
    
    # 滑动窗口配置
    "window_size": 6,   # 6个月的历史数据（增加训练样本）
    "horizon": 1,       # 预测1个月
    "test_ratio": 0.2,  # 20% 作为测试集
    
    # 深度学习模型配置（优化避免模式崩溃）
    "lstm": LSTMConfig(
        hidden_size=64,      # 增大隐藏层
        num_layers=1,        # 减少层数避免过拟合
        dropout=0.1,         # 降低 dropout
        epochs=200,          # 增加训练轮数
        patience=25,         # 增加早停耐心
        batch_size=4,        # 减小 batch size
        learning_rate=0.005  # 增大学习率
    ),
    "gru": GRUConfig(
        hidden_size=64,
        num_layers=1,
        dropout=0.1,
        epochs=200,
        patience=25,
        batch_size=4,
        learning_rate=0.005
    ),
    "transformer": TransformerConfig(
        d_model=32,
        nhead=4,
        num_encoder_layers=1,    # 减少到1层避免过拟合
        dim_feedforward=64,
        dropout=0.2,             # 增加 dropout
        epochs=300,              # 增加训练轮数
        patience=40,             # 增加早停耐心
        batch_size=8,            # 增大 batch size 稳定训练
        learning_rate=0.001      # 使用更低的学习率
    )
}


def step1_data_engineering() -> Dict[str, Any]:
    """
    Step 1: 数据工程
    
    - 加载原始数据
    - 按月聚合
    - ADF 平稳性检验
    - 归一化
    - 滑动窗口构建
    """
    print("\n" + "="*60)
    print("  Step 1: 数据工程 (Data Engineering)")
    print("="*60)
    
    # 1.1 加载数据
    print("\n[1.1] 加载原始数据...")
    loader = DataLoader(CONFIG["data_path"])
    raw_df = loader.load_raw_data()
    
    # 1.2 按月聚合
    print("\n[1.2] 按月聚合...")
    monthly_df = loader.get_full_monthly_series()
    
    # 获取统计信息
    stats = loader.get_statistics()
    print(f"\n--- 数据统计 ---")
    print(f"  总记录数: {stats['total_records']}")
    print(f"  时间范围: {stats['date_range'][0]} ~ {stats['date_range'][1]}")
    print(f"  月份数量: {stats['num_months']}")
    print(f"  月均发证: {stats['monthly_mean']:.1f} ± {stats['monthly_std']:.1f}")
    
    # 1.3 ADF 平稳性检验
    print("\n[1.3] ADF 平稳性检验...")
    preprocessor = TimeSeriesPreprocessor()
    adf_result = preprocessor.adf_test(monthly_df["count"])
    print(adf_result)
    
    # 1.4 为深度学习模型准备数据
    print("\n[1.4] 数据预处理（归一化 + 滑动窗口）...")
    normalized = preprocessor.normalize(monthly_df["count"].values)
    
    X, y = preprocessor.create_sequences(
        normalized,
        window_size=CONFIG["window_size"],
        horizon=CONFIG["horizon"]
    )
    
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(
        X, y, test_ratio=CONFIG["test_ratio"]
    )
    
    # 划分训练/验证（从训练集中划出验证集）
    val_split = int(len(X_train) * 0.8)
    X_val = X_train[val_split:]
    y_val = y_train[val_split:]
    X_train = X_train[:val_split]
    y_train = y_train[:val_split]
    
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  验证集: {len(X_val)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    
    # 可视化
    print("\n[1.5] 绑制时间序列趋势图...")
    plotter = TimeSeriesPlotter(figure_dir=CONFIG["output_dir"] / "figures")
    plotter.plot_time_series(
        monthly_df,
        title="高密市国有土地发证数量月度时间序列 (2016-2025)",
        filename="01_time_series_trend",
        show=False
    )
    
    # ACF/PACF 图
    plotter.plot_acf_pacf(
        monthly_df["count"],
        lags=min(40, len(monthly_df) // 2 - 1),
        title="发证数量自相关分析",
        filename="02_acf_pacf",
        show=False
    )
    
    return {
        "monthly_df": monthly_df,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "stats": stats,
        "adf_result": adf_result,
        "plotter": plotter
    }


def step2_train_models(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 2: 模型训练
    
    训练所有模型并收集预测结果。
    """
    print("\n" + "="*60)
    print("  Step 2: 模型训练 (Model Training)")
    print("="*60)
    
    monthly_df = data["monthly_df"]
    preprocessor = data["preprocessor"]
    
    predictions = {}
    metrics_all = {}
    training_histories = {}
    
    # 获取深度学习测试集的样本数
    n_test = len(data["X_test"])
    
    # 计算深度学习测试集对应的日期索引
    # 滑动窗口后，测试集对应原序列中的最后 n_test 个位置
    test_dates = monthly_df.index[-(n_test):]
    
    # 获取原始尺度的测试集真实值（用于深度学习模型评估）
    y_test_original = preprocessor.inverse_normalize(data["y_test"]).flatten()
    
    # 获取对应日期范围的原始真实值（用于统一对比图）
    actual_values = monthly_df["count"].values[-(n_test):]
    
    # ==================== Baseline: ARIMA ====================
    print("\n[2.1] 训练 ARIMA...")
    
    # ARIMA 使用原始序列，训练集为测试集之前的所有数据
    arima_train = monthly_df["count"][:-n_test]
    arima_test_values = monthly_df["count"][-(n_test):]
    
    arima_model = ARIMAModel(max_p=5, max_d=2, max_q=5)
    arima_model.fit(arima_train)
    arima_pred = arima_model.predict(n_test)
    
    predictions["ARIMA"] = arima_pred
    metrics_all["ARIMA"] = calculate_metrics(arima_test_values.values, arima_pred)
    print_metrics(metrics_all["ARIMA"], "ARIMA")
    
    # ==================== Baseline: Prophet ====================
    print("\n[2.2] 训练 Prophet...")
    
    prophet_train = monthly_df[:-n_test]
    prophet_model = ProphetModel(yearly_seasonality=True)
    prophet_model.fit(prophet_train)
    prophet_pred = prophet_model.predict(n_test)
    
    predictions["Prophet"] = prophet_pred
    metrics_all["Prophet"] = calculate_metrics(arima_test_values.values, prophet_pred)
    print_metrics(metrics_all["Prophet"], "Prophet")
    
    # ==================== Core: LSTM ====================
    print("\n[2.3] 训练 LSTM...")
    
    lstm_model = LSTMModel(CONFIG["lstm"])
    lstm_model.fit(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        verbose=True
    )
    
    lstm_pred_normalized = lstm_model.predict(data["X_test"])
    lstm_pred = preprocessor.inverse_normalize(lstm_pred_normalized.reshape(-1, 1)).flatten()
    
    predictions["LSTM"] = lstm_pred
    metrics_all["LSTM"] = calculate_metrics(y_test_original, lstm_pred)
    training_histories["LSTM"] = lstm_model.get_training_history()
    print_metrics(metrics_all["LSTM"], "LSTM")
    
    # ==================== Core: GRU ====================
    print("\n[2.4] 训练 GRU...")
    
    gru_model = GRUModel(CONFIG["gru"])
    gru_model.fit(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        verbose=True
    )
    
    gru_pred_normalized = gru_model.predict(data["X_test"])
    gru_pred = preprocessor.inverse_normalize(gru_pred_normalized.reshape(-1, 1)).flatten()
    
    predictions["GRU"] = gru_pred
    metrics_all["GRU"] = calculate_metrics(y_test_original, gru_pred)
    training_histories["GRU"] = gru_model.get_training_history()
    print_metrics(metrics_all["GRU"], "GRU")
    
    # ==================== Exploratory: Transformer ====================
    print("\n[2.5] 训练 Transformer...")
    
    transformer_model = TransformerModel(CONFIG["transformer"])
    transformer_model.fit(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        verbose=True
    )
    
    transformer_pred_normalized = transformer_model.predict(data["X_test"])
    transformer_pred = preprocessor.inverse_normalize(
        transformer_pred_normalized.reshape(-1, 1)
    ).flatten()
    
    predictions["Transformer"] = transformer_pred
    metrics_all["Transformer"] = calculate_metrics(y_test_original, transformer_pred)
    training_histories["Transformer"] = transformer_model.get_training_history()
    print_metrics(metrics_all["Transformer"], "Transformer")
    
    print(f"\n  Transformer 参数量: {transformer_model.count_parameters():,}")
    
    return {
        "predictions": predictions,
        "metrics": metrics_all,
        "training_histories": training_histories,
        "test_dates": test_dates,
        "actual_values": actual_values,
        "y_test_original": y_test_original
    }


def step3_evaluation_visualization(
    data: Dict[str, Any], 
    results: Dict[str, Any]
) -> None:
    """
    Step 3: 评估与可视化
    """
    print("\n" + "="*60)
    print("  Step 3: 评估与可视化 (Evaluation & Visualization)")
    print("="*60)
    
    plotter = data["plotter"]
    
    # 3.1 预测对比图
    print("\n[3.1] 绑制预测对比图...")
    
    # 使用统一的测试集真实值
    actual = results["actual_values"]
    
    plotter.plot_predictions_comparison(
        actual=actual,
        predictions=results["predictions"],
        dates=results["test_dates"],
        title="各模型预测结果对比",
        filename="03_predictions_comparison",
        show=False
    )
    
    # 3.2 训练损失曲线
    print("\n[3.2] 绑制训练损失曲线...")
    
    for model_name, (train_losses, val_losses) in results["training_histories"].items():
        plotter.plot_training_curves(
            train_losses=train_losses,
            val_losses=val_losses if len(val_losses) > 0 else None,
            title="训练/验证损失曲线",
            model_name=model_name,
            filename=f"04_loss_curve_{model_name.lower()}",
            show=False
        )
        
    # 3.3 生成评估指标表格
    print("\n[3.3] 生成评估指标表格...")
    
    metrics_df = plotter.create_metrics_table(
        results["metrics"],
        filename="metrics_comparison"
    )
    
    print("\n--- 模型评估指标对比 ---")
    print(metrics_df.to_string())
    
    # 保存为 CSV
    csv_path = CONFIG["output_dir"] / "tables" / "metrics_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(csv_path)
    print(f"\n✓ 指标表格已保存: {csv_path}")


def save_experiment_summary(
    data: Dict[str, Any], 
    results: Dict[str, Any]
) -> None:
    """保存实验摘要"""
    summary_path = CONFIG["output_dir"] / "experiment_summary.txt"
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("  多模型融合土地发证数量时间序列预测实验\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. 数据概况\n")
        f.write("-"*40 + "\n")
        stats = data["stats"]
        f.write(f"  总记录数: {stats['total_records']}\n")
        f.write(f"  时间范围: {stats['date_range'][0]} ~ {stats['date_range'][1]}\n")
        f.write(f"  月份数量: {stats['num_months']}\n")
        f.write(f"  月均发证: {stats['monthly_mean']:.1f} ± {stats['monthly_std']:.1f}\n\n")
        
        f.write("2. ADF 平稳性检验\n")
        f.write("-"*40 + "\n")
        adf = data["adf_result"]
        f.write(f"  统计量: {adf.statistic:.4f}\n")
        f.write(f"  p-value: {adf.p_value:.4f}\n")
        f.write(f"  结论: {'平稳' if adf.is_stationary else '非平稳'}\n\n")
        
        f.write("3. 模型参数配置\n")
        f.write("-"*40 + "\n")
        f.write(f"  滑动窗口: {CONFIG['window_size']} 个月\n")
        f.write(f"  预测步长: {CONFIG['horizon']} 个月\n")
        f.write(f"  测试集比例: {CONFIG['test_ratio']*100:.0f}%\n\n")
        
        f.write("4. 模型评估结果\n")
        f.write("-"*40 + "\n")
        f.write(f"{'模型':<15} {'MAE':>10} {'RMSE':>10} {'MAPE (%)':>10}\n")
        f.write("-"*45 + "\n")
        
        for model_name, metrics in results["metrics"].items():
            f.write(
                f"{model_name:<15} {metrics['MAE']:>10.4f} "
                f"{metrics['RMSE']:>10.4f} {metrics['MAPE']:>10.2f}\n"
            )
            
    print(f"\n✓ 实验摘要已保存: {summary_path}")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("  多模型融合土地发证数量时间序列预测实验")
    print("  Multi-Model Time Series Forecasting Experiment")
    print("="*60)
    print(f"\n随机种子: {RANDOM_SEED}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    # Step 1: 数据工程
    data = step1_data_engineering()
    
    # Step 2: 模型训练
    results = step2_train_models(data)
    
    # Step 3: 评估与可视化
    step3_evaluation_visualization(data, results)
    
    # 保存实验摘要
    save_experiment_summary(data, results)
    
    print("\n" + "="*60)
    print("  ✅ 实验完成！")
    print("="*60)
    print(f"\n产出目录: {CONFIG['output_dir']}")
    print("  - figures/: 可视化图表")
    print("  - tables/: 评估指标表格")
    print("  - experiment_summary.txt: 实验摘要")


if __name__ == "__main__":
    main()

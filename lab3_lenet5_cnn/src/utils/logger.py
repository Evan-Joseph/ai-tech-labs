import logging
import time
import os
from datetime import datetime
from typing import Optional, Dict, Any
import json

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str = "assets/logs", model_name: str = "model"):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
            model_name: 模型名称
        """
        self.log_dir = log_dir
        self.model_name = model_name
        self.start_time = None
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{model_name}_{timestamp}.log")
        
        # 配置日志记录器
        self.logger = logging.getLogger(f"{model_name}_logger")
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            # 添加处理器
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def start_training(self, config: Optional[Dict[str, Any]] = None) -> None:
        """开始训练"""
        self.start_time = time.time()
        self.logger.info("=" * 60)
        self.logger.info(f"开始训练模型: {self.model_name}")
        self.logger.info(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if config:
            self.logger.info("训练配置:")
            for key, value in config.items():
                self.logger.info(f"  {key}: {value}")
        
        self.logger.info("=" * 60)
    
    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                 val_loss: Optional[float] = None, val_acc: Optional[float] = None,
                 additional_metrics: Optional[Dict[str, float]] = None) -> None:
        """
        记录每个epoch的信息
        
        Args:
            epoch: 当前epoch
            train_loss: 训练损失
            train_acc: 训练准确率
            val_loss: 验证损失
            val_acc: 验证准确率
            additional_metrics: 额外指标
        """
        log_msg = f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
        
        if val_loss is not None and val_acc is not None:
            log_msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        
        if additional_metrics:
            for name, value in additional_metrics.items():
                log_msg += f" | {name}: {value:.4f}"
        
        self.logger.info(log_msg)
    
    def log_batch(self, epoch: int, batch: int, total_batches: int, 
                  loss: float, acc: Optional[float] = None,
                  log_interval: int = 100) -> None:
        """
        记录批次信息
        
        Args:
            epoch: 当前epoch
            batch: 当前批次
            total_batches: 总批次数
            loss: 批次损失
            acc: 批次准确率
            log_interval: 日志记录间隔
        """
        if batch % log_interval == 0:
            progress = batch / total_batches * 100
            log_msg = f"Epoch {epoch} [{batch}/{total_batches} ({progress:.1f}%)] Loss: {loss:.4f}"
            
            if acc is not None:
                log_msg += f" Acc: {acc:.4f}"
            
            self.logger.info(log_msg)
    
    def log_learning_rate(self, epoch: int, learning_rate: float) -> None:
        """记录学习率变化"""
        self.logger.info(f"Epoch {epoch} | Learning Rate: {learning_rate:.6f}")
    
    def log_model_info(self, model_summary: str) -> None:
        """记录模型信息"""
        self.logger.info("模型结构:")
        self.logger.info(model_summary)
    
    def log_dataset_info(self, train_samples: int, test_samples: int,
                         input_shape: tuple, num_classes: int) -> None:
        """记录数据集信息"""
        self.logger.info("数据集信息:")
        self.logger.info(f"  训练样本数: {train_samples}")
        self.logger.info(f"  测试样本数: {test_samples}")
        self.logger.info(f"  输入形状: {input_shape}")
        self.logger.info(f"  类别数: {num_classes}")
    
    def log_final_metrics(self, final_metrics: Dict[str, float]) -> None:
        """记录最终指标"""
        self.logger.info("=" * 60)
        self.logger.info("训练完成！最终性能指标:")
        
        for metric_name, metric_value in final_metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
    
    def end_training(self) -> None:
        """结束训练"""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            
            self.logger.info("=" * 60)
            self.logger.info(f"训练结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"总训练时间: {hours:02d}:{minutes:02d}:{seconds:02d}")
            self.logger.info(f"日志文件: {self.log_file}")
            self.logger.info("=" * 60)
    
    def save_training_config(self, config: Dict[str, Any]) -> None:
        """保存训练配置"""
        config_file = os.path.join(self.log_dir, f"{self.model_name}_config.json")
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"训练配置已保存至: {config_file}")
    
    def log_error(self, error_msg: str) -> None:
        """记录错误信息"""
        self.logger.error(f"错误: {error_msg}")
    
    def log_warning(self, warning_msg: str) -> None:
        """记录警告信息"""
        self.logger.warning(f"警告: {warning_msg}")


class ProgressTracker:
    """训练进度跟踪器"""
    
    def __init__(self, total_epochs: int, total_batches: int):
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.current_epoch = 0
        self.current_batch = 0
        self.epoch_losses = []
        self.batch_losses = []
        self.learning_rates = []
    
    def update_epoch(self, epoch: int, loss: float) -> None:
        """更新epoch进度"""
        self.current_epoch = epoch
        self.epoch_losses.append(loss)
    
    def update_batch(self, batch: int, loss: float) -> None:
        """更新batch进度"""
        self.current_batch = batch
        self.batch_losses.append(loss)
    
    def update_learning_rate(self, lr: float) -> None:
        """更新学习率"""
        self.learning_rates.append(lr)
    
    def get_progress_info(self) -> Dict[str, Any]:
        """获取进度信息"""
        return {
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'epoch_progress': self.current_epoch / self.total_epochs * 100,
            'batch_progress': self.current_batch / self.total_batches * 100,
            'epoch_losses': self.epoch_losses[-10:],  # 最近10个epoch的损失
            'learning_rates': self.learning_rates[-10:] if self.learning_rates else []
        }


if __name__ == "__main__":
    # 测试日志记录器
    print("测试训练日志记录器...")
    
    logger = TrainingLogger(model_name="test_model")
    
    # 测试配置
    config = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'epochs': 10,
        'optimizer': 'Adam'
    }
    
    logger.start_training(config)
    logger.log_dataset_info(60000, 10000, (1, 28, 28), 10)
    
    # 模拟训练过程
    for epoch in range(1, 4):
        train_loss = 2.0 - epoch * 0.3
        train_acc = 0.5 + epoch * 0.15
        val_loss = train_loss + 0.1
        val_acc = train_acc - 0.05
        
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
        
        # 模拟批次训练
        for batch in range(1, 5):
            batch_loss = train_loss + np.random.uniform(-0.1, 0.1)
            logger.log_batch(epoch, batch, 100, batch_loss, train_acc + 0.01)
    
    # 记录最终指标
    final_metrics = {
        'test_accuracy': 0.9876,
        'test_precision': 0.9872,
        'test_recall': 0.9881,
        'test_f1': 0.9876
    }
    logger.log_final_metrics(final_metrics)
    logger.end_training()
    
    print("日志记录器测试完成！")

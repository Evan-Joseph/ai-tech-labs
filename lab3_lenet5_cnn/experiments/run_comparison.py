#!/usr/bin/env python3
"""
LeNet-5 vs 现代CNN对比实验主脚本
执行完整的训练、评估和可视化流程
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.mnist_loader import MNISTLoader
from models.lenet_numpy import LeNet5NumPy
from models.modern_torch import VGG_MNIST, ResNet18_MNIST, ModernCNNTrainer
from visualization.plotting import (plot_loss_curves, plot_confusion_matrix, 
                                   plot_error_samples, plot_accuracy_comparison, 
                                   plot_training_time_comparison, plot_model_comparison_table)
from visualization.probes import visualize_conv_kernels, visualize_feature_maps, visualize_all_conv_layers
from utils.metrics import MetricsCalculator
from utils.logger import TrainingLogger, ProgressTracker


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs('assets/figures', exist_ok=True)
        os.makedirs('assets/tables', exist_ok=True)
        os.makedirs('assets/logs', exist_ok=True)
        
        # 初始化数据加载器
        self.mnist_loader = MNISTLoader()
        
        # 获取数据
        self.train_x_numpy, self.train_y_numpy, self.test_x_numpy, self.test_y_numpy = self.mnist_loader.get_numpy_data()
        self.train_loader_pytorch, self.test_loader_pytorch = self.mnist_loader.get_torch_loaders(
            batch_size=config['batch_size'])
        
        print(f"数据加载完成 - 训练: {self.train_x_numpy.shape[0]}, 测试: {self.test_x_numpy.shape[0]}")
    
    def train_lenet_numpy(self) -> Tuple[LeNet5NumPy, Dict]:
        """训练LeNet-5 NumPy版本"""
        print("\n" + "="*60)
        print("开始训练 LeNet-5 (NumPy)")
        print("="*60)
        
        # 创建模型和日志记录器
        model = LeNet5NumPy()
        logger = TrainingLogger(model_name="LeNet5_NumPy")
        progress_tracker = ProgressTracker(
            total_epochs=self.config['epochs'],
            total_batches=len(self.train_x_numpy) // self.config['batch_size']
        )
        
        # 记录训练配置
        lenet_config = {
            'model_type': 'LeNet-5 NumPy',
            'learning_rate': self.config['lenet_lr'],
            'batch_size': self.config['batch_size'],
            'epochs': self.config['epochs'],
            'activation': 'Sigmoid',
            'device': 'NumPy'
        }
        logger.start_training(lenet_config)
        logger.log_dataset_info(
            len(self.train_x_numpy), len(self.test_x_numpy),
            self.train_x_numpy.shape[1:], 10
        )
        
        start_time = time.time()
        
        # 训练循环
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_loss = 0.0
            num_batches = 0
            
            # 打乱数据
            indices = np.random.permutation(len(self.train_x_numpy))
            
            for i in range(0, len(indices), self.config['batch_size']):
                batch_indices = indices[i:i + self.config['batch_size']]
                
                # 获取批次数据
                batch_x = self.train_x_numpy[batch_indices]
                batch_y = self.train_y_numpy[batch_indices]
                
                # 训练步骤
                loss = model.train_step(batch_x, batch_y, self.config['lenet_lr'])
                epoch_loss += loss
                num_batches += 1
                
                # 记录批次信息
                if i % (self.config['batch_size'] * 10) == 0:
                    batch_acc = self._compute_accuracy_numpy(model, batch_x, batch_y)
                    logger.log_batch(epoch, i // self.config['batch_size'] + 1, 
                                  num_batches, loss, batch_acc, log_interval=1)
            
            # 计算平均损失和准确率
            avg_loss = epoch_loss / num_batches
            train_acc = self._compute_accuracy_numpy(model, self.train_x_numpy[:1000], 
                                                   self.train_y_numpy[:1000])
            val_acc, val_loss = self._evaluate_lenet_numpy(model)
            
            # 记录epoch信息
            logger.log_epoch(epoch, avg_loss, train_acc, val_loss, val_loss)
            progress_tracker.update_epoch(epoch, avg_loss)
        
        training_time = time.time() - start_time
        
        # 最终评估
        final_acc, final_loss = self._evaluate_lenet_numpy(model)
        
        # 记录最终结果
        final_metrics = {
            'test_accuracy': final_acc,
            'test_loss': final_loss,
            'training_time': training_time
        }
        logger.log_final_metrics(final_metrics)
        logger.end_training()
        
        return model, {
            'model': model,
            'loss_history': model.loss_history,
            'test_accuracy': final_acc,
            'test_loss': final_loss,
            'training_time': training_time,
            'final_metrics': final_metrics
        }
    
    def train_pytorch_model(self, model_name: str) -> Tuple[nn.Module, Dict]:
        """训练PyTorch模型"""
        print(f"\n" + "="*60)
        print(f"开始训练 {model_name} (PyTorch)")
        print("="*60)
        
        # 创建模型
        if model_name == "VGG_MNIST":
            model = VGG_MNIST()
            lr = self.config['vgg_lr']
        elif model_name == "ResNet18_MNIST":
            model = ResNet18_MNIST()
            lr = self.config['resnet_lr']
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # 创建训练器和优化器
        trainer = ModernCNNTrainer(model, self.device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 日志记录器
        logger = TrainingLogger(model_name=model_name)
        progress_tracker = ProgressTracker(
            total_epochs=self.config['epochs'],
            total_batches=len(self.train_loader_pytorch)
        )
        
        # 记录配置
        model_config = {
            'model_type': model_name,
            'learning_rate': lr,
            'batch_size': self.config['batch_size'],
            'epochs': self.config['epochs'],
            'activation': 'ReLU',
            'device': str(self.device)
        }
        logger.start_training(model_config)
        
        start_time = time.time()
        
        # 训练循环
        for epoch in range(1, self.config['epochs'] + 1):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader_pytorch):
                # 训练步骤
                loss = trainer.train_step(images, labels, optimizer)
                epoch_loss += loss
                num_batches += 1
                
                # 记录批次信息
                if batch_idx % 10 == 0:
                    batch_acc = self._compute_accuracy_pytorch(model, images, labels)
                    logger.log_batch(epoch, batch_idx + 1, len(self.train_loader_pytorch), 
                                  loss, batch_acc, log_interval=1)
            
            # 评估
            train_acc, train_loss = trainer.evaluate(self.train_loader_pytorch)
            val_acc, val_loss = trainer.evaluate(self.test_loader_pytorch)
            
            # 记录epoch信息
            avg_loss = epoch_loss / num_batches
            logger.log_epoch(epoch, avg_loss, train_acc, val_loss, val_loss)
            progress_tracker.update_epoch(epoch, avg_loss)
        
        training_time = time.time() - start_time
        
        # 最终评估
        final_acc, final_loss = trainer.evaluate(self.test_loader_pytorch)
        
        # 记录最终结果
        final_metrics = {
            'test_accuracy': final_acc,
            'test_loss': final_loss,
            'training_time': training_time
        }
        logger.log_final_metrics(final_metrics)
        logger.end_training()
        
        return model, {
            'model': model,
            'trainer': trainer,
            'loss_history': trainer.loss_history,
            'test_accuracy': final_acc,
            'test_loss': final_loss,
            'training_time': training_time,
            'final_metrics': final_metrics
        }
    
    def _compute_accuracy_numpy(self, model: LeNet5NumPy, x: np.ndarray, y: np.ndarray) -> float:
        """计算NumPy模型的准确率"""
        predictions = model.predict(x)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)
    
    def _evaluate_lenet_numpy(self, model: LeNet5NumPy) -> Tuple[float, float]:
        """评估LeNet-5 NumPy模型"""
        return model.evaluate(self.test_x_numpy, self.test_y_numpy)
    
    def _compute_accuracy_pytorch(self, model: nn.Module, images: torch.Tensor, 
                                  labels: torch.Tensor) -> float:
        """计算PyTorch模型的准确率"""
        model.eval()
        with torch.no_grad():
            images = images.to(self.device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            return (predicted.cpu() == labels).float().mean().item()
    
    def run_full_experiment(self) -> Dict:
        """运行完整实验"""
        print("开始完整实验...")
        print(f"训练配置: {self.config}")
        
        results = {}
        
        # 1. 训练LeNet-5
        lenet_model, lenet_results = self.train_lenet_numpy()
        results['LeNet-5'] = lenet_results
        
        # 2. 训练VGG
        vgg_model, vgg_results = self.train_pytorch_model("VGG_MNIST")
        results['VGG'] = vgg_results
        
        # 3. 训练ResNet
        resnet_model, resnet_results = self.train_pytorch_model("ResNet18_MNIST")
        results['ResNet'] = resnet_results
        
        # 4. 生成对比图表
        self._generate_comparison_plots(results)
        
        # 5. 进行错误分析
        self._perform_error_analysis(results)
        
        # 6. 生成网络探针可视化
        self._generate_network_probes(lenet_model, resnet_model)
        
        # 7. 保存实验结果
        self._save_experiment_results(results)
        
        print("\n" + "="*60)
        print("实验完成！")
        print("="*60)
        
        # 打印最终结果对比
        print("最终结果对比:")
        print(f"{'模型':<12} {'测试准确率':<12} {'训练时间(s)':<12}")
        print("-" * 36)
        for model_name, result in results.items():
            acc = result['test_accuracy']
            time_cost = result['training_time']
            print(f"{model_name:<12} {acc:<12.4f} {time_cost:<12.2f}")
        
        return results
    
    def _generate_comparison_plots(self, results: Dict) -> None:
        """生成对比图表"""
        print("\n生成对比图表...")
        
        # 损失曲线对比
        loss_histories = {
            'LeNet': results['LeNet-5']['loss_history'],
            'VGG': results['VGG']['loss_history'],
            'ResNet': results['ResNet']['loss_history']
        }
        plot_loss_curves(loss_histories)
        
        # 性能指标对比 - 分别绘制准确率和训练时间
        accuracy_metrics = {
            'LeNet-5': {'准确率': results['LeNet-5']['test_accuracy']},
            'VGG': {'准确率': results['VGG']['test_accuracy']},
            'ResNet': {'准确率': results['ResNet']['test_accuracy']}
        }
        time_metrics = {
            'LeNet-5': {'训练时间': results['LeNet-5']['training_time']},
            'VGG': {'训练时间': results['VGG']['training_time']},
            'ResNet': {'训练时间': results['ResNet']['training_time']}
        }
        plot_accuracy_comparison(accuracy_metrics, "模型准确率对比")
        plot_training_time_comparison(time_metrics, "模型训练时间对比")
        
        # 性能总结表格
        summary_results = {
            'LeNet-5': {
                '准确率': results['LeNet-5']['test_accuracy'],
                '训练时间(s)': results['LeNet-5']['training_time']
            },
            'VGG': {
                '准确率': results['VGG']['test_accuracy'],
                '训练时间(s)': results['VGG']['training_time']
            },
            'ResNet': {
                '准确率': results['ResNet']['test_accuracy'],
                '训练时间(s)': results['ResNet']['training_time']
            }
        }
        plot_model_comparison_table(summary_results, "模型性能总结")
    
    def _perform_error_analysis(self, results: Dict) -> None:
        """进行错误分析"""
        print("\n进行错误分析...")
        
        # 获取所有模型的预测结果
        lenet_pred = results['LeNet-5']['model'].predict(self.test_x_numpy)
        
        # PyTorch模型预测
        vgg_model = results['VGG']['model']
        resnet_model = results['ResNet']['model']
        
        vgg_pred = []
        resnet_pred = []
        
        vgg_model.eval()
        resnet_model.eval()
        
        with torch.no_grad():
            # 分批预测以避免内存问题
            batch_size = 100
            for i in range(0, len(self.test_x_numpy), batch_size):
                batch_x = torch.FloatTensor(self.test_x_numpy[i:i+batch_size]).to(self.device)
                
                vgg_out = vgg_model(batch_x)
                resnet_out = resnet_model(batch_x)
                
                vgg_pred.extend(torch.argmax(vgg_out, dim=1).cpu().numpy())
                resnet_pred.extend(torch.argmax(resnet_out, dim=1).cpu().numpy())
        
        vgg_pred = np.array(vgg_pred)
        resnet_pred = np.array(resnet_pred)
        true_labels = np.argmax(self.test_y_numpy, axis=1)
        
        # 生成错误样本图
        plot_error_samples(
            self.test_x_numpy, true_labels, 
            np.array([lenet_pred, vgg_pred, resnet_pred]),
            ['LeNet-5', 'VGG', 'ResNet']
        )
        
        # 生成混淆矩阵
        cm_lenet = np.zeros((10, 10), dtype=int)
        cm_vgg = np.zeros((10, 10), dtype=int)
        cm_resnet = np.zeros((10, 10), dtype=int)
        
        for true_pred, cm in [(lenet_pred, cm_lenet), 
                              (vgg_pred, cm_vgg), 
                              (resnet_pred, cm_resnet)]:
            for t, p in zip(true_labels, true_pred):
                cm[t, p] += 1
        
        plot_confusion_matrix(cm_lenet, "LeNet5", "LeNet-5 混淆矩阵")
        plot_confusion_matrix(cm_vgg, "VGG", "VGG 混淆矩阵")  
        plot_confusion_matrix(cm_resnet, "ResNet", "ResNet 混淆矩阵")
    
    def _generate_network_probes(self, lenet_model: LeNet5NumPy, resnet_model: nn.Module) -> None:
        """生成网络内部探针可视化"""
        print("\n生成网络探针可视化...")
        
        # LeNet-5多层卷积核可视化
        lenet_all_kernels = lenet_model.get_all_conv_weights()
        visualize_all_conv_layers(lenet_all_kernels, "LeNet-5 多层卷积核可视化")
        
        # 保留单层可视化作为对比
        lenet_kernels = lenet_model.get_first_layer_weights()
        visualize_conv_kernels(lenet_kernels, "LeNet5_Conv1", "LeNet-5 第一层卷积核")
        
        # ResNet特征图
        resnet_model.eval()
        with torch.no_grad():
            # 选择一个样本
            sample_image = torch.FloatTensor(self.test_x_numpy[:1]).to(self.device)
            
            # 获取特征图
            feature_maps = resnet_model.get_feature_maps(sample_image)
            
            visualize_feature_maps(
                feature_maps, sample_image,
                ["Conv1", "Layer1", "Layer2", "Layer3", "Layer4"],
                title="ResNet-18 特征图可视化"
            )
    
    def _save_experiment_results(self, results: Dict) -> None:
        """保存实验结果"""
        print("\n保存实验结果...")
        
        import json
        
        # 准备保存的结果
        save_results = {}
        for model_name, result in results.items():
            save_results[model_name] = {
                'test_accuracy': float(result['test_accuracy']),
                'test_loss': float(result['test_loss']),
                'training_time': float(result['training_time']),
                'final_metrics': {k: float(v) for k, v in result['final_metrics'].items() if k != 'training_time'}
            }
        
        # 保存到JSON文件
        with open('assets/tables/experiment_results.json', 'w', encoding='utf-8') as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        
        print("实验结果已保存至: assets/tables/experiment_results.json")


def get_default_config() -> Dict:
    """获取默认配置"""
    return {
        'batch_size': 64,
        'epochs': 20,
        'lenet_lr': 0.01,    # LeNet学习率需要小一些
        'vgg_lr': 0.001,      # VGG学习率
        'resnet_lr': 0.001,   # ResNet学习率
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='LeNet-5 vs 现代CNN对比实验')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lenet-lr', type=float, default=0.01, help='LeNet学习率')
    parser.add_argument('--vgg-lr', type=float, default=0.001, help='VGG学习率')
    parser.add_argument('--resnet-lr', type=float, default=0.001, help='ResNet学习率')
    parser.add_argument('--quick-test', action='store_true', help='快速测试模式（较少训练轮数）')
    
    args = parser.parse_args()
    
    # 获取配置
    config = get_default_config()
    config.update(vars(args))
    
    # 快速测试模式
    if args.quick_test:
        config['epochs'] = 5
        config['batch_size'] = 32
        print("使用快速测试模式...")
    
    # 运行实验
    runner = ExperimentRunner(config)
    results = runner.run_full_experiment()
    
    return results


if __name__ == "__main__":
    results = main()

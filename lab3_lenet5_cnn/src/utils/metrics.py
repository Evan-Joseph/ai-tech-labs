import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional
import json
import os


class MetricsCalculator:
    """性能指标计算器"""
    
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self) -> None:
        """重置所有统计信息"""
        self.all_predictions = []
        self.all_targets = []
        self.all_probabilities = []
    
    def update(self, predictions: np.ndarray, targets: np.ndarray, 
               probabilities: Optional[np.ndarray] = None) -> None:
        """
        更新预测结果
        
        Args:
            predictions: 预测类别 (batch_size,)
            targets: 真实类别 (batch_size,)
            probabilities: 预测概率 (batch_size, num_classes)
        """
        self.all_predictions.extend(predictions)
        self.all_targets.extend(targets)
        
        if probabilities is not None:
            self.all_probabilities.extend(probabilities)
    
    def compute_confusion_matrix(self) -> np.ndarray:
        """计算混淆矩阵"""
        return confusion_matrix(self.all_targets, self.all_predictions)
    
    def compute_accuracy(self) -> float:
        """计算准确率"""
        return accuracy_score(self.all_targets, self.all_predictions)
    
    def compute_precision_recall_f1(self, average: str = 'macro') -> Tuple[float, float, float]:
        """
        计算精确率、召回率、F1分数
        
        Args:
            average: 平均方式 ('macro', 'micro', 'weighted')
            
        Returns:
            (precision, recall, f1)
        """
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_targets, self.all_predictions, average=average, zero_division=0)
        return precision, recall, f1
    
    def compute_per_class_metrics(self) -> Dict:
        """计算每个类别的详细指标"""
        return classification_report(
            self.all_targets, self.all_predictions, 
            target_names=[str(i) for i in range(self.num_classes)],
            output_dict=True, zero_division=0)
    
    def compute_class_accuracies(self) -> List[float]:
        """计算每个类别的准确率"""
        cm = self.compute_confusion_matrix()
        class_accuracies = []
        
        for i in range(self.num_classes):
            if cm[i].sum() > 0:
                acc = cm[i, i] / cm[i].sum()
            else:
                acc = 0.0
            class_accuracies.append(acc)
        
        return class_accuracies
    
    def get_all_metrics(self) -> Dict:
        """获取所有性能指标"""
        cm = self.compute_confusion_matrix()
        accuracy = self.compute_accuracy()
        precision, recall, f1 = self.compute_precision_recall_f1('macro')
        per_class_metrics = self.compute_per_class_metrics()
        class_accuracies = self.compute_class_accuracies()
        
        return {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'per_class_metrics': per_class_metrics,
            'class_accuracies': [float(acc) for acc in class_accuracies],
            'confusion_matrix': cm.tolist()
        }
    
    def save_metrics(self, save_path: str, model_name: str = "model") -> None:
        """
        保存指标到文件
        
        Args:
            save_path: 保存路径
            model_name: 模型名称
        """
        metrics = self.get_all_metrics()
        
        # 确保目录存在（如果是相对路径，确保从项目根目录开始）
        if not os.path.isabs(save_path):
            save_path = os.path.join(os.getcwd(), save_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存为JSON格式
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({model_name: metrics}, f, indent=2, ensure_ascii=False)
        
        print(f"指标已保存至: {save_path}")
    
    def compare_models(self, other_metrics: Dict, model_names: List[str]) -> Dict:
        """
        比较多个模型的指标
        
        Args:
            other_metrics: 其他模型的指标字典
            model_names: 模型名称列表
            
        Returns:
            比较结果
        """
        comparison = {}
        
        # 当前模型指标
        current_metrics = self.get_all_metrics()['overall_metrics']
        
        # 添加当前模型
        comparison[model_names[0]] = current_metrics
        
        # 添加其他模型
        for i, (name, metrics) in enumerate(zip(model_names[1:], other_metrics)):
            if 'overall_metrics' in metrics:
                comparison[name] = metrics['overall_metrics']
            else:
                comparison[name] = metrics
        
        return comparison
    
    def find_error_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        找出所有错误分类的样本
        
        Returns:
            (error_indices, true_labels, predicted_labels)
        """
        all_predictions = np.array(self.all_predictions)
        all_targets = np.array(self.all_targets)
        
        # 找出错误分类的样本
        error_mask = all_predictions != all_targets
        error_indices = np.where(error_mask)[0]
        
        true_labels = all_targets[error_mask]
        predicted_labels = all_predictions[error_mask]
        
        return error_indices, true_labels, predicted_labels


def compute_model_comparison_table(metrics_list: List[Dict], 
                                  model_names: List[str]) -> Dict:
    """
    创建模型比较表格
    
    Args:
        metrics_list: 各模型指标列表
        model_names: 模型名称列表
        
    Returns:
        比较表格
    """
    comparison_table = {
        'model_names': model_names,
        'metrics': {}
    }
    
    # 提取关键指标
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for metric_name in metric_names:
        comparison_table['metrics'][metric_name] = []
        
        for metrics in metrics_list:
            if 'overall_metrics' in metrics:
                comparison_table['metrics'][metric_name].append(
                    metrics['overall_metrics'][metric_name])
            else:
                comparison_table['metrics'][metric_name].append(
                    metrics[metric_name])
    
    # 找出最佳模型
    for metric_name in metric_names:
        values = comparison_table['metrics'][metric_name]
        if metric_name in ['accuracy', 'precision', 'recall', 'f1_score']:
            best_idx = np.argmax(values)
            comparison_table[f'best_{metric_name}'] = model_names[best_idx]
    
    return comparison_table


def save_comparison_table(comparison_table: Dict, save_path: str) -> None:
    """
    保存比较表格到文件
    
    Args:
        comparison_table: 比较表格
        save_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_table, f, indent=2, ensure_ascii=False)
    
    print(f"比较表格已保存至: {save_path}")


if __name__ == "__main__":
    # 测试指标计算器
    print("测试指标计算器...")
    
    calculator = MetricsCalculator()
    
    # 模拟一些预测结果
    true_labels = np.random.randint(0, 10, 100)
    pred_labels = true_labels.copy()
    # 随机修改一些预测作为错误样本
    error_indices = np.random.choice(100, 10, replace=False)
    pred_labels[error_indices] = np.random.randint(0, 10, 10)
    
    calculator.update(pred_labels, true_labels)
    
    # 计算指标
    metrics = calculator.get_all_metrics()
    print("总体指标:", metrics['overall_metrics'])
    print("类别准确率:", metrics['class_accuracies'])
    
    # 找出错误样本
    error_idx, error_true, error_pred = calculator.find_error_samples()
    print(f"错误样本数量: {len(error_idx)}")
    
    print("指标计算器测试完成！")

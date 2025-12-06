
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix
import numpy as np

# Font Configuration
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
sns.set_theme(style="whitegrid", font='Songti SC')

def load_results(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_accuracy_comparison(results, output_dir):
    data = []
    for model, res in results.items():
        best_val_acc = max(e['val_acc'] for e in res['epochs'])
        data.append({'Model': model.upper(), 'Accuracy': best_val_acc})
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 7))
    ax = sns.barplot(x='Model', y='Accuracy', data=df, palette='viridis')
    plt.title('Validation Accuracy Comparison', fontsize=16, pad=20)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Model', fontsize=14)
    plt.ylim(0, 1.05)
    
    for i, row in df.iterrows():
        ax.text(i, row.Accuracy + 0.01, f"{row.Accuracy:.2%}", ha='center', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300)
    plt.close()

def plot_efficiency(results, output_dir):
    data = []
    for model, res in results.items():
        best_epoch = res['epochs'][-1]
        data.append({
            'Model': model.upper(),
            'Params (M)': res['params_millions'],
            'Accuracy': best_epoch['val_acc'],
            'Inference Time (ms)': best_epoch['inference_time_ms']
        })
    
    df = pd.DataFrame(data)
    
    # Setup Figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Efficiency Analysis (Lower Params/Time is Better)', fontsize=18, y=1.05)
    
    # 1. Accuracy (Higher is better)
    sns.barplot(ax=axes[0], x='Model', y='Accuracy', data=df, palette='viridis')
    axes[0].set_title('Validation Accuracy (Higher is Better)', fontsize=14)
    axes[0].set_ylim(0.9, 1.0) # Zoom in to show differences
    for i, row in df.iterrows():
        axes[0].text(i, row.Accuracy, f"{row.Accuracy:.2%}", ha='center', va='bottom', fontweight='bold')

    # 2. Parameters (Lower is better)
    sns.barplot(ax=axes[1], x='Model', y='Params (M)', data=df, palette='magma')
    axes[1].set_title('Model Size (Params in Millions)', fontsize=14)
    for i, row in df.iterrows():
        axes[1].text(i, row['Params (M)'], f"{row['Params (M)']:.1f}M", ha='center', va='bottom', fontweight='bold')

    # 3. Inference Time (Lower is better)
    sns.barplot(ax=axes[2], x='Model', y='Inference Time (ms)', data=df, palette='rocket')
    axes[2].set_title('Inference Latency (ms/image)', fontsize=14)
    for i, row in df.iterrows():
        axes[2].text(i, row['Inference Time (ms)'], f"{row['Inference Time (ms)']:.1f}ms", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'efficiency_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix_heatmap(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 12})
    plt.title('Confusion Matrix (ResNet50)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

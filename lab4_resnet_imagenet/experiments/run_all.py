
import sys
import json
import torch
import statistics
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.dataset_loader import get_dataloaders
from src.models.model_factory import ModelFactory
from src.utils.trainer import Trainer

MODELS = ['alexnet', 'vgg16', 'resnet50']
EPOCHS = 5
RESULTS_FILE = Path(__file__).resolve().parent.parent / "assets" / "tables" / "results.json"

def main():
    # Device detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    train_loader, val_loader, class_names = get_dataloaders()
    # Load existing results if skipping training
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    for model_name in MODELS:
        print(f"\n{'='*20} Processing {model_name} {'='*20}")
        # Need model instance for checkpoint loading later, but don't need to retrain
        pass # Training loop logic handled above/skipped
        
        if False: # SKIP TRAINING (Already Done)
            for epoch in range(EPOCHS):
                train_loss, train_acc = trainer.train_epoch()
                val_loss, val_acc, inf_time = trainer.validate()
                
                print(f"Epoch {epoch+1}/{EPOCHS} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"Inf Time: {inf_time:.2f} ms/img")
                
        if False: # SKIP TRAINING (Already Done)
            for epoch in range(EPOCHS):
                # ... (code hidden)
                pass
            
        # results[model_name] = model_results # REMOVED to prevent NameError
        
        # Save Checkpoint (SKIP)
        # ckpt_dir = RESULTS_FILE.parent.parent / "checkpoints"
        # ckpt_dir.mkdir(parents=True, exist_ok=True)
        # torch.save(model.state_dict(), ckpt_dir / f"{model_name}.pth")
        
        # Free memory (especially for MPS)
        # del model
        # del trainer
        if device.type == "mps":
            torch.mps.empty_cache()
        


    # Save results (Skip overwriting if we didn't train)
    # RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    # with open(RESULTS_FILE, 'w') as f:
    #     json.dump(results, f, indent=4)
    # print(f"\nResults saved to {RESULTS_FILE}")
    
    # --- Visualization Phase ---
    print("\nStarting Visualization...")
    from src.visualization.plotting import plot_accuracy_comparison, plot_efficiency, plot_confusion_matrix_heatmap
    from src.visualization.cam import CamGenerator
    
    figs_dir = RESULTS_FILE.parent.parent / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Performance Charts
    print("Generating charts...")
    plot_accuracy_comparison(results, figs_dir)
    plot_efficiency(results, figs_dir)
    
    # 2. Re-load models for CAM and Confusion Matrix
    print("Generating CAM and Confusion Matrix...")
    models_dict = {}
    best_model_name = max(results, key=lambda x: results[x]['epochs'][-1]['val_acc'])
    print(f"Best model: {best_model_name}")
    
    # Monkey-patch the dataset classes to readable names for CAM
    val_loader.dataset.classes = class_names

    ckpt_dir = RESULTS_FILE.parent.parent / "checkpoints"

    # Reload all for CAM
    for name in MODELS:
        m = ModelFactory.get_model(name)
        m.load_state_dict(torch.load(ckpt_dir / f"{name}.pth", map_location=device))
        models_dict[name] = m
        
    # CAM
    cam_gen = CamGenerator(device)
    # Pick random images, fixed for reproducibility
    # 0, 100: Tench (Fish)
    # others: varied classes
    indices = [0, 100, 400, 800, 1200] 
    for idx in indices:
        # Verify label
        _, label_idx = val_loader.dataset[idx]
        label_name = class_names[label_idx]
        print(f"Generating CAM for Index {idx}: Label={label_name}")
        
        cam_gen.generate_comparison(models_dict, val_loader.dataset, idx, figs_dir / f"cam_{idx}.png")
        
    # Confusion Matrix (Best Model)
    best_model = models_dict[best_model_name]
    best_model.to(device)
    best_model.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running inference for Confusion Matrix...")
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = best_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    plot_confusion_matrix_heatmap(all_labels, all_preds, class_names, figs_dir / "confusion_matrix.png")
    print("Visualization Complete.")

if __name__ == "__main__":
    main()

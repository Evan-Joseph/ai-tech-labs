
import torch
import torch.nn as nn
from tqdm import tqdm
import time

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.001, momentum=0.9):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        # Only optimize parameters that require gradients (the head)
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=learning_rate, 
            momentum=momentum
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(self.train_loader, desc="Training", leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for inputs, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        end_time = time.perf_counter()
        inference_time_ms = (end_time - start_time) * 1000 / total if total > 0 else 0
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc, inference_time_ms

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters()) / 1e6  # Millions

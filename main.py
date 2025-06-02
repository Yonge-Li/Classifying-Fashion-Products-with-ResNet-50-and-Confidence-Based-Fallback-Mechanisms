import torch
import numpy as np
import time
import torch.nn as nn

from configs.settings import config, setup_visualization_dir
from data.dataset import create_balanced_dataloaders
from models.hierarchical_resnet import HierarchicalResNet
from training.train import train_model
from training.evaluator import evaluate_model

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    
    setup_visualization_dir()
    
    print("\nCreating dataloaders...")
    train_loader, val_loader, subclass_to_main = create_balanced_dataloaders()
    
    print("\nInitializing model...")
    model = HierarchicalResNet(
        config['num_main_classes'], 
        config['num_sub_classes']
    ).to(config['device'])
    
    start_time = time.time()
    model, history = train_model(model, train_loader, val_loader, subclass_to_main)
    print(f"\nTotal training time: {time.time()-start_time:.2f}s")
    
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(config['save_path']))
    final_loss, final_acc, final_report, _, _, _, _, _ = evaluate_model(
        model, val_loader, 
        nn.CrossEntropyLoss(), 
        nn.CrossEntropyLoss(), 
        subclass_to_main
    )
    print(f"\nFinal Validation Accuracy: {final_acc:.4f}")
    print("\nClassification Report:")
    print(final_report)
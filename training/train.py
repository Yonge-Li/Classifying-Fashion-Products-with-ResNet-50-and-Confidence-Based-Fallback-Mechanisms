import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from configs.settings import config

def train_model(model, train_loader, val_loader, subclass_to_main):
    main_weights, sub_weights = compute_class_weights(train_loader.dataset.samples)
    criterion_main = nn.CrossEntropyLoss(weight=main_weights)
    criterion_sub = nn.CrossEntropyLoss(weight=sub_weights)
    
    optimizer = optim.Adam([
        {'params': model.base.parameters(), 'lr': config['lr']/10},
        {'params': model.main_classifier.parameters()},
        {'params': model.sub_classifier.parameters()},
    ], lr=config['lr'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    best_acc = 0.0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_main_acc': [],
        'val_main_acc': [],
        'train_sub_acc': [],
        'val_sub_acc': [],
        'train_overall_acc': [],
        'val_overall_acc': []
    }
    
    print("\nStarting training...")
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        correct_main = 0
        correct_sub = 0
        correct_overall = 0
        total = 0
        
        for inputs, (main_labels, sub_labels) in train_loader:
            inputs = inputs.to(config['device'])
            main_labels = main_labels.to(config['device'])
            sub_labels = sub_labels.to(config['device'])
            
            optimizer.zero_grad()
            main_out, sub_out = model(inputs)
            
            loss_main = criterion_main(main_out, main_labels)
            loss_sub = criterion_sub(sub_out, sub_labels)
            total_loss = loss_main + loss_sub
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item() * inputs.size(0)
            _, main_preds = torch.max(main_out, 1)
            _, sub_preds = torch.max(sub_out, 1)
            correct_main += (main_preds == main_labels).sum().item()
            correct_sub += (sub_preds == sub_labels).sum().item()
            correct_overall += ((main_preds == main_labels) & (sub_preds == sub_labels)).sum().item()
            total += inputs.size(0)
        
        epoch_loss = running_loss / total
        epoch_main_acc = correct_main / total
        epoch_sub_acc = correct_sub / total
        epoch_overall_acc = correct_overall / total
        
        history['train_loss'].append(epoch_loss)
        history['train_main_acc'].append(epoch_main_acc)
        history['train_sub_acc'].append(epoch_sub_acc)
        history['train_overall_acc'].append(epoch_overall_acc)
        
        val_loss, val_acc, val_report, main_true, main_pred, main_probs, sub_true, sub_pred = evaluate_model(
            model, val_loader, criterion_main, criterion_sub, subclass_to_main
        )
        
        history['val_loss'].append(val_loss)
        history['val_main_acc'].append(val_acc)
        
        valid_sub_mask = np.array(sub_pred) != -1
        if sum(valid_sub_mask) > 0:
            val_sub_acc = (np.array(sub_true)[valid_sub_mask] == np.array(sub_pred)[valid_sub_mask]).mean()
        else:
            val_sub_acc = 0.0
        history['val_sub_acc'].append(val_sub_acc)
        
        val_overall_acc = ((np.array(main_true) == np.array(main_pred)) & 
                         (np.array(sub_true) == np.array(sub_pred))).mean()
        history['val_overall_acc'].append(val_overall_acc)
        
        scheduler.step(val_acc)
        
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == config['num_epochs'] - 1:
            plot_confusion_matrix(main_true, main_pred, 
                                config['main_classes'] + [config['other_class_name']],
                                f'Main Class Confusion Matrix (Epoch {epoch+1})',
                                f'main_confusion_matrix_epoch_{epoch+1}.png')
            
            valid_sub_mask = np.array(sub_pred) != -1
            if sum(valid_sub_mask) > 0:
                plot_confusion_matrix(np.array(sub_true)[valid_sub_mask], 
                                    np.array(sub_pred)[valid_sub_mask],
                                    config['subclass_names'],
                                    f'Sub Class Confusion Matrix (Epoch {epoch+1})',
                                    f'sub_confusion_matrix_epoch_{epoch+1}.png',
                                    figsize=(12, 10))
            
            plot_roc_curve_multi_class(np.array(main_true), 
                                     np.array(main_probs), 
                                     len(config['main_classes']) + 1,
                                     f'ROC Curves (Epoch {epoch+1})',
                                     f'roc_curve_epoch_{epoch+1}.png')
            
            plot_training_curves(history, 'training_curves.png')
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        print(f"Train Loss: {epoch_loss:.4f} | Main Acc: {epoch_main_acc:.4f} | Sub Acc: {epoch_sub_acc:.4f} | Overall Acc: {epoch_overall_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Main Acc: {val_acc:.4f} | Val Sub Acc: {val_sub_acc:.4f} | Val Overall Acc: {val_overall_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config['save_path'])
            print(f"New best model saved with acc: {val_acc:.4f}")
    
    plot_training_curves(history, 'final_training_curves.png')
    print(f"\nTraining complete. Best val Acc: {best_acc:.4f}")
    return model, history

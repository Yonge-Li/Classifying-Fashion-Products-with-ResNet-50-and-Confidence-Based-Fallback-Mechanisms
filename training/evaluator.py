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

def evaluate_model(model, dataloader, criterion_main, criterion_sub, subclass_to_main):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    all_main_true = []
    all_main_pred = []
    all_sub_true = []
    all_sub_pred = []
    all_main_probs = []
    
    with torch.no_grad():
        for inputs, (main_labels, sub_labels) in dataloader:
            inputs = inputs.to(config['device'])
            main_labels = main_labels.to(config['device'])
            sub_labels = sub_labels.to(config['device'])
            
            main_out, sub_out = model(inputs)
            loss_main = criterion_main(main_out, main_labels)
            loss_sub = criterion_sub(sub_out, sub_labels)
            total_loss = loss_main + loss_sub
            running_loss += total_loss.item() * inputs.size(0)
            
            main_probs = torch.softmax(main_out, dim=1)
            main_preds = torch.argmax(main_probs, dim=1)
            main_conf, _ = torch.max(main_probs, dim=1)
            
            sub_probs = torch.softmax(sub_out, dim=1)
            sub_conf, sub_preds = torch.max(sub_probs, dim=1)
            
            sub_preds_np = sub_preds.cpu().numpy()
            main_preds_np = main_preds.cpu().numpy()
            main_conf_np = main_conf.cpu().numpy()
            sub_conf_np = sub_conf.cpu().numpy()
            main_labels_np = main_labels.cpu().numpy()
            sub_labels_np = sub_labels.cpu().numpy()
            main_probs_np = main_probs.cpu().numpy()
            
            all_main_true.extend(main_labels_np)
            all_sub_true.extend(sub_labels_np)
            all_main_probs.extend(main_probs_np)
            
            final_main_pred = []
            final_sub_pred = []
            
            for i in range(len(sub_preds_np)):
                sub_idx = sub_preds_np[i]
                main_idx = main_preds_np[i]
                sub_c = sub_conf_np[i]
                main_c = main_conf_np[i]
                
                belongs_to_main = subclass_to_main[sub_idx] == main_idx
                meets_sub_threshold = sub_c >= config['subclass_threshold']
                meets_main_threshold = main_c >= config['main_class_threshold']
                
                if not meets_main_threshold:
                    final_main_pred.append(config['num_main_classes'])
                    final_sub_pred.append(-1)
                elif belongs_to_main and meets_sub_threshold:
                    final_main_pred.append(subclass_to_main[sub_idx])
                    final_sub_pred.append(sub_idx)
                else:
                    final_main_pred.append(main_idx)
                    final_sub_pred.append(-1)
            
            final_main_pred = torch.tensor(final_main_pred, device=config['device'])
            
            correct = (final_main_pred == main_labels).sum().item()
            running_correct += correct
            total += inputs.size(0)
            
            all_main_pred.extend(final_main_pred.cpu().numpy())
            all_sub_pred.extend(final_sub_pred)
    
    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    
    all_main_probs = np.array(all_main_probs)
    other_probs = 1 - np.max(all_main_probs, axis=1, keepdims=True)
    all_main_probs = np.concatenate([all_main_probs, other_probs], axis=1)
    
    target_names = config['main_classes'] + [config['other_class_name']]
    main_report = classification_report(all_main_true, all_main_pred, 
                                      target_names=target_names)
    
    valid_sub_mask = np.array(all_sub_pred) != -1
    valid_sub_true = np.array(all_sub_true)[valid_sub_mask]
    valid_sub_pred = np.array(all_sub_pred)[valid_sub_mask]
    
    if len(valid_sub_true) > 0:
        sub_report = classification_report(valid_sub_true, valid_sub_pred,
                                         target_names=config['subclass_names'])
    else:
        sub_report = "No valid subclass predictions"
    
    report = f"Main Classes:\n{main_report}\n\nSubclasses (only valid predictions):\n{sub_report}"
    
    return epoch_loss, epoch_acc, report, all_main_true, all_main_pred, all_main_probs, all_sub_true, all_sub_pred
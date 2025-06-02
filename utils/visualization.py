import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
import os
from collections import defaultdict
from sklearn.model_selection import train_test_split
import matplotlib

import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from configs.setting import config

def plot_confusion_matrix(y_true, y_pred, classes, title, filename, figsize=(10,8)):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    save_path = os.path.join(config['visualization_dir'], filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_roc_curve_multi_class(y_true, y_score, n_classes, title, filename):
    y_true_onehot = np.eye(n_classes)[y_true]
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_onehot.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='Class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))
    
    plt.plot(fpr["micro"], tpr["micro"],
             label='Micro-avg (AUC = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    save_path = os.path.join(config['visualization_dir'], filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_training_curves(history, filename):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(history['train_main_acc'], label='Train Main Acc')
    plt.plot(history['val_main_acc'], label='Val Main Acc')
    plt.title('Main Class Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(history['train_sub_acc'], label='Train Sub Acc')
    plt.plot(history['val_sub_acc'], label='Val Sub Acc')
    plt.title('Sub Class Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history['train_overall_acc'], label='Train Overall Acc')
    plt.plot(history['val_overall_acc'], label='Val Overall Acc')
    plt.title('Overall Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(config['visualization_dir'], filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

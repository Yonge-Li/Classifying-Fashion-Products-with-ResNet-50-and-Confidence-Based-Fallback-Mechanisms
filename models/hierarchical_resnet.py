import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoad
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from configs.settings import config

class HierarchicalResNet(nn.Module):
    def __init__(self, num_main_classes, num_sub_classes):
        super().__init__()
        self.base = models.resnet50(pretrained=True)
        num_ftrs = self.base.fc.in_features
        self.base.fc = nn.Identity()
        self.main_classifier = nn.Linear(num_ftrs, num_main_classes)
        self.sub_classifier = nn.Linear(num_ftrs, num_sub_classes)
        
    def forward(self, x):
        features = self.base(x)
        main_output = self.main_classifier(features)
        sub_output = self.sub_classifier(features)
        return main_output, sub_output

def compute_class_weights(samples):
    main_labels = [s[1][0] for s in samples if s[1][0] is not None]
    sub_labels = [s[1][1] for s in samples if s[1][1] is not None]
    
    main_classes = np.unique(main_labels)
    main_weights = compute_class_weight('balanced', classes=main_classes, y=main_labels)
    
    sub_classes = np.unique(sub_labels)
    sub_weights = compute_class_weight('balanced', classes=sub_classes, y=sub_labels)
    
    return (torch.FloatTensor(main_weights).to(config['device']),
            torch.FloatTensor(sub_weights).to(config['device']))
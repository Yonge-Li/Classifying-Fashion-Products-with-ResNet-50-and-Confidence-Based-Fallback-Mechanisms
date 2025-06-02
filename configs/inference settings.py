import torch

config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'save_path': './balanced_model_with_others.pth',
    'main_classes': ['Clothing', 'Bags', 'Shoes'],
    'subclass_to_idx': {
        'Dresses': 0, 
        'Skirts': 1, 
        'Outerwear': 2,
        'Shoulder Bags': 3, 
        'Tote Bags': 4, 
        'Clutches': 5,
        'Flats': 6, 
        'Boots': 7, 
        'High Heels': 8
    },
    'subclass_to_main': [0, 0, 0, 1, 1, 1, 2, 2, 2],
    'mainclass_threshold': 0.7,
    'subclass_threshold': 0.7
}

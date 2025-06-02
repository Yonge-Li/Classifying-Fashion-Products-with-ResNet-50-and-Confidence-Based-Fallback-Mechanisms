import os
import torch

# ==================== Configuration ====================
config = {
    'data_root': '/home/liyo23ac/project/DFdata',
    'main_classes': ['Clothing', 'Bags', 'Shoes'],
    'batch_size': 32,
    'num_epochs': 60,
    'lr': 0.001,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'save_path': './best_balanced_hierarchical_model.pth',
    'subclass_threshold': 0.7,
    'main_class_threshold': 0.7,
    'max_samples_per_subclass': 2000,
    'test_size': 0.2,
    'visualization_dir': './visualizations',
    'other_class_name': 'Other/Unknown'
}

def setup_visualization_dir():
    """Ensure the visualization output directory is writable."""
    try:
        os.makedirs(config['visualization_dir'], exist_ok=True)
        test_file = os.path.join(config['visualization_dir'], 'permission_test.txt')
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print("Visualization directory setup complete.")
    except Exception as e:
        print(f"Failed to setup visualization directory: {str(e)}")
        exit()

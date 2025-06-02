#  Hierarchical Image Classification with PyTorch

This project implements a **hierarchical classification system** for fashion product images. It classifies an image into both **main categories** (e.g., Clothing, Bags, Shoes) and more specific **subcategories** (e.g., Dresses, Boots, Tote Bags). The model is built with PyTorch and ResNet50 and supports interactive inference via Jupyter widgets.

## Project Structure

```
├── data/                      # Data processing and dataset
│   ├── processing/           # Scripts for dataset filtering
│   │   ├── subset_creator.py
│   ├── dataset.py            # Custom Dataset and Dataloader
│
├── models/
│   └── hierarchical_resnet.py  # ResNet50 with dual-head for hierarchical classification

├── train/
│   ├── train.py              # Training logic
│   └── evaluate.py           # Evaluation logic and reports

├── app/
│   ├── inference.py          # Core inference logic
│   └── widgets.py            # Interactive UI (Jupyter-based)

├── utils/
│   └── helpers.py            # Utility functions

├── visualizations/           # Output visualizations (e.g., confusion matrix)

├── configs/
│   └── config.py             # Centralized configuration

├── requirements.txt          # Dependency list
├── README.md                 # Project documentation
├── .gitignore                # Git ignored files
└── main.py                   # Optional entry point
```


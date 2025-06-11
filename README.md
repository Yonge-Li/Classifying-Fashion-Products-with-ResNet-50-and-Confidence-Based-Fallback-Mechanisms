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

## Result
![image](https://github.com/user-attachments/assets/07dc33d0-9abc-48c5-ba65-9f2c0e015d64)

Figure 1 - Phase 2 Training Accuracy Curves (Main & Subclass)
Trained 60 epochs, the best model was obtained with batch szie 32, epoch 35.

![image](https://github.com/user-attachments/assets/a9253e89-999e-40d3-bd7d-dac66e8d41aa)
Figure 2 shows the substantial improvement in subclass recall for rare categories after dataset expansion in Phase 2, demonstrating the effectiveness of our balancing strategy.


## Deployment

Interactive prototype result example

![image](https://github.com/user-attachments/assets/26f36080-3f0f-4aac-a990-5b4348d81e24)






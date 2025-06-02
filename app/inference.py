import torch
from PIL import Image
from torchvision import transforms
from configs.settings import config

# Image transform
transform = transforms.Compose([
    transforms.Lambda(lambda image: image.convert("RGB")),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model definition
class HierarchicalResNet(torch.nn.Module):
    def __init__(self, num_main_classes, num_sub_classes):
        super().__init__()
        self.base = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        num_ftrs = self.base.fc.in_features
        self.base.fc = torch.nn.Identity()
        self.main_classifier = torch.nn.Linear(num_ftrs, num_main_classes)
        self.sub_classifier = torch.nn.Linear(num_ftrs, num_sub_classes)

    def forward(self, x):
        features = self.base(x)
        return self.main_classifier(features), self.sub_classifier(features)

# Load model
def load_model():
    model = HierarchicalResNet(
        len(config['main_classes']),
        len(config['subclass_to_idx'])
    ).to(config['device'])
    model.load_state_dict(torch.load(config['save_path'], map_location=config['device']))
    model.eval()
    return model

# Predict function
def predict_image(model, image):
    image_tensor = transform(image).unsqueeze(0).to(config['device'])
    with torch.no_grad():
        main_out, sub_out = model(image_tensor)
        main_probs = torch.softmax(main_out, dim=1)[0]
        sub_probs = torch.softmax(sub_out, dim=1)[0]
    return main_probs, sub_probs

import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
import io

from app.inference import load_model, predict_image
from configs.settings import config

# Mapping
idx_to_subclass = {v: k for k, v in config['subclass_to_idx'].items()}
model = load_model()

# UI elements
uploader = widgets.FileUpload(accept='.jpg,.jpeg,.png', multiple=False, description='Upload Image')
out = widgets.Output()
display(uploader, out)

def on_classify_click(b):
    with out:
        clear_output()
        if not uploader.value:
            print("Please upload an image first.")
            return
        file_info = uploader.value[0]
        image = Image.open(io.BytesIO(file_info['content']))
        display(image.resize((224, 224)))

        try:
            main_probs, sub_probs = predict_image(model, image)

            main_conf, main_pred = torch.max(main_probs, 0)
            sub_conf, sub_pred = torch.max(sub_probs, 0)

            main_class = config['main_classes'][main_pred.item()]
            sub_class = idx_to_subclass.get(sub_pred.item(), "Unknown Subclass")

            print("\n=== Prediction Results ===")
            if main_conf.item() < config['mainclass_threshold']:
                print(f"Main class confidence {main_conf.item():.4f} < threshold.")
                print("Final classification: Unknown / Others")
            else:
                valid_subclass = config['subclass_to_main'][sub_pred.item()] == main_pred.item()
                meets_threshold = sub_conf.item() >= config['subclass_threshold']
                print(f"Main class: {main_class} (Conf: {main_conf.item():.4f})")
                print(f"Subclass: {sub_class} (Conf: {sub_conf.item():.4f})")
                if valid_subclass and meets_threshold:
                    print("Final classification:", f"{main_class} -> {sub_class}")
                else:
                    print("Final classification:", main_class)

            print("\nMain class probabilities:")
            for i, p in enumerate(main_probs):
                print(f"{config['main_classes'][i]}: {p.item():.4f}")
            print("\nTop 3 subclass probabilities:")
            top_vals, top_ids = torch.topk(sub_probs, 3)
            for val, idx in zip(top_vals, top_ids):
                name = idx_to_subclass.get(idx.item(), "Unknown")
                print(f"{name}: {val.item():.4f}")

        except Exception as e:
            print(f"Error: {e}")

classify_btn = widgets.Button(description="Classify Image")
classify_btn.on_click(on_classify_click)
display(classify_btn)

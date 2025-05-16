import torch
import torchvision.transforms as T
import torch.nn.functional as F
from PIL import Image

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define class names
class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    model.to(device)
    return model

def preprocess_image(image_path):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def get_probabilities(logits):
    probabilities = F.softmax(logits, dim=1)
    return probabilities * 100

def predict_face_shape(image_path, model):
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():  # Use no_grad for inference
        outputs = model(image_tensor)
        percentages = get_probabilities(outputs)
        
        # Get the predicted class from the probabilities (percentages)
        _, predicted_class = torch.max(percentages, 1)
    
    predicted_label = class_names[predicted_class.item()]
    
    # Format the probabilities (sorted by highest to lowest)
    result = {class_names[i]: round(percentages[0, i].item(), 2) for i in range(len(class_names))}
    sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    
    # Return the predicted label and sorted probabilities
    return {"predicted": predicted_label, "probabilities": sorted_result}

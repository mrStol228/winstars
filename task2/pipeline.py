import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from transformers import pipeline
from image_classification_model.model_inference import AnimalCNN

def load_ner_pipeline(bert_model):
    return pipeline("ner", model=bert_model, tokenizer=bert_model)

def extract_animal(text, ner_pipe, valid_animals):
    entities = ner_pipe(text)
    for ent in entities:
        candidate = ent["word"].replace("##", "").lower()
        if candidate in valid_animals:
            return candidate
    return None

def preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  
    return image

def load_image_model(model_path, num_classes, device):
    model = AnimalCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def classify_image(model, image_tensor, class_names, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probs).item()
    return class_names[predicted_idx] if class_names else str(predicted_idx)

def main():
    parser = argparse.ArgumentParser(description="Animal Detection Pipeline: Text & Image")
    parser.add_argument("--text", type=str, required=True,
                        help="Input text that may contain an animal name")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the image classifier model (.pth file)")
    parser.add_argument("--classes", type=str, required=True,
                        help="Path to JSON file containing the list of animal class names")
    parser.add_argument("--image_size", type=int, default=128,
                        help="Size to which input images will be resized (default: 128)")
    parser.add_argument("--bert_model", type=str, default="dslim/bert-base-NER",
                        help="BERT model to use for NER (default: dslim/bert-base-NER)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cpu or cuda)")
    
    args = parser.parse_args()

    valid_animals = set(a.strip().lower() for a in args.valid_animals.split(","))
    print(valid_animals)
    ner_pipe = load_ner_pipeline(args.bert_model)

    animal_from_text = extract_animal(args.text, ner_pipe, valid_animals)
    if animal_from_text is None:
        print("0")  
        return
    print(f"Animal found in text: {animal_from_text}")

    with open(args.classes, "r") as f:
        class_names = json.load(f)

    image_tensor = preprocess_image(args.image, args.image_size)
    device = args.device
    image_model = load_image_model(args.model, num_classes=len(class_names), device=device)

    animal_from_image = classify_image(image_model, image_tensor, class_names, device)
    print(f"Animal found in image: {animal_from_image}")

    if animal_from_text.lower() == animal_from_image.lower():
        print("1")
    else:
        print("0")

if __name__ == "__main__":
    main()
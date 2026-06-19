import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import io

class DiseaseDetector:
    def __init__(self, model_path='best.pt'):
        self.model_path = model_path
        self.model = None
        self.disease_info = self._initialize_disease_info()
        self.transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        try:
            # Load YOLOv8 model directly from local file
            self.model = YOLO(self.model_path)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def preprocess_image(self, image):
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        return self.transform(image).unsqueeze(0)

    def detect_disease(self, image):
        if self.model is None:
            if not self.load_model():
                return None

        try:
            # YOLO expects raw images, not preprocessed tensors
            results = self.model(image)
            
            # Check if any detections were found
            if len(results) == 0 or len(results[0].boxes.data) == 0:
                return None

            # Get the prediction with highest confidence
            best_pred = max(results[0].boxes.data, key=lambda x: x[4])
            disease_name = self.model.names[int(best_pred[5])]
            confidence = float(best_pred[4])

            return {
                'disease_name': disease_name,
                'confidence': confidence,
                'bounding_box': best_pred[:4].tolist(),
                'info': self.get_disease_info(disease_name)
            }
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return None

    def get_disease_info(self, disease_name):
        # Try exact match first
        if disease_name in self.disease_info:
            return self.disease_info[disease_name]
        
        # Try case-insensitive match
        for key in self.disease_info.keys():
            if key.lower() == disease_name.lower():
                return self.disease_info[key]
        
        # Return default if no match found
        return {
            'description': 'Information not available',
            'symptoms': [],
            'treatment': [],
            'prevention': []
        }

    def _initialize_disease_info(self):
        return {
        'Apple Scab Leaf': {
            'description': 'A fungal disease that causes dark, scaly lesions on apple tree leaves and fruit.',
            'symptoms': ['Dark olive-green spots on leaves', 'Black, scaly spots on fruit', 'Deformed fruit'],
            'treatment': ['Apply fungicides early in the growing season', 'Remove infected leaves'],
            'prevention': ['Plant resistant varieties', 'Improve air circulation', 'Clean up fallen leaves']
        },
        'Apple leaf (Healthy)': {
            'description': 'A healthy apple leaf showing normal characteristics.',
            'symptoms': ['Vibrant green color', 'No spots or lesions', 'Normal leaf shape'],
            'treatment': [],
            'prevention': ['Regular watering', 'Proper fertilization', 'Good air circulation']
        },
        'Apple Rust Leaf': {
            'description': 'A fungal disease causing rust-colored spots on apple leaves.',
            'symptoms': ['Rust-colored spots on leaves', 'Premature leaf drop'],
            'treatment': ['Apply fungicides', 'Remove infected leaves'],
            'prevention': ['Plant resistant varieties', 'Maintain good air circulation']
        },
        'Bell Pepper Leaf Spot': {
            'description': 'A bacterial infection causing spots on bell pepper leaves.',
            'symptoms': ['Small, dark spots on leaves', 'Yellowing around spots'],
            'treatment': ['Apply copper-based bactericides', 'Remove infected leaves'],
            'prevention': ['Avoid overhead watering', 'Use disease-free seeds']
        },
        'Bell Pepper Leaf (Healthy)': {
            'description': 'A healthy bell pepper leaf showing normal characteristics.',
            'symptoms': ['Vibrant green color', 'No spots or lesions'],
            'treatment': [],
            'prevention': ['Proper fertilization', 'Regular watering']
        },
        'Blueberry Leaf': {
            'description': 'A healthy blueberry leaf showing normal characteristics.',
            'symptoms': ['Bright green color', 'No signs of disease'],
            'treatment': ['Maintain proper soil pH', 'Apply balanced fertilizer'],
            'prevention': ['Proper watering', 'Good air circulation', 'Regular pruning']
        },
        'Cherry Leaf': {
            'description': 'A healthy cherry leaf showing normal characteristics.',
            'symptoms': ['Green, vibrant leaves', 'No lesions or spots'],
            'treatment': [],
            'prevention': ['Avoid water stress', 'Ensure proper soil nutrition']
        },
        'Corn Gray Leaf Spot': {
            'description': 'A fungal disease causing gray lesions on corn leaves.',
            'symptoms': ['Gray rectangular lesions on leaves', 'Reduced photosynthesis'],
            'treatment': ['Apply fungicides', 'Rotate crops'],
            'prevention': ['Improve air circulation', 'Use disease-resistant varieties']
        },
        'Corn Leaf Blight': {
            'description': 'A fungal infection that causes blight on corn leaves.',
            'symptoms': ['Brown, elongated lesions on leaves', 'Leaf death in severe cases'],
            'treatment': ['Apply fungicides', 'Crop rotation'],
            'prevention': ['Use resistant varieties', 'Maintain proper field hygiene']
        },
        'Corn Rust Leaf': {
            'description': 'A fungal disease causing rust-colored pustules on corn leaves.',
            'symptoms': ['Rust-colored pustules on leaves', 'Leaf yellowing'],
            'treatment': ['Apply fungicides', 'Remove infected plant debris'],
            'prevention': ['Plant resistant varieties', 'Ensure proper field sanitation']
        },
        'Peach Leaf': {
            'description': 'A healthy peach leaf showing normal characteristics.',
            'symptoms': ['Bright green, smooth texture', 'No spots or lesions'],
            'treatment': [],
            'prevention': ['Regular pruning', 'Avoid overwatering']
        },
        'Potato Leaf Early Blight': {
            'description': 'A fungal disease that causes brown spots with concentric rings on potato leaves.',
            'symptoms': ['Brown spots with rings', 'Leaf curling'],
            'treatment': ['Apply fungicides', 'Remove infected leaves'],
            'prevention': ['Use disease-free seeds', 'Crop rotation']
        },
        'Potato Leaf Late Blight': {
            'description': 'A severe fungal disease causing dark, water-soaked lesions on potato leaves.',
            'symptoms': ['Dark, water-soaked lesions', 'Rapid leaf decay'],
            'treatment': ['Apply fungicides', 'Remove infected plants'],
            'prevention': ['Use resistant varieties', 'Avoid overhead watering']
        },
        'Potato Leaf (Healthy)': {
            'description': 'A healthy potato leaf showing no signs of disease.',
            'symptoms': ['Bright green color', 'No spots or lesions'],
            'treatment': [],
            'prevention': ['Ensure balanced nutrition', 'Avoid excess moisture']
        },
        'Raspberry Leaf': {
            'description': 'A healthy raspberry leaf showing normal characteristics.',
            'symptoms': ['Bright green color', 'No spots or lesions'],
            'treatment': [],
            'prevention': ['Proper watering', 'Good air circulation']
        },
        'Soybean Leaf': {
            'description': 'A healthy soybean leaf showing normal characteristics.',
            'symptoms': ['Vibrant green color', 'No signs of disease'],
            'treatment': [],
            'prevention': ['Proper fertilization', 'Crop rotation']
        },
        'Squash Powdery Mildew Leaf': {
            'description': 'A fungal disease causing white powdery spots on squash leaves.',
            'symptoms': ['White powdery spots on leaves', 'Leaf yellowing'],
            'treatment': ['Apply fungicides', 'Remove infected leaves'],
            'prevention': ['Improve air circulation', 'Avoid overhead watering']
        },
        'Strawberry Leaf': {
            'description': 'A healthy strawberry leaf showing normal characteristics.',
            'symptoms': ['Dark green color', 'No spots or lesions'],
            'treatment': [],
            'prevention': ['Proper spacing', 'Regular watering']
        },
        'Tomato Early Blight Leaf': {
            'description': 'A fungal disease causing brown spots with concentric rings on tomato leaves.',
            'symptoms': ['Brown spots with rings', 'Yellowing leaves'],
            'treatment': ['Apply fungicides', 'Remove infected leaves'],
            'prevention': ['Crop rotation', 'Proper plant spacing']
        },
        'Tomato Septoria Leaf Spot': {
            'description': 'A fungal disease causing small brown spots on tomato leaves.',
            'symptoms': ['Small brown circular spots', 'Yellowing of leaves'],
            'treatment': ['Apply fungicides', 'Remove infected leaves'],
            'prevention': ['Use resistant varieties', 'Avoid overhead watering']
        },
        'Tomato Leaf Bacterial Spot': {
            'description': 'A bacterial infection causing water-soaked lesions on tomato leaves.',
            'symptoms': ['Water-soaked dark spots', 'Leaf curling'],
            'treatment': ['Apply copper-based bactericides', 'Remove infected leaves'],
            'prevention': ['Use disease-free seeds', 'Ensure proper air circulation']
        },
        'Tomato Leaf Mosaic Virus': {
            'description': 'A viral disease that causes mottling and distortion of tomato leaves.',
            'symptoms': ['Yellow-green mottling', 'Leaf curling and distortion'],
            'treatment': ['No cure; remove infected plants'],
            'prevention': ['Use virus-resistant varieties', 'Control insect vectors']
        },
        'Tomato Leaf Yellow Virus': {
            'description': 'A viral disease causing yellowing and curling of tomato leaves.',
            'symptoms': ['Yellowing and curling of leaves', 'Stunted growth'],
            'treatment': ['No cure; remove infected plants'],
            'prevention': ['Control whiteflies', 'Use virus-resistant varieties']
        },
        'Tomato Leaf (Healthy)': {
            'description': 'A healthy tomato leaf showing normal characteristics.',
            'symptoms': ['Vibrant green color', 'No spots or lesions'],
            'treatment': [],
            'prevention': ['Proper fertilization', 'Regular watering']
        },
        'Tomato Leaf Late Blight': {
            'description': 'A severe fungal disease causing dark, water-soaked lesions on tomato leaves.',
            'symptoms': ['Dark, water-soaked lesions', 'Rapid leaf decay'],
            'treatment': ['Apply fungicides', 'Remove infected plants'],
            'prevention': ['Use resistant varieties', 'Avoid overhead watering']
        },
        'Tomato Mold Leaf': {
            'description': 'A fungal disease causing moldy patches on tomato leaves.',
            'symptoms': ['Gray mold patches on leaves', 'Leaf wilting'],
            'treatment': ['Apply fungicides', 'Improve air circulation'],
            'prevention': ['Avoid excess humidity', 'Remove infected leaves']
        },
        'Tomato Two-Spotted Spider Mites Leaf': {
            'description': 'A pest infestation causing stippling and webbing on tomato leaves.',
            'symptoms': ['Yellow stippling', 'Fine webbing on leaves', 'Leaf discoloration'],
            'treatment': ['Apply miticides', 'Spray with water to dislodge mites'],
            'prevention': ['Encourage natural predators', 'Avoid plant stress']
        },
        'Grape Leaf Black Rot': {
            'description': 'A fungal disease that causes black spots on grape leaves.',
            'symptoms': ['Black spots on leaves', 'Shriveled grapes', 'Leaf browning'],
            'treatment': ['Apply fungicides', 'Remove infected fruit'],
            'prevention': ['Ensure proper spacing', 'Use disease-resistant varieties']
        },
        'Grape Leaf': {
            'description': 'A healthy grape leaf showing normal characteristics.',
            'symptoms': ['Bright green, normal leaf structure', 'No spots or lesions'],
            'treatment': [],
            'prevention': ['Proper pruning', 'Regular watering', 'Adequate sunlight']
        }
    }
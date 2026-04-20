import io
import os
import gc
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Optimize TensorFlow memory usage for Render Free Tier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

# -------------------------------
# 1. Initialize App & CORS
# -------------------------------
app = FastAPI(title="CIFAR-100 Image Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo purposes, matching frontend deployed on alternative URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 2. Lazy Model Loader
# -------------------------------
model = None

def get_model():
    global model
    if model is not None:
        return model
    
    # Check if model file exists before trying to load
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "model", "cifar100_finetuned.h5"),
        os.path.join(os.path.dirname(__file__), "cifar100_finetuned.h5"),
        "model/cifar100_finetuned.h5",
        "backend/model/cifar100_finetuned.h5",
        "cifar100_finetuned.h5",
        "../cifar100_finetuned.h5"
    ]

    found_model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_model_path = path
            break

    if not found_model_path:
        print(f"CRITICAL: Model file not found. Checked: {possible_paths}")
        return None
    
    print(f"Loading heavy model from {found_model_path} (Neural Labs initialization)...")
    try:
        model = load_model(found_model_path)
        print("Model successfully cached in memory.")
        # Explicitly collect garbage after heavy loading
        gc.collect()
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

# CIFAR-10 class names
CLASS_NAMES = [
    "Apple", "Aquarium Fish", "Baby", "Bear", "Beaver", "Bed", "Bee", "Beetle", 
    "Bicycle", "Bottle", "Bowl", "Boy", "Bridge", "Bus", "Butterfly", "Camel", 
    "Can", "Castle", "Caterpillar", "Cattle", "Chair", "Chimpanzee", "Clock", 
    "Cloud", "Cockroach", "Couch", "Crab", "Crocodile", "Cup", "Dinosaur", 
    "Dolphin", "Elephant", "Flatfish", "Forest", "Fox", "Girl", "Hamster", 
    "House", "Kangaroo", "Keyboard", "Lamp", "Lawn Mower", "Leopard", "Lion", 
    "Lizard", "Lobster", "Man", "Maple Tree", "Motorcycle", "Mountain", "Mouse", 
    "Mushroom", "Oak Tree", "Orange", "Orchid", "Otter", "Palm Tree", "Pear", 
    "Pickup Truck", "Pine Tree", "Plain", "Plate", "Poppy", "Porcupine", 
    "Possum", "Rabbit", "Raccoon", "Ray", "Road", "Rocket", "Rose", "Sea", 
    "Seal", "Shark", "Shrew", "Skunk", "Skyscraper", "Snail", "Snake", "Spider", 
    "Squirrel", "Streetcar", "Sunflower", "Sweet Pepper", "Table", "Tank", 
    "Telephone", "Television", "Tiger", "Tractor", "Train", "Trout", "Tulip", 
    "Turtle", "Wardrobe", "Whale", "Willow Tree", "Wolf", "Woman", "Worm"
]

IMG_SIZE = 96 

# -------------------------------
# 3. Helper: Process Uploaded Image
# -------------------------------
def read_image(file_data: bytes):
    try:
        image = Image.open(io.BytesIO(file_data))
        image = image.convert("RGB") # Ensure 3 channels
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(image, dtype=np.float32)
        image_array = preprocess_input(image_array)
        image_array = np.expand_dims(image_array, axis=0) # Batch dimension
        return image_array
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

# -------------------------------
# 4. Prediction Endpoint
# -------------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file content immediately (UploadFile is a generator/spooled file)
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")

    # Process the image
    try:
        img = read_image(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Make prediction
    try:
        # Lazily load model if not already cached
        current_model = get_model()
        if current_model is None:
             # Fallback for testing if model is missing
             print("Predict called but model could not be loaded")
             return {"class": "Model Missing/OOM", "confidence": 0.0}

        preds = current_model.predict(img)
        pred_class_index = np.argmax(preds[0])
        confidence = float(preds[0][pred_class_index])
        
        return {
            "class": CLASS_NAMES[pred_class_index], 
            "confidence": confidence
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -------------------------------
# 5. Root Endpoint
# -------------------------------
@app.get("/")
def read_root():
    # Use get_model status check without triggering a full load if possible
    is_loaded = (model is not None)
    return {
        "message": "CIFAR-100 API is running", 
        "model_status": "loaded" if is_loaded else "standby",
        "working_dir": os.getcwd()
    }
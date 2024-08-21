from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from cnn_model import load_model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model_path = "./MINST_saved_model.pth"
model, optimizer = load_model(model_path, device)

# Define your Flask app
app = Flask(__name__)

# Image preprocessing pipeline
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Resize to match model input size
        transforms.ToTensor(),  # Convert to a PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Standardize
    ])
    return transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

# Home route serving the HTML page
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# Route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files["image"].read()
    image = Image.open(io.BytesIO(image_file)).convert("L")  # Convert to grayscale

    # Preprocess the image for model input
    image_tensor = preprocess_image(image)

    # Make a prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    return jsonify({"prediction": predicted.item()})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=3050)

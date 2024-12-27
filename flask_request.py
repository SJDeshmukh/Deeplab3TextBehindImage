import cv2
import numpy as np
import torch
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from torchvision import transforms, models

app = Flask(__name__)
CORS(app)

# Load pre-trained DeepLabV3 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.segmentation.deeplabv3_resnet101(weights='DEFAULT').to(device)
model.eval()

# Function to process the image
def process_image(image):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    
    mask = output.argmax(0).byte().cpu().numpy()

    # Apply mask to remove background
    mask = (mask > 0).astype(np.uint8) * 255
    object_mask = cv2.bitwise_and(image, image, mask=mask)

    # Create a transparent image with alpha channel
    result_with_alpha = cv2.cvtColor(object_mask, cv2.COLOR_BGR2BGRA)
    result_with_alpha[:, :, 3] = mask

    # Convert image to base64
    _, buffer = cv2.imencode('.png', result_with_alpha)
    img_bytes = buffer.tobytes()
    return base64.b64encode(img_bytes).decode('utf-8')

# Route to upload and process image
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    try:
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        processed_image = process_image(img)
        return jsonify({"result": processed_image}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()

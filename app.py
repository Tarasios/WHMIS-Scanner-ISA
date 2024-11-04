import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from transformers import pipeline
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the summarization model
summarizer = pipeline('summarization', model='t5-base', tokenizer='t5-base')

# Placeholder for pictogram model
def load_pictogram_model():
    return None

pictogram_model = load_pictogram_model()

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess image for OCR
def preprocess_image(image_path):
    # Read image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Save preprocessed image
    preprocessed_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        'preprocessed_' + os.path.basename(image_path)
    )
    cv2.imwrite(preprocessed_path, thresh)

    return preprocessed_path

# Extract text using Tesseract OCR
def extract_text(image_path):
    # Perform OCR
    raw_text = pytesseract.image_to_string(Image.open(image_path))
    return raw_text

# Detect pictograms in the image
def detect_pictograms(image_path):
    # Placeholder function since the model isn't available
    return ['Mock Pictogram']

# Summarize text
def summarize_text(text):
    if not text.strip():
        return "No text detected."

    # Handle long texts by splitting them into smaller chunks
    max_chunk = 500
    text = text.replace('\n', ' ')
    chunks = [text[i:i + max_chunk] for i in range(0, len(text), max_chunk)]
    summary = ''
    for chunk in chunks:
        summarized = summarizer(
            chunk,
            max_length=150,
            min_length=40,
            do_sample=False
        )[0]['summary_text']
        summary += summarized + ' '
    return summary.strip()

# Define API endpoint
@app.route('/process_label', methods=['POST'])
def process_label():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Preprocess image
        preprocessed_image_path = preprocess_image(image_path)

        # Extract text
        text = extract_text(preprocessed_image_path)

        # Detect pictograms
        pictograms = detect_pictograms(image_path)

        # Summarize text
        summary = summarize_text(text)

        # Prepare response
        result = {
            'summary': summary,
            'pictograms': pictograms,
            'full_text': text
        }

        # Clean up files
        os.remove(image_path)
        os.remove(preprocessed_image_path)

        return jsonify(result), 200
    else:
        return jsonify({'error': 'Allowed image types are png, jpg, jpeg'}), 400

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

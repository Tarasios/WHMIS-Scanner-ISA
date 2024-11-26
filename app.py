import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pytesseract
from flask_cors import CORS
import re
from spellchecker import SpellChecker
import traceback
import logging
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Configure CORS to allow all origins
CORS(app, resources={r"/*": {"origins": ["https://isa-lab1-69jry.ondigitalocean.app", "https://tarasios.ca/public/whmis.html"]}})


# Configure logging
logging.basicConfig(level=logging.DEBUG)

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize the spell checker
spell = SpellChecker()

# Load the summarization model
summarizer = pipeline('summarization', model='t5-base', tokenizer='t5-base')

# Placeholder for pictogram detection
def detect_pictograms(image_path):
    # Placeholder implementation
    return ['Pictogram1', 'Pictogram2']

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

# Resize image to a maximum width while maintaining aspect ratio
def resize_image(image_path, max_width=800):
    image = Image.open(image_path)
    width_percent = (max_width / float(image.size[0]))
    height_size = int((float(image.size[1]) * float(width_percent)))
    resized_image = image.resize((max_width, height_size), Image.LANCZOS)
    resized_image.save(image_path)
    return image_path

# Preprocess image for OCR
def preprocess_image(image_path):
    # Resize image to reduce processing time
    image_path = resize_image(image_path)

    # Read image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    # Save preprocessed image
    preprocessed_path = os.path.join(
        app.config['UPLOAD_FOLDER'],
        'preprocessed_' + os.path.basename(image_path)
    )
    cv2.imwrite(preprocessed_path, thresh)

    return preprocessed_path

# Correct spelling errors in the text
def correct_spelling(text):
    words = text.split()
    misspelled = spell.unknown([word for word in words if len(word) > 2])

    corrected_words = []
    for word in words:
        if word in misspelled:
            corrected = spell.correction(word)
            if corrected is not None:
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)  # Use the original word if no correction is found
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

# Extract text using Tesseract OCR
def extract_text(image_path):
    raw_text = pytesseract.image_to_string(Image.open(image_path))
    return raw_text

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

        try:
            # Preprocess image
            preprocessed_image_path = preprocess_image(image_path)

            # Extract text
            text = extract_text(preprocessed_image_path)

            # Correct spelling errors
            corrected_text = correct_spelling(text)

            # Summarize text
            summary = summarize_text(corrected_text)

            # Detect pictograms (placeholder function)
            pictograms = detect_pictograms(image_path)

            # Prepare response
            result = {
                'summary': summary,
                'pictograms': pictograms,
                'full_text': corrected_text
            }

        except Exception as e:
            # Log the exception details
            app.logger.error(f"Exception occurred: {e}")
            traceback.print_exc()

            # Return error response
            result = {
                'error': f'An error occurred: {str(e)}'
            }
            return jsonify(result), 500

        finally:
            # Clean up files
            if os.path.exists(image_path):
                os.remove(image_path)
            if 'preprocessed_image_path' in locals() and os.path.exists(preprocessed_image_path):
                os.remove(preprocessed_image_path)

        return jsonify(result), 200
    else:
        return jsonify({'error': 'Allowed image types are png, jpg, jpeg'}), 400

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

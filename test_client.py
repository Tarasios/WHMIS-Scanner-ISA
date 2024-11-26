import requests

# API endpoint
url = 'http://127.0.0.1:5000/process_label'

# Image file to send
image_path = 'gas.png'  # Replace with your image path
image_file = {'image': open(image_path, 'rb')}

# Send POST request
response = requests.post(url, files=image_file)

# Check response
if response.status_code == 200:
    data = response.json()
    print("Formatted Output:")
    print(data['formatted_output'])
    print("\nDetected Pictograms:")
    print(data['pictograms'])
    print("\nFull Text:")
    print(data['full_text'])
else:
    print("Error:", response.status_code)
    print(response.json())

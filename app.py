import random
import urllib

import cloudinary
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from cloudinary.uploader import upload
from flask import Flask, request, jsonify
from scipy.spatial.distance import cosine

app = Flask(__name__)

# Initialize a list to store similarity scores and pairs
matching_pairs = []

# Load the pre-trained ResNet model with 18 layers
resnet18 = models.resnet18(pretrained=True)

# Set the model to evaluation mode
resnet18.eval()

dress_images = []
top_images = []
below_torso_images = []

# Cloudinary configuration
cloudinary.config(
    cloud_name="dhvjpt6ii",
    api_key="767734596527217",
    api_secret="wvsX0w1G1AIkfN9-k0R9s2FH_ks"
)


# Load and preprocess cloth images
def preprocess_image(image_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
    if image_path.startswith("http://") or image_path.startswith("https://"):
        req = urllib.request.Request(image_path, headers=headers)
        # If it's a URL, open the image from the URL and convert it to a NumPy array
        with urllib.request.urlopen(req) as url:
            img = Image.open(url)
            img = img.resize((224, 224))  # Resize to your desired size
            img = np.array(img)  # Convert to NumPy array
    else:
        # If it's a local file path, open it directly
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize to your desired size
        img = np.array(img)  # Convert to NumPy array

    # Normalize and prepare the image for PyTorch
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension

    return input_tensor


# Calculate cosine similarity between feature vectors
def calculate_cosine_similarity(features1, features2):
    return cosine(features1, features2)


# Calculate color similarity using a suitable metric
def calculate_color_similarity(hist1, hist2):
    # Check if either histogram is None
    if hist1 is None or hist2 is None:
        return 0.0  # Return a similarity score of 0 if either histogram is None

    # Ensure that both histograms have the same type and depth
    hist1 = hist1.astype(np.float32)
    hist2 = hist2.astype(np.float32)

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)  # Example: Using Histogram Intersection


# Calculate color histogram
def calculate_color_histogram(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception("Failed to load image from path: " + image_path)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert image to HSV color space

        # Calculate histograms for the H, S, and V channels
        h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX)
        s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX)
        v_hist = cv2.normalize(v_hist, v_hist, 0, 1, cv2.NORM_MINMAX)

        # Concatenate the normalized histograms to create a single histogram
        color_hist = np.concatenate((h_hist, s_hist, v_hist), axis=None)

        return color_hist
    except Exception as e:
        # Handle the exception, log it, and return an error value or message.
        print("Error:", str(e))
        return None


def train_model():
    # Initialize variables to store best match information
    best_match_index1, best_match_index2 = None, None
    best_similarity_score = -1

    # Load and preprocess all cloth images
    input_tensors1 = [preprocess_image(image_path1) for image_path1 in top_images]
    input_tensors2 = [preprocess_image(image_path2) for image_path2 in below_torso_images]

    # Extract features from the pre-trained model for each cloth image
    with torch.no_grad():
        features_list1 = [resnet18(input_tensor).squeeze().numpy() for input_tensor in input_tensors1]
        features_list2 = [resnet18(input_tensor).squeeze().numpy() for input_tensor in input_tensors2]

    # Compare all pairs of cloth images to find the best match
    for i in range(len(top_images)):
        hist1 = calculate_color_histogram(top_images[i])
        for j in range(len(below_torso_images)):
            hist2 = calculate_color_histogram(below_torso_images[j])
            feature_similarity = calculate_cosine_similarity(features_list1[i], features_list2[j])
            color_similarity = calculate_color_similarity(hist1, hist2)
            # Combine color and feature similarity metrics, e.g., using weighted average
            similarity_score = 0.9 * feature_similarity + 0.1 * color_similarity
            matching_pairs.append((i, j, similarity_score))
            if similarity_score > best_similarity_score:
                best_similarity_score = similarity_score
                best_match_index1 = i
                best_match_index2 = j

    # Sort the matching pairs based on the similarity score in descending order
    matching_pairs.sort(key=lambda x: x[2], reverse=True)
    return top_images[best_match_index1], below_torso_images[best_match_index2]


def find_best_match(input_image):
    # Initialize variables to store best match information
    best_match_index1, best_match_index2 = None, None
    best_similarity_score = -1

    # Load and preprocess all cloth images
    input_tensors1 = [preprocess_image(image_path1) for image_path1 in top_images]
    input_tensors2 = [preprocess_image(image_path2) for image_path2 in below_torso_images]

    # Extract features from the pre-trained model for each cloth image
    with torch.no_grad():
        features_list1 = [resnet18(input_tensor).squeeze().numpy() for input_tensor in input_tensors1]
        features_list2 = [resnet18(input_tensor).squeeze().numpy() for input_tensor in input_tensors2]

    # Extract color histograms for all cloth images
    color_histograms1 = [calculate_color_histogram(image_path1) for image_path1 in top_images]
    color_histograms2 = [calculate_color_histogram(image_path2) for image_path2 in below_torso_images]

    # Calculate the similarity between the input image and all cloth images
    for i in range(len(top_images)):
        feature_similarity = calculate_cosine_similarity(features_list1[i], input_image)
        color_similarity = calculate_color_similarity(color_histograms1[i], calculate_color_histogram(input_image))
        similarity_score = 0.9 * feature_similarity + 0.1 * color_similarity
        if similarity_score > best_similarity_score:
            best_similarity_score = similarity_score
            best_match_index1 = i

    # Return the best-matched top image
    return top_images[best_match_index1], below_torso_images[best_match_index1]


def display_matches():
    # Load and preprocess all cloth images
    input_tensors1 = [preprocess_image(image_path1) for image_path1 in top_images]
    input_tensors2 = [preprocess_image(image_path2) for image_path2 in below_torso_images]

    # Extract features from the pre-trained model for each cloth image
    with torch.no_grad():
        features_list1 = [resnet18(input_tensor).squeeze().numpy() for input_tensor in input_tensors1]
        features_list2 = [resnet18(input_tensor).squeeze().numpy() for input_tensor in input_tensors2]

    matching_pairs.clear()

    # Compare all pairs of cloth images to find the best match
    for i in range(len(top_images)):
        hist1 = calculate_color_histogram(top_images[i])
        for j in range(len(below_torso_images)):
            hist2 = calculate_color_histogram(below_torso_images[j])
            feature_similarity = calculate_cosine_similarity(features_list1[i], features_list2[j])
            color_similarity = calculate_color_similarity(hist1, hist2)
            # Combine color and feature similarity metrics, e.g., using weighted average
            similarity_score = 0.9 * feature_similarity + 0.1 * color_similarity
            matching_pairs.append((i, j, similarity_score))

    # Sort the matching pairs based on the similarity score in descending order
    matching_pairs.sort(key=lambda x: x[2], reverse=True)

    # Display the top matching pairs (adjust 'num_pairs_to_display' as needed)
    num_pairs_to_display = len(top_images) * len(below_torso_images)
    top_matches = []
    bottom_matches = []
    for i in range(num_pairs_to_display):
        if i == 5:
            break
        pair = matching_pairs[i]
        top_matches.append(top_images[pair[0]])
        bottom_matches.append(below_torso_images[pair[1]])
    return top_matches, bottom_matches


def random_dress_selection():
    # Display the best dress of cloth images
    selected_dress = random.choice(dress_images)
    return selected_dress


@app.route('/train', methods=['GET', 'POST'])
def train_clothing_model():
    global top_images
    global below_torso_images
    if request.method == 'GET':
        top_match, bottom_match = train_model()

        # Upload the best-matched images to Cloudinary
        top_match_url = upload(top_match)["secure_url"]
        bottom_match_url = upload(bottom_match)["secure_url"]

        return jsonify({"top_match_url": top_match_url, "bottom_match_url": bottom_match_url})

    elif request.method == 'POST':
        try:
            data = request.get_json()
            top_images = data.get('top_images', [])
            below_torso_images = data.get('below_torso_images', [])

            # Rest of your training code here

            return jsonify({"message": "Training completed successfully"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route('/process_image', methods=['POST'])
def process_image():
    global top_images

    try:
        data = request.get_json()
        image_url = data.get('image_url', "")

        # Convert the image URL to a local image
        input_image = preprocess_image(image_url)

        # Find the best match locally
        best_top_match, best_bottom_match = find_best_match(input_image)

        # Upload the best-matched images to Cloudinary
        top_match_url = upload(best_top_match)["secure_url"]
        bottom_match_url = upload(best_bottom_match)["secure_url"]

        return jsonify({"top_match_url": top_match_url, "bottom_match_url": bottom_match_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/matches', methods=['GET', 'POST'])
def display_clothing_matches():
    global top_images
    global below_torso_images
    if request.method == 'GET':
        top_matches, bottom_matches = display_matches()
        return jsonify({"top_matches": top_matches, "bottom_matches": bottom_matches})
    elif request.method == 'POST':
        try:
            data = request.get_json()
            top_images = data.get('tops', [])
            below_torso_images = data.get('bottoms', [])

            top_matches, bottom_matches = display_matches()
            return jsonify({"message": "Matching completed successfully",
                            "top_matches": top_matches,
                            "bottom_matches": bottom_matches})
            # Rest of your matching code here

        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route('/random_dress', methods=['GET'])
def select_random_dress():
    if not dress_images:
        return jsonify({"message": "No dresses available"})

    selected_dress = random.choice(dress_images)
    return jsonify({"selected_dress": selected_dress})


if __name__ == '__main__':
    app.run(debug=True)  # You can customize host and port if needed
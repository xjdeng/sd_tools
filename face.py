import cv2
import numpy as np

from path import Path as path
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import tempfile
import pandas as pd
from PIL import UnidentifiedImageError, Image

# Initialize MTCNN
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(img):
    if isinstance(img, str):
        try:
            img = Image.open(img).convert('RGB')
        except (OSError, UnidentifiedImageError):
            return None
    
    img_cropped, probs = mtcnn(img, return_prob=True)
    
    if img_cropped is None or probs is None:
        return None
    
    # Filter faces with probability > 0.9
    filtered_faces = [(face, prob, box) for face, prob, box in zip(img_cropped, probs, mtcnn.detect(img)[0]) if prob > 0.9]
    
    if len(filtered_faces) == 0:
        return None
    
    # Sort faces by area
    filtered_faces.sort(key=lambda x: (x[2][2] - x[2][0]) * (x[2][3] - x[2][1]), reverse=True)
    
    largest_face_area = (filtered_faces[0][2][2] - filtered_faces[0][2][0]) * (filtered_faces[0][2][3] - filtered_faces[0][2][1])
    
    if len(filtered_faces) == 1:
        largest_face_embedding = resnet(filtered_faces[0][0].unsqueeze(0).to(device)).detach().cpu().squeeze().tolist()
        return largest_face_embedding
    
    second_largest_face_area = (filtered_faces[1][2][2] - filtered_faces[1][2][0]) * (filtered_faces[1][2][3] - filtered_faces[1][2][1])
    
    if largest_face_area > 2 * second_largest_face_area:
        largest_face_embedding = resnet(filtered_faces[0][0].unsqueeze(0).to(device)).detach().cpu().squeeze().tolist()
        return largest_face_embedding
    
    return None

def extract_features(image):
    tmpdir = tempfile.TemporaryDirectory()
    if isinstance(image, str):
        img0 = image
        image = f"{tmpdir.name}/tmp.jpg"
        cv2.imwrite(image, cv2.imread(img0))
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
    # Detect faces and landmarks
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    
    if landmarks is None:
        return []
    
    features = []
    for landmark in landmarks:
        features.append(landmark.tolist())
    tmpdir.cleanup()
    return features

def variance_of_embeddings(embeddings):
    """
    Calculate the total variance of a set of embeddings.

    Args:
    embeddings (list of list or numpy.ndarray): A list of embeddings where each embedding
                                                is a list or NumPy array of the same dimensionality.

    Returns:
    float: The total variance of the embeddings.
    """
    # Convert the list of embeddings to a NumPy array for easier manipulation
    embeddings = np.array(embeddings)
    
    # Calculate the mean for each dimension
    mean_embedding = np.mean(embeddings, axis=0)
    
    # Calculate the variance for each dimension
    variance_per_dimension = np.var(embeddings, axis=0)
    
    # Sum the variances across all dimensions to get the total variance
    total_variance = np.sum(variance_per_dimension)
    
    return total_variance

def calculate_neoteny_score(features):
    if isinstance(features, str):
        features = extract_features(features)
    
    scores = []
    for landmarks in features:
        # Convert to numpy array for easier calculations
        landmarks = np.array(landmarks)

        # Eye size
        left_eye_width = np.linalg.norm(landmarks[0] - landmarks[1])
        right_eye_width = np.linalg.norm(landmarks[0] - landmarks[1])
        eye_size = (left_eye_width + right_eye_width) / 2
        face_width = np.linalg.norm(landmarks[0] - landmarks[1])

        # Since we don't have forehead, we'll assume it's above the eyes
        forehead_height = np.linalg.norm(landmarks[0] - landmarks[2])
        face_height = np.linalg.norm(landmarks[2] - landmarks[4])

        # Nose size
        nose_width = 0  # Not available in 5-point landmarks
        nose_length = np.linalg.norm(landmarks[2] - landmarks[4])

        # Lip fullness
        lip_thickness = np.linalg.norm(landmarks[3] - landmarks[4])

        # Chin size (assuming it's the distance from the mouth to the bottom)
        chin_size = np.linalg.norm(landmarks[4] - landmarks[3])

        # Cheekbone prominence (not available with 5-point landmarks)
        cheekbone_width = 0

        # Calculate ratios
        eye_ratio = eye_size / face_width if face_width else 0
        forehead_ratio = forehead_height / face_height if face_height else 0
        nose_ratio = nose_width / face_width + nose_length / face_height if face_width and face_height else 0
        lip_ratio = lip_thickness / face_width if face_width else 0
        chin_ratio = chin_size / face_height if face_height else 0
        cheekbone_ratio = cheekbone_width / face_width if face_width else 0

        # Combine ratios into a neoteny score
        score = (eye_ratio * 0.3) + (forehead_ratio * 0.2) - (nose_ratio * 0.2) + (lip_ratio * 0.1) - (chin_ratio * 0.1) + (cheekbone_ratio * 0.1)
        scores.append(score)

    return scores

def run_neoteny(img):
    try:
        ans = calculate_neoteny_score(extract_features(img))
    except ValueError as e:
        print(img)
        raise(e)
    if len(ans) == 0:
        print(img)
        return 0
    return max(ans)


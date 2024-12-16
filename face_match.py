import cv2
import dlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the images
image1 = cv2.imread('aadhar_card_photo.jpg')
image2 = cv2.imread('current_photo.jpg')

if image1 is None or image2 is None:
    print("Error: One or both images not found.")
    exit()

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib's website
face_rec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")  # Download from dlib's website

def get_face_embedding(image):
    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        print("No face detected in one of the images.")
        return None

    # Extract facial landmarks
    landmarks = predictor(gray, faces[0])

    # Compute face embedding
    face_descriptor = face_rec.compute_face_descriptor(image, landmarks)
    return np.array(face_descriptor)

# Get embeddings for both images
embedding1 = get_face_embedding(image1)
embedding2 = get_face_embedding(image2)

if embedding1 is None or embedding2 is None:
    print("Error: Could not extract face embeddings.")
    exit()

# Calculate cosine similarity
similarity = cosine_similarity([embedding1], [embedding2])[0][0]

# Convert to percentage
match_percentage = similarity * 100
print(f"Match Percentage: {match_percentage:.2f}%")

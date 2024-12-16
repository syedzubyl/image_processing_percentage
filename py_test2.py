from deepface import DeepFace
import cv2

def compare_images(image1_path, image2_path):
    try:
        # Load images
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        if img1 is None or img2 is None:
            print("Error: One or both image paths are invalid.")
            return

        # Analyze similarity using DeepFace
        result = DeepFace.verify(img1_path, image2_path, enforce_detection=False)
        
        # Extract match details
        match = result["verified"]
        confidence = result["distance"]  # Lower distance means higher similarity
        match_percentage = (1 - confidence) * 100  # Convert to percentage

        # Display result
        print(f"Match: {'Yes' if match else 'No'}")
        print(f"Match Percentage: {match_percentage:.2f}%")

        return match, match_percentage

    except Exception as e:
        print(f"Error: {str(e)}")
        return None

# Paths to Aadhaar photo and current photo
aadhaar_photo_path = "aadhaar_photo.jpg"
current_photo_path = "current_photo.jpg"

# Compare the two images
compare_images(aadhaar_photo_path, current_photo_path)

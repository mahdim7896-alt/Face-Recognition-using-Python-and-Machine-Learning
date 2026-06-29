"""
Group Photo Face Recognition

This script detects and recognizes known people in a group image using the
`face_recognition` library and OpenCV.

How it works:
1. Load known face images from the `data/known_faces` directory.
2. Extract face encodings for each known person.
3. Load the target group image.
4. Detect all faces in the group image.
5. Compare each detected face with the known encodings.
6. Draw bounding boxes and labels on recognized faces.
7. Save the final annotated image to the `output` directory.

This project is designed as a simple and clean demonstration of a face
recognition pipeline for portfolio and GitHub presentation purposes.
"""


import face_recognition as fr
import cv2
import numpy as np
import os
print("Current directory:", os.getcwd())
KNOWN_DIR = "data/known_faces"
GROUP_IMAGE = "data/group.jpg"

# --- Map image files to person names----------
PERSONS_MAP = {
    "moh.jpg": "Mohsen",
    "kom.jpg": "Komeil",
    "swer.jpg": "Amirreza",
    "am.jpg": "Amir Iravani",
    "jav.jpg": "Javad",
    "mah.jpg" :'mahdi'
}

#-----------------------------------------------------------------------
def load_known_faces():
    """Load known face encodings and their names from a directory."""
    encodings = []
    names = []

   
    for file_name, real_name in PERSONS_MAP.items():
        path = os.path.join(KNOWN_DIR,file_name)
        #print(os.listdir(KNOWN_DIR))
        if os.path.exists(path):
                image = fr.load_image_file(path)
                
                encoding = fr.face_encodings(image)[0]
                #print(file_name, len(encoding))
                if len(encoding) > 0:
                    encodings.append(encoding)
                    names.append(real_name) 
                
        else:
            print(f"Warning: File {file_name} not found in {KNOWN_DIR}")

    return encodings, names

#------------------------------------------------------------------------

def recognize_faces():
    """Detect and recognize faces in the input image."""
    known_encodings, known_names = load_known_faces()
    image = fr.load_image_file(GROUP_IMAGE)

    face_locations = fr.face_locations(image, model="cnn")
    face_encodings = fr.face_encodings(image, face_locations)


    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_encodings, face_encoding) #matches = for example [True, False, False]
        face_distances = fr.face_distance(known_encodings, face_encoding)
        print("Distances:", fr.face_distance(known_encodings, face_encoding))

        name = "Unknown"
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ------------- save and show ------------------------------
    if not os.path.exists("output"): os.makedirs("output")
    cv2.imwrite("output/result.jpg", img)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
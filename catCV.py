import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


gender_model = load_model('gender_detection_model.h5')

def predict_gender_age(face_img):
    # Preprocess the image
    face_img = tf.keras.preprocessing.image.img_to_array(face_img)
    face_img = np.expand_dims(face_img, axis=0)
    face_img /= 255.
    
    # Predict gender and age
    gender, age = gender_model.predict(face_img)
    gender = "Male" if gender > 0.5 else "Female"
    age = int(age[0][0])
    
    return gender, age

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def predict_gender_age_image(image):
    # Detect faces
    faces = detect_faces(image)
    
    # Predict gender and age for each face
    for (x,y,w,h) in faces:
        face_img = image[y:y+h, x:x+w]
        gender, age = predict_gender_age(face_img)
        
        # Draw a rectangle around the face and display gender and age
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f"{gender}, {age}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Display the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
# Load the image
image = cv2.imread('example_image.jpg')


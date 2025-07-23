from keras.models import load_model
import cv2
#from keras.preprocessing import image


model = load_model('emotion_recognition_model.keras')

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

def preprocess_img(frame, target_size=(64,64)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    gray = gray / 255.0
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)
    return gray

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        processed_face = preprocess_img(face_img)
        
        emotion_prediction = model.predict(processed_face)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    cv2.imshow('Emotion Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
        
    
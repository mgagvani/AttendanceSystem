from deepface import DeepFace
from stream_face_identification import *
from headless_id import *
import tqdm

MODELS = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace",
  "Dlib", 
  "SFace",
]

BACKENDS = [
  'opencv', 
  'ssd', 
  'dlib', 
  'mtcnn', 
  'retinaface', 
  'mediapipe',
  'yolov8',
  'yunet',
  'fastmtcnn',
]

METRICS = [
    "cosine",
    "euclidean",
    "euclidean_l2"
]

DeepFace.stream(
    db_path='/Users/pranavv/Library/CloudStorage/GoogleDrive-1823210@fcpsschools.net/My Drive/Machine Learning/Facial Recognition/AttendanceSystem/Face_Identification/test2',
    model_name=MODELS[2],
    detector_backend=BACKENDS[4],
    distance_metric=METRICS[2],
    enable_face_analysis=False,
    source='/Users/pranavv/Library/CloudStorage/GoogleDrive-1823210@fcpsschools.net/My Drive/Machine Learning/Facial Recognition/AttendanceSystem/Face_Identification/face_test.MOV',
    time_threshold=2,
    frame_threshold=8
)
            

'/Users/pranavv/Library/CloudStorage/GoogleDrive-1823210@fcpsschools.net/My Drive/Machine Learning/Facial Recognition/AttendanceSystem/Face_Identification/test2/Ashwin Pulla/IMG_3382.jpeg'







'''
opencv
{'Pranav Vadde', 'Emi Zhang', 'Mihika Dusad', 'Manav Gagvani', 'Ashwin Pulla', 'Tejesh Dandu', 'Shreyan Dey', 'Daniel Qiu', 'Darren Kao', 'Sritan Motati', 'Matthew Palamarchuk', 'Lucas Marschoun', 'Vishal Nandakumar', 'Samarth Bhargav', 'Ayaan Siddiqui', 'Akshat Alok'}

retinaface

'''


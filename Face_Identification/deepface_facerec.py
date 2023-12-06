from deepface import DeepFace

from stream_face_identification import *

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

def main():
    # See https://github.com/serengil/deepface/blob/master/deepface/commons/realtime.py
    # for implementation of stream()

    rtmp = "rtmp://127.0.0.1/live/SyWPfOiBa"
    
    analysis(
        db_path = '/Users/pranavv/Library/CloudStorage/GoogleDrive-1823210@fcpsschools.net/My Drive/Machine Learning/Facial Recognition/AttendanceSystem/Face_Identification/test2',
        model_name = MODELS[6], # ArcFace
        detector_backend = 'opencv',
        enable_face_analysis = False,
        time_threshold = 0.1,
        frame_threshold = 0.1,
        source=rtmp,
        distance_metric = 'cosine',
        face_size_threshold = 100,
        enable_viz=True,
    )

if __name__ == "__main__":
    main()

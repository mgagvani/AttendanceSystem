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

    analysis(
        db_path = 'test2',
        model_name = MODELS[6], # ArcFace
        detector_backend = 'opencv',
        enable_face_analysis = False,
        time_threshold = 3,
        frame_threshold = 3,
        source=0,
        distance_metric = 'cosine',
        face_size_threshold = 100,
        enable_viz=False,
    )

if __name__ == "__main__":
    main()

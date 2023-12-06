from deepface import DeepFace

from stream_face_identification import *
from headless_id import *

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

    # rtmp = "rtmp://127.0.0.1/live/SyWPfOiBa"
    rtmp = "rtmp://172.20.10.3/live/SyWPfOiBa"
    
    '''
    analysis(
        # db_path = '/Users/pranavv/Library/CloudStorage/GoogleDrive-1823210@fcpsschools.net/My Drive/Machine Learning/Facial Recognition/AttendanceSystem/Face_Identification/test2',
        db_path="test2",
        model_name = MODELS[6], # ArcFace
        detector_backend = 'opencv',
        enable_face_analysis = False,
        time_threshold = 1,
        frame_threshold = 1,
        source=rtmp,
        distance_metric = 'cosine',
        face_size_threshold = 100,
        enable_viz=True,
    )
    '''
    face = streamed_id_no_preview(
            db_path="test2",
            model_name = MODELS[6], # ArcFace
            detector_backend = 'opencv',
            source=rtmp,
            distance_metric = 'cosine',
            face_size_threshold = 100,
        )
    
    for i in range(10):
        t0 = time.perf_counter()
        recognitions = next(face)
        t1 = time.perf_counter()
        print(f"Time taken: {t1-t0:.2f}s", recognitions)

if __name__ == "__main__":
    main()

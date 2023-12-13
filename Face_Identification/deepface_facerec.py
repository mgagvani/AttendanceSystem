from deepface import DeepFace

from stream_face_identification import *
from headless_id import *

import json

MODELS = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", # not recognizing multiple faces
  "Dlib", 
  "SFace",
]

def main():
    # See https://github.com/serengil/deepface/blob/master/deepface/commons/realtime.py
    # for implementation of stream()

    rtmp = "rtmp://127.0.0.1/live/S1v7MEvL6"
    #rtmp = "rtmp://172.20.10.3/live/SyWPfOiBa"
    
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
            #db_path="test2",
            db_path='/Users/pranavv/Library/CloudStorage/GoogleDrive-1823210@fcpsschools.net/My Drive/Machine Learning/Facial Recognition/AttendanceSystem/Face_Identification/test2',
            #db_path = '/Users/shreyandey/Documents/Grade12/ML/AttendanceSystem/Face_Identification/test2',
            model_name = MODELS[2],
            detector_backend = 'retinaface',
            # source='/Users/shreyandey/Documents/Grade12/ML/AttendanceSystem/Face_Identificationd/face_test.MOV',
            source = rtmp, # web cam
            distance_metric = 'euclidean_l2',
            face_size_threshold = 50,
        )
    
    # faces = set()
    names = {'Akshat Alok', 'Samarth Bhargav', 'Preston Brown', 'Jesse Choe', 'Santiago Criado', 'Tejesh Dandu', 'Shreyan Dey', 'Mihika Dusad', 'Manav Gagvani', 'Om Gole', 'Rohan Kalahasty', 'Darren Kao', 'Dev Kodre', 'Pranav Kuppa', 'Grace Liu', 'Krish Malik', 'Lucas Marschoun', 'Lakshmi Sritan Motati', 'Vishal Nandakumar', 'Matthew Palamarchuk', 'Pranav Panicker', 'Tanvi Pedireddi', 'Ashwin Pulla', 'Daniel Qiu', 'Abhisheik Sharma', 'Ayaan Siddiqui', 'Raghav Sriram', 'Pranav Vadde', 'Akash Wudali'}
    facesDct = {name:False for name in names}

    output_path = '/Users/pranavv/Library/CloudStorage/GoogleDrive-1823210@fcpsschools.net/My Drive/Machine Learning/Facial Recognition/AttendanceSystem/Face_Identification/identified_people.json'
    #output_path = '/Users/shreyandey/Documents/Grade12/ML/AttendanceSystem/Face_Identification/identified_people.json'
    
    # try:
    #     with open(output_path, 'x') as f: pass
    # except FileExistsError: pass
    json_object_prelim = json.dumps(facesDct, indent=4)
    with open(output_path, 'w') as f:
        f.write(json_object_prelim)    # resets all students to 'not detected'

    while True:
        #t0 = time.perf_counter()
        try: recognitions = next(face)
        except StopIteration: break
        #t1 = time.perf_counter()
        #print(f"Time taken: {t1-t0:.2f}s", recognitions)
        for path in recognitions:
            # person = path[160:path.index('/', 160)]
            nameStartIdx = path.index('test2/')+len('test2/')
            person = path[nameStartIdx:path.index('/', nameStartIdx)]
            # print(path)
            if not facesDct[(fN:=formatName(person))]:
                facesDct[fN] = True
                json_object = json.dumps(facesDct, indent=4)
                with open(output_path, 'w') as f:
                    f.write(json_object)
                print(f"{fN} dectected! Reload site to see updates")
                
def formatName(withUnderscore):
    return f"{withUnderscore[0].upper()}{withUnderscore[1:withUnderscore.index('_')]} {withUnderscore[withUnderscore.index('_')+1].upper()}{withUnderscore[withUnderscore.index('_')+2:]}"

if __name__ == "__main__":
    main()

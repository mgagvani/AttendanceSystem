import os
import time
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from deepface.commons import functions

# dependency configuration
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# pylint: disable=too-many-nested-blocks

def streamed_id_no_preview(
    db_path,
    model_name="ArcFace",
    detector_backend="opencv",
    distance_metric="cosine",
    source=0,
    face_size_threshold=100,
):    
    pivot_img_size = 224  # face recognition result image

    # find custom values for this input set
    target_size = functions.find_target_size(model_name=model_name)
    
    # build models once to store them in the memory
    # otherwise, they will be built after cam started and this will cause delays
    DeepFace.build_model(model_name=model_name)
    print(f"facial recognition model {model_name} is just built")

    # call a dummy find function for db_path once to create embeddings in the initialization
    DeepFace.find(
        img_path=np.zeros([pivot_img_size, pivot_img_size, 3]),
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=False,
    )

    cap = cv2.VideoCapture(source)  # webcam

    frame_skip = 10  # every 10 frames will be processed

    while True:
        _, img = cap.read()

        if frame_skip % 10 != 0:
            frame_skip += 1
            continue

        # T0 - Camera read
        T0 = time.perf_counter()

        if img is None:
            break

        raw_img = img.copy()
        resolution_x = img.shape[1]
        resolution_y = img.shape[0]


        # detect faces
        face_included_frames = 0
        try:
            # just extract the regions to highlight in webcam
            face_objs = DeepFace.extract_faces(
                img_path=img,
                target_size=target_size,
                detector_backend=detector_backend,
                enforce_detection=False,
            )
            faces = []
            for face_obj in face_objs:
                facial_area = face_obj["facial_area"]
                faces.append(
                    (
                        facial_area["x"],
                        facial_area["y"],
                        facial_area["w"],
                        facial_area["h"],
                    )
                )
        except:  # to avoid exception if no face detected
            faces = []

            if len(faces) == 0:
                face_included_frames = 0

        # T1 - Face detection
        T1 = time.perf_counter()
        print(f"Face detection took {T1 - T0} seconds. Found {len(faces)} faces")
        print("faces:", faces)

        # for each face, if bigger than threshold, predict
        labels = []
        for i, face in enumerate(faces):
            x, y, w, h = face

            if w > face_size_threshold:
                face_included_frames += 1

                # predict
                custom_face = raw_img[y : y + h, x : x + w]  # slice face

                dfs = DeepFace.find(
                    img_path=custom_face,
                    db_path=db_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    distance_metric=distance_metric,
                    enforce_detection=False,
                    silent=True,
                )

                T2 = time.perf_counter()
                # print(f"Face recognition {i} took {T2 - T1} seconds")
                T1 = T2
                
                if len(dfs) > 0:
                    # directly access 1st item because custom face is extracted already
                    df = dfs[0]

                    if df.shape[0] > 0:
                        candidate = df.iloc[0]
                        label = candidate["identity"]
                        labels.append(label)

            yield labels    

                
    cap.release()

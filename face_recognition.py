import cv2
import os
import glob
import numpy as np
import cv2
import tensorflow as tf
import time
from mtcnn_master.mtcnn.mtcnn import MTCNN
from facenet_face_recognition_master.fr_utils import *
from facenet_face_recognition_master.inception_blocks_v2 import *
from keras import backend as K
import joblib
import argparse
import pymongo
from datetime import datetime
import json
import pytz


def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)

#create triploss function
def triplet_loss(y_true, y_pred, alpha = 0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,
               positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, 
               negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
   
    return loss


#define database
def prepare_database():
    database = {}


    for folder in glob.glob("facenet_face_recognition_master/images/*"):
        for file in glob.glob(folder+"/*"):
           identity = os.path.splitext(os.path.basename(file))[0]
           database[identity] = img_path_to_encoding(file, FRmodel)

    return database

#define recog function
def who_is_it(image, database, model):
    encoding = img_to_encoding(image, model)
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' %(name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    if min_dist > 0.52:
        return None
    else:
        identity=identity.split("_")[0]
        return identity

#config
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
color = (255, 255, 255)
thickness = 2

#prepare face_detection
detector = MTCNN()
K.set_image_data_format('channels_first')
FRmodel = faceRecoModel(input_shape=(3, 96, 96))

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)

# Load model
model_name = "trained_models/replay_attack_trained_models/replay-attack_ycrcb_luv_extraTreesClassifier.pkl"
thresh = 0.725
clf = None
sample_number = 1
count = 0
measures = np.zeros(sample_number, dtype=np.float)

try:
    clf = joblib.load(model_name)
except IOError as e:
    print ("Error loading model")
    exit(0)

#connect database
myclient = pymongo.MongoClient("mongodb+srv://VuGiaBao:bao0902429190@cluster0-c4dmj.azure.mongodb.net/face_recognition?retryWrites=true&w=majority")
mydb = myclient["Attendance_checking"]
CSDL_col = mydb["CSDL"]
Cham_cong_col = mydb["Cham_cong"]

#call database func
data=prepare_database()
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, image = cap.read()
    result = detector.detect_faces(image)
    measures[count%sample_number]=0
    point = (0,0)

    for person in result:
        bounding_box = person['box']
        keypoints = person['keypoints']
        cv2.rectangle(image,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                      (0,155,255),
                      2)

        crop_img = image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]

        (x, y, h, w) = bounding_box
        roi = image[y:y+h, x:x+w]

        img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
        img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

        ycrcb_hist = calc_hist(img_ycrcb)
        luv_hist = calc_hist(img_luv)

        feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
        feature_vector = feature_vector.reshape(1, len(feature_vector))

        prediction = clf.predict_proba(feature_vector)
        prob = prediction[0][1]

        measures[count % sample_number] = prob
        point = (x, y-5)


        print (measures, np.mean(measures))
        if 0 not in measures:
            text = "FAKE"
            if np.mean(measures) >= np.float(thresh):
                text = "REAL"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=image, text=text, org=point, fontFace=font, fontScale=0.9, color=(0, 255, 0),
                            thickness=2, lineType=cv2.LINE_AA)
                real = True
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img=image, text=text, org=point, fontFace=font, fontScale=0.9,
                            color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
                real = False
         #call recog func
        if (real):
            id=who_is_it(crop_img, data, FRmodel)
            image=cv2.putText(image,id,(bounding_box[0],bounding_box[1]+bounding_box[3]),font,  
                           fontScale, color, thickness, cv2.LINE_AA)
            print (id)

            #push database
            if id == None:
                pass
            else:
                ID_found={"ID":id}
                res=CSDL_col.find_one(ID_found,{"_id":0})
                res['realtime']= datetime.now(pytz.timezone("Asia/Bangkok"))
                Cham_cong_col.insert_one(res)


    count+=1
    cv2.imshow("image",image)




    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

    

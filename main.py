import sys
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
from dialog import Ui_Dialog
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
from tensorflow.keras import backend as K
from sklearn.externals import joblib 
import argparse
import pymongo
from datetime import datetime
import json
import pytz

class AppWindow(QDialog):
	def __init__(self):
		super().__init__()
		self.ui = Ui_Dialog()
		self.ui.setupUi(self)

		print("prepare face_detection")
		self.detector = MTCNN()
		K.set_image_data_format('channels_first')
		self.FRmodel = faceRecoModel(input_shape=(3, 96, 96))

		self.FRmodel.compile(optimizer = 'adam', loss = self.triplet_loss, metrics = ['accuracy'])
		load_weights_from_FaceNet(self.FRmodel)

		print("Load model security")
		self.model_name = "trained_models/replay_attack_trained_models/replay-attack_ycrcb_luv_extraTreesClassifier.pkl"
		self.thresh = 0.725
		self.clf = None
		self.sample_number = 1
		self.count = 0
		self.measures = np.zeros(self.sample_number, dtype=np.float)

		try:
			self.clf = joblib.load(self.model_name)
		except IOError as e:
			print ("Error loading model")
			exit(0)

		print("onnect database-server")
		self.myclient = pymongo.MongoClient("mongodb+srv://VuGiaBao:bao0902429190@cluster0-c4dmj.azure.mongodb.net/face_recognition?retryWrites=true&w=majority")
		self.mydb = self.myclient["Attendance_checking"]
		self.CSDL_col = self.mydb["CSDL"]
		self.Cham_cong_col = self.mydb["Cham_cong"]

		print("call database func")
		self.data=self.prepare_database()

		print("create a timer")
		self.timer = QTimer()
		print("set timer timeout callback function")
		self.timer.timeout.connect(self.recog_pushdata)
		print("Get control_bt callback clicked  function")
		self.ui.Open_bt.clicked.connect(self.controlTimer)

	def calc_hist(self,img):
		histogram = [0] * 3
		for j in range(3):
			histr = cv2.calcHist([img], [j], None, [256], [0, 256])
			histr *= 255.0 / histr.max()
			histogram[j] = histr
		return np.array(histogram)

	#create triploss function
	def triplet_loss(self,y_true, y_pred, alpha = 0.3):
		anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
		pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)), axis=-1)
		neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), axis=-1)
		basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
		self.loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
		return self.loss

	#define database
	def prepare_database(self):
		self.database = {}
		for folder in glob.glob("facenet_face_recognition_master/images/*"):
			for file in glob.glob(folder+"/*"):
				self.identity = os.path.splitext(os.path.basename(file))[0]
				self.database[self.identity] = img_path_to_encoding(file, self.FRmodel)
		return self.database

	#define recog function
	def who_is_it(self,image, database, model):
		self.encoding = img_to_encoding(image, model)
		self.min_dist = 100
		self.identity = None

		# Loop over the database dictionary's names and encodings.
		for (name, db_enc) in database.items():
			dist = np.linalg.norm(db_enc - self.encoding)
			if dist < self.min_dist:
				self.min_dist = dist
				self.identity = name

		if self.min_dist > 0.52:
			return None
		else:
			self.identity=self.identity.split("_")[0]
			return self.identity

	def recog_pushdata(self):
		# Capture frame-by-frame
		ret, image = self.cap.read()
		result = self.detector.detect_faces(image)
		self.measures[self.count%self.sample_number]=0
		point = (0,0)


		print("Running...")
		for person in result:
			bounding_box = person['box']
			cv2.rectangle(image,(bounding_box[0], bounding_box[1]),(bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),(0,155,255),2)
			crop_img = image[bounding_box[1]:bounding_box[1]+bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]]

			(x, y, h, w) = bounding_box
			roi = image[y:y+h, x:x+w]

			img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
			img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

			ycrcb_hist = self.calc_hist(img_ycrcb)
			luv_hist = self.calc_hist(img_luv)

			feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
			feature_vector = feature_vector.reshape(1, len(feature_vector))

			prediction = self.clf.predict_proba(feature_vector)
			prob = prediction[0][1]

			self.measures[self.count % self.sample_number] = prob
			point = (x, y-5)

			if 0 not in self.measures:
				text = "FAKE"
				if np.mean(self.measures) >= np.float(self.thresh):
					text = "REAL"
					cv2.putText(img=image, text=text, org=point, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9, color=(0, 255, 0),thickness=2, lineType=cv2.LINE_AA)
					real = True
				else:
					cv2.putText(img=image, text=text, org=point, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.9,color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
					real = False

			#call recog func
			if (real):
				id=self.who_is_it(crop_img, self.data, self.FRmodel)
				image=cv2.putText(image,id,(bounding_box[0],bounding_box[1]+bounding_box[3]),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2, cv2.LINE_AA)
				print (id)

				#push database
				if id == None:
					pass
				else:
					self.ID_found={"ID":id}
					self.res=self.CSDL_col.find_one(self.ID_found,{"_id":0})
					self.res['realtime']= datetime.now(pytz.timezone("Asia/Bangkok"))
					self.Cham_cong_col.insert_one(self.res)


		# get frame infos
		image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		height, width, channel = image.shape
		step = channel * width
		print("create QImage from RGB frame")
		qImg = QImage(image, width, height, step, QImage.Format_RGB888)
		print("show frame in img_label")
		self.ui.label.setPixmap(QPixmap.fromImage(qImg))

	# start/stop timer
	def controlTimer(self):
		# if timer is stopped
		if not self.timer.isActive():
			# create video capture
			self.cap = cv2.VideoCapture(0)
			# start timer
			self.timer.start(20)
			# update control_bt text
			self.ui.Open_bt.setText("Close")

		# if timer is started
		else:
			# stop timer
			self.timer.stop()
			# release video capture
			self.cap.release()
			# update control_bt text
			self.ui.Open_bt.setText("Open")


app = QApplication(sys.argv)
w = AppWindow()
w.show()
sys.exit(app.exec_())


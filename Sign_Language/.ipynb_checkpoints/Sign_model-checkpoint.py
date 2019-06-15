import numpy as np
import cv2
from tensorflow.keras.models import model_from_json

class Sign_Language_model(object):
	"""This model Recieves Input From the webacam and gives to presaved model for predictions."""
	WORD_LIST = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'.split()

	def __init__(self, model_json_file, model_weight_file):
		with open(model_json_file, 'r') as json_file:
			loaded_model_json = json_file.read()
			self.loaded_model = model_from_json(loaded_model_json)

		self.loaded_model.load_weights(model_weight_file)
		print("Convolution model for Sign Language Detection is loaded \n Summary :- \n")
		self.loaded_model.summary()

	def predict_signs(self, image):
		self.preds = self.loaded_model.predict(np.array(cv2.resize(image,(28,28))).reshape(-1,28,28,1))
		return Sign_Language_model.WORD_LIST[np.argmax(self.preds)]

def start(cnn):
	cap = cv2.VideoCapture(0)
	font = cv2.FONT_HERSHEY_SIMPLEX

	while(True):
		ret, frame = cap.read()

		if ret:
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			gray = cv2.GaussianBlur(gray, (5, 5), 0)
			x = cv2.rectangle(gray, (5,5), (250,250), (255,0,0), 2)
			clone = gray.copy()
			pred = cnn.predict_signs(clone[5:250,5:250])
			cv2.putText(gray, 'Press Q to quit', (300,300), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
			cv2.putText(gray, pred, (180,180), font, 2, (255, 255, 0), 2)
			cv2.imshow('frame',gray)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	
	cap.release()
	cv2.destroyAllWindows()


model = Sign_Language_model('Data/Sign_model.json','Data/Sign_model.h5')
start(model)
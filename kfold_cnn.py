import os
import csv
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

from keras import optimizers, applications
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Activation

"""
Images and folder structure must be organized and named as follows:
	+---ROOT (place this script script here)
		+---checkpoints (empty folder)
		+---data
			+---k_folds
				+---split_1
					+---training
						+---AQ (acceptable quality images)
						+---NAQ (not acceptable quality images)
					+---validation
						+---AQ
						+---NAQ
				+---split_2
					+---training
						+---AQ
						+---NAQ
					+---validation
						+---AQ
						+---NAQ
				+---etc
"""


## mute TensorFlow build warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


def trainCNN(num_train, num_val, k=5, epochs=25, batch_size=20, img_width=150, img_height=150, img_channel=3):
	"""
	Performs k-fold cross-validation, using InceptionV3, on a previously separated dataset.

	@params:
		num_train		- Required 	: number of training images (images in K-1 splits)
		num_val			- Required 	: number of validation images (images in one split)
		k				- Optional  : number of splits (i.e. the 'K' in k-fold cross-validation)
		epochs			- Optional 	: total number of epochs (note: advise to use early stopping)
		batch_size		- Optional 	: number of images to feed CPU/GPU per batch
		img_width		- Optional 	: image width
		img_height		- Optional 	: image height
		img_channel		- Optional 	: number of channels in each image (e.g. grayscale = 1, RGB = 3, RGB + Alpha = 4)

	Returns:
		Numerous models for each of the K models into a 'Checkpoints' folder.
		Manual selection of best model (lowest loss/accuracy) will be necessary.
	"""

	for k_fold in range(0, k):
		train_split = 'data/k_folds/split_' + str(k_fold) + '/training'
		val_split = 'data/k_folds/split_' + str(k_fold) + '/validation'
		print('Training Set: ' + train_split)
		print('Validation Set: ' + val_split)

		base_model = applications.InceptionV3(
			weights = 'imagenet',
			include_top = False,
			input_shape = (img_width, img_height, img_channel)
		)

		top_model = Sequential()
		top_model.add(Flatten(input_shape = base_model.output_shape[1:]))
		top_model.add(Dense(4096, activation = 'relu'))
		top_model.add(Dropout(0.5))
		top_model.add(Dense(1, activation = 'sigmoid'))

		model = Model(
			inputs = base_model.input,
			outputs = top_model(base_model.output)
		)

		model.summary()

		model.compile(
			loss='binary_crossentropy',
			optimizer = optimizers.SGD(lr = 0.001, momentum = 0.9),
			metrics = ['accuracy']
		)

		train_datagen = ImageDataGenerator(
			rescale = 1. / 255,
			zoom_range = 0.2,
			horizontal_flip = True,
			vertical_flip = True
		)

		val_datagen = ImageDataGenerator(
			rescale = 1. / 255
		)

		train_generator = train_datagen.flow_from_directory(
			directory = train_split,
			target_size = (img_width, img_height),
			batch_size = batch_size,
			class_mode = 'binary'
		)

		validation_generator = val_datagen.flow_from_directory(
			directory = val_split,
			target_size = (img_width, img_height),
			batch_size = batch_size,
			class_mode = 'binary'
		)

		checkpoint = [ModelCheckpoint(
			filepath = 'checkpoints/IQA_split_' + str(k_fold) + '_{val_acc:.4f}_{val_loss:4f}',
			monitor = 'val_acc',
			save_best_only = True
		)]

		model.fit_generator(
			generator = train_generator,
			steps_per_epoch = num_train // batch_size,
			epochs = epochs,
			validation_data = validation_generator,
			validation_steps = num_val // batch_size,
			callbacks = checkpoint,
			shuffle = True
		)



def KFoldROC(k=5, batch_size=20, img_width=150, img_height=150, img_channels=150):
	"""
	Calculates and plots ROC curves for each of the cross-validated models.

	@params:
		k				- Optional  : number of splits (i.e. the 'K' in k-fold cross-validation)
		batch_size		- Optional 	: number of images to feed CPU/GPU per batch
		img_width		- Optional 	: image width
		img_height		- Optional 	: image height
		img_channel		- Optional 	: number of channels in each image (e.g. grayscale = 1, RGB = 3, RGB + Alpha = 4)

	Returns:
		A single image file with all k ROC curves plotted.
		A .csv file of all images, their true labels, and the predicted labels.
	"""

	for idx, model in enumerate(os.listdir('checkpoints')):
		model_path = 'checkpoints/' + str(model)
		split = idx + 1

		test_dir = 'data/k_folds/split_' + str(split) + '/val'
		test_datagen = ImageDataGenerator(rescale = 1. / 255)
		test_generator = test_datagen.flow_from_directory(
			test_dir,
			shuffle = False,
			target_size = (img_width, img_height),
			batch_size = batch_size,
			class_mode = 'binary')

		print('Loading model: ' + str(model))
		model = load_model(model_path)

		print('\n\nModel Summary:')
		model.summary()


		print('Predicting...')
		results = model.predict_generator(test_generator)
		img_id = [img for img in test_generator.filenames]
		img_label = [label for label in test_generator.classes]

		print('Writing predictions to file...')
		with open('data/analysis/split_' + str(split) + '.txt', 'w') as fh:
			for idx, result in enumerate(results):
				fh.write(img_id[idx] + '\t' + str(img_label[idx]) + '\t' + str(result[0]) + '\n')


		plt.figure(num = 0, dpi = 300, frameon = True).clf()


		for split in range(1, k + 1):
			label = []
			pred = []

			with open('data/analysis/split_' + str(split) + '.txt') as fh:
				data = csv.reader(fh, delimiter = '\t')
				for line in data:
					label.append(int(line[1]))
					pred.append(float(line[2]))

			fpr, tpr, thresh = metrics.roc_curve(label, pred)
			auc = metrics.roc_auc_score(label, pred)
			plt.plot(fpr, tpr, label='Model ' + str(split) + ', AUC=' + "%.3f" % round(auc, 3))

		plt.plot([0, 1], [0, 1], 'r--')
		plt.xlabel('False-Positive Rate')
		plt.ylabel('True-Positive Rate')
		plt.legend(loc=(0.5,0.1))
		plt.grid(b=True, which='major', color='gray', linestyle='-')
		plt.savefig('kfold_AUC.png')

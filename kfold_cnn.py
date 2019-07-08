import os
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, applications
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Activation


## mute TensorFlow build warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"


def KFoldCV(num_train, num_val, k=5, epochs=25, batch_size=20, img_width=150, img_height=150, img_channel=3):
	"""
	Performs k-fold cross-validation, using InceptionV3, on a previously separated dataset.
	Images and folder structure must be organized and named as follows:
		+checkpoints
		+---data
			+---k_folds
				+---split_1
					+---training
					+---validation
				+---split_2
					+---training
					+---validation
				+---etc

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
			directory = train_data_dir,
			target_size = (img_width, img_height),
			batch_size = batch_size,
			class_mode = 'binary'
		)

		validation_generator = val_datagen.flow_from_directory(
			directory = validation_data_dir,
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

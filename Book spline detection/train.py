from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from PIL import ImageFile
from sklearn.externals import joblib

ImageFile.LOAD_TRUNCATED_IMAGES = True

classifier = Sequential()
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation="relu"))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, 3, 3, activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=128, activation="relu"))
classifier.add(Dense(output_dim=128, activation="relu"))
classifier.add(Dense(output_dim=62, activation="sigmoid"))

classifier.compile(optimizer="adam",
                   loss="categorical_crossentropy",
                   metrics=["accuracy"])

from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory("Try/train",
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode="categorical")

test_set = test_datagen.flow_from_directory("Try/test",
                                                 target_size=(64,64),
                                                 batch_size=32,
                                                 class_mode="categorical")

classifier.fit_generator(training_set,
                         steps_per_epoch=500,
                         epochs=20,
                         validation_data=test_set,
                         validation_steps=100)

joblib.dump(classifier, "minor.pkl")
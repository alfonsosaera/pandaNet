# ##############################################################################
#  P a c k a g e s
# ##############################################################################
import os
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use("Agg") # set the matplotlib backend so figures can be saved in the background
import matplotlib.pyplot as plt
import numpy as np

# custom
from datasets import DataLoader
from preprocessing import Resizing
from NN import NeuralNetwork

# fix to make cuDNN work, from https://github.com/tensorflow/tensorflow/issues/24828#issuecomment-464910864
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# ##############################################################################
#  V a r i a b l e s
# ##############################################################################
width = 64
height = width
depth = 3
verb_prep = 1000
myPath = 'datasets/animals'
labelNames = ['cat', 'dog', 'panda']
opt='SGD'
test_size, random_state = 0.2, 42
batch_size, epochs, verbose = 64, 40, 1
lr, momentum, nesterov = 0.01 , 0.9, True
decay = lr / epochs
ESP = 5 # EARLY_STOPPING_PATIENCE

# ##############################################################################
#  O u t p u t   c o n f i g
# ##############################################################################
output='output/{}_w.{}_h.{}_ES.{}/'.format( opt, width, height, ESP )
os.makedirs(output, exist_ok=True)
visualizationFile = output + 'model.visualization.png'
plotFile = output + 'plot.png'
modelFile = output + 'model.serialization.hdf5'

# ##############################################################################
#  P r o c e s s i n g
# ##############################################################################
# tf.random.set_seed(random_state)
# get images
myImages = list(paths.list_images(myPath))

# preprocessor
rs = Resizing(width=width,height=height)

# dataloader
dl = DataLoader(preprocessors=[rs])

# get data and labels
(X, y) = dl.load(myImages, verbose=verb_prep)

# split and scale images to 0-1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                    random_state=random_state)
X_train = X_train.astype("float") / 255.0
X_test = X_test.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
# opt = Adam(lr=lr)
model = NeuralNetwork.build(width=width, height=height, depth=depth, classes=len(labelNames))

# save model visualization
plot_model(model, to_file=visualizationFile, show_shapes=True)

# compile model
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# initialize an early stopping callback
es = EarlyStopping(
	monitor="val_loss",
	patience=ESP,
	restore_best_weights=True)

# train the network
print("[INFO] training network...")
H = model.fit(X_train, y_train, validation_data=(X_test, y_test),
	batch_size=batch_size, epochs=epochs, callbacks=[es], verbose=verbose)

# save the network to disk
print("[INFO] serializing network...")
model.save( modelFile )

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(X_test, batch_size=batch_size)
print(classification_report(y_test.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, len(H.history['loss']) ), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, len(H.history["val_loss"]) ), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, len(H.history["accuracy"]) ), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, len(H.history["val_accuracy"]) ), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(plotFile)

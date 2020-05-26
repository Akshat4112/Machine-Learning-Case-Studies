from sklearn.preprocessing import LabelBinarizer
from ShallowNet.nn.conv.minivggnet import MiniVGGNet
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to weights directory")
args = vars(ap.parse_args())

print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])

fname = os.path.sep.join([args["weights"], "weights={epoch:03d}-{val_loss:.4f}.hdf5"])

checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

print("[INFO] training network . . .")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=2)

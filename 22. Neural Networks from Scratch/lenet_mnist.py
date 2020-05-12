from ShallowNet.nn.conv.lenet import LeNet
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

print("[INFO] accessing MNIST...")
dataset = fetch_openml("mnist_784")
data = dataset.data

if K.image_data_format() == "channels_first":
    data = data.reshape(data.shape[0], 1, 28, 28)

# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
    data = data.reshape(data.shape[0], 28, 28, 1)
    
(trainX, testX, trainY, testY) = train_test_split(data / 255.0,
dataset.target.astype("int"), test_size=0.25, random_state=42)

# convert the labels from integers to vectors
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)
    
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
batch_size=128, epochs=20, verbose=1)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
predictions.argmax(axis=1),
target_names=[str(x) for x in le.classes_]))


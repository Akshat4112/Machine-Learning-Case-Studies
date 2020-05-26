from keras.utils import plot_model
from ShallowNet.nn.conv.lenet import LeNet

model = LeNet.build(28,28,1,10)
plot_model(model, to_file='lenet.png', show_shapes=True)
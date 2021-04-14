import numpy as np
import keras as ke
import matplotlib.pyplot as plt
x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
for i in range(2):
    print("Image Class - "+str(np.argmax(y_train[i])))
    print(x_train[i])
    plt.imshow(x_train[i])
    plt.xlabel(np.argmax(y_train[i]))
    plt.show()
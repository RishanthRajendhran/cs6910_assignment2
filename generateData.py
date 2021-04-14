import keras as ke
import numpy as np
from keras.preprocessing.image import ImageDataGenerator 
from sys import argc, argv

if argc > 1 and argv[1] == "-dataAug":
    datasetAugmentation = True 
else:
    datasetAugmentation = False

batchSize = 1
targetSize = (256,256)

if datasetAugmentation:
    train_data = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.1,
        zoom_range=[0.5,1.0],
        # brightness_range=[0.2,1.0],
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        # height_shift_range=0.5,
        # width_shift_range=[-200,200]
    )
else:
    train_data = ImageDataGenerator(
        rescale=1.0/255.0,
        validation_split=0.1,
    )
test_data = ImageDataGenerator(
    rescale=1.0/255.0,
)

train_generator = train_data.flow_from_directory(
    "./inaturalist_12K/train",
    target_size=targetSize,
    batch_size=batchSize,
    subset="training",
    class_mode="categorical",
    shuffle=True
)

validation_generator = train_data.flow_from_directory(
    "./inaturalist_12K/train",
    target_size=targetSize,
    batch_size=batchSize,
    subset="validation",
    class_mode="categorical",
    shuffle=True
)

test_generator = test_data.flow_from_directory(
    "./inaturalist_12K/val",
    target_size=targetSize,
    batch_size=batchSize,
    class_mode="categorical",
    shuffle=True
)

x_train = []
y_train = []

i = 0

while i <= train_generator.batch_index:
    x, y = train_generator.next()
    # x_train.extend(x)
    # y_train.extend(y)
    x_train.append(x[0])
    y_train.append(y[0])
    # y_train.append(int(np.argmax(y[0])))
    i += 1

x_valid = []
y_valid = []

i = 0

while i <= validation_generator.batch_index:
    x, y = validation_generator.next()
    # x_valid.extend(x)
    # y_valid.extend(y)
    x_valid.append(x[0])
    y_valid.append(y[0])
    # y_valid.append(int(np.argmax(y[0])))
    i += 1

x_test = []
y_test = []

i = 0

while i <= train_generator.batch_index:
    x, y = train_generator.next()
    # x_test.extend(x)
    # y_test.extend(y)
    x_train.append(x[0])
    y_train.append(y[0])
    # y_train.append(int(np.argmax(y[0])))
    i += 1

x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)
x_test = np.array(x_test)
y_test = np.array(y_test)

if datasetAugmentation:
    np.save("x_train_aug.npy",x_train)
    np.save("y_train_aug.npy",y_train)
    np.save("x_valid_aug.npy",x_valid)
    np.save("y_valid_aug.npy",y_valid)
else:
    np.save("x_train.npy",x_train)
    np.save("y_train.npy",y_train)
    np.save("x_valid.npy",x_valid)
    np.save("y_valid.npy",y_valid)
np.save("x_test.npy",x_valid)
np.save("y_test.npy",y_valid)
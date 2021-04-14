import keras as ke
import numpy as np
import tensorflow as tf
from ConvNeuralNetwork import ConvNeuralNetwork
from sys import argv

evaluate = False
guidedBackProp = False
visualiseFilters = False
predictedClasses = False

if len(argv) > 1:
    if argv[1] == "-eval":
        evaluate = True 
    elif argv[1] == "-guidedBackProp":
        guidedBackProp = True
    elif argv[1] == "-visualiseFilters":
        visualiseFilters = True
    elif argv[1] == "-predictedClasses":
        predictedClasses = True

import wandb
hyperparameters_default = dict(
    numFilters = 32,
    numFiltersScheme = 1.0,
    datasetAugmentation = True,
    batchNormalization = False,
    dropout = 0.25,
    numEpochs = 50,
    defaultActivationFunction = "selu",
    numNeuronsInDenseLayers = 64,
    filterSize = 5,
    numConvBlocks = 1,
    numDenseLayers = 2,
    inputDimensions = (256,256,3),
    numOutputClasses = 10
)

run = wandb.init(project="cs6910_assignment2", entity="rishanthrajendhran", config=hyperparameters_default)
config = wandb.config
wandb.run.name = f"{config.numFilters}_{config.numFiltersScheme}_{config.datasetAugmentation}_{config.batchNormalization}_{config.dropout}_{config.numEpochs}_{config.defaultActivationFunction}_{config.numNeuronsInDenseLayers}_{config.filterSize}_{config.numConvBlocks}_{config.numDenseLayers}"
wandb.run.save(wandb.run.name)

cnn = ConvNeuralNetwork(config.inputDimensions, config.numOutputClasses, config.numConvBlocks, config.numDenseLayers, config.numFilters, config.filterSize, config.numNeuronsInDenseLayers, config.numFiltersScheme, config.defaultActivationFunction)
cnn.createCNN(config.dropout, config.batchNormalization)

if guidedBackProp:
    x_valid = np.load("./x_valid.npy")
    y_valid = np.load("./y_valid.npy")
    genGrads = False
    randNeuNos = [23, 26, 19, 20, 0, 9, 21, 10, 6, 2] #Previously generated
    if len(argv) > 2 and argv[2] == "-generateGrads":
        genGrads = True
    cnn.guidedBackProp(x_valid, y_valid,genGrads,randNeuNos)
elif evaluate:          #Onus on user to ensure that the hyperparameters_default matches the best_model config
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")
    cnn.evaluateCNN(x_test,y_test)
elif visualiseFilters: 
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")
    sample = np.random.randint(len(x_test))
    cnn.visualiseFilters(x_test[sample:sample+1], y_test[sample:sample+1])
elif predictedClasses:
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")
    inputs, outputs = [], []
    for i in range(10):
        sample = np.random.randint(len(x_test))
        inputs.append(x_test[sample])
        outputs.append(y_test[sample])
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    cnn.predictedClasses(inputs, outputs)
else:
    if config.datasetAugmentation:
        x_train = np.load("x_train_aug.npy")
        y_train = np.load("y_train_aug.npy")
        x_valid = np.load("x_valid_aug.npy")
        y_valid = np.load("y_valid_aug.npy")
    else:
        x_train = np.load("./x_train.npy")
        y_train = np.load("./y_train.npy")
        x_valid = np.load("./x_valid.npy")
        y_valid = np.load("./y_valid.npy")
    cnn.trainCNN(x_train, y_train, x_valid, y_valid, config.numEpochs)

# #Log in wandb
# metrics = {
#     'loss': history.history["loss"], 
#     "categorical_accuracy": history.history["categorical_accuracy"],
#     "val_loss": history.history["val_loss"],
#     "val_categorical_accuracy": history.history["val_categorical_accuracy"]
# }
# wandb.log(metrics)
run.finish()
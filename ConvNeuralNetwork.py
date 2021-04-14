import tensorflow.keras as ke
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

@tf.RegisterGradient("GuidedRelu")
def _GuidedReluGrad(op, grad):
   gate_f = tf.cast(op.outputs[0] > 0., "float32") #for f^l > 0
   gate_R = tf.cast(grad > 0., "float32") #for R^l+1 > 0
   return gate_f * gate_R * grad

class ConvNeuralNetwork:
  def __init__(self, inputDimensions, numOutputClasses, numConvBlocks, numDenseLayers, numFilters, filterSize, numNeuronsInDenseLayers, numFiltersScheme = 1, outputFunction = "softmax", defaultActivationFunction = "relu", activationFunctions = []):
    self.inputDimensions = inputDimensions
    self.numOutputClasses = numOutputClasses
    self.numConvBlocks = numConvBlocks
    self.numDenseLayers = numDenseLayers
    self.numFilters = numFilters
    self.numFiltersScheme = numFiltersScheme
    self.filterSize = filterSize
    self.numNeuronsInDenseLayers = numNeuronsInDenseLayers
    self.defaultActivationFunction = defaultActivationFunction
    self.activationFunctions = []
    self.numFiltersLayerwise = [numFilters]
    self.outputFunction = outputFunction
    self.model = None

  def createCNN(self, dropout = 0.0, batchNormalization = False):
    while len(self.activationFunctions) < (self.numConvBlocks + self.numDenseLayers):
      self.activationFunctions.append(self.defaultActivationFunction)
    for i in range(1, self.numConvBlocks):
      size = max(1,int(self.numFiltersLayerwise[i-1]*self.numFiltersScheme))
      self.numFiltersLayerwise.append(size)
    self.model = ke.Sequential()
    # if self.numConvBlocks > 0:
    #   self.model.add(ke.layers.Conv2D(self.numFiltersLayerwise[0], self.filterSize, input_shape = self.inputDimensions, padding="same"))
    #   self.model.add(ke.layers.Activation(self.activationFunctions[0]))
    #   if batchNormalization:
    #     self.model.add(ke.layers.BatchNormalization())
    #   self.model.add(ke.layers.MaxPooling2D(pool_size=(2, 2)))
    for i in range(0,self.numConvBlocks):
    # for i in range(1,self.numConvBlocks):
      self.model.add(ke.layers.Conv2D(self.numFiltersLayerwise[i], self.filterSize, padding="same"))
      self.model.add(ke.layers.Activation(self.activationFunctions[i]))
      if batchNormalization:
        self.model.add(ke.layers.BatchNormalization())
      self.model.add(ke.layers.MaxPooling2D(pool_size=(2, 2)))
    self.model.add(ke.layers.Flatten())
    for i in range(self.numDenseLayers):
      self.model.add(ke.layers.Dropout(dropout))
      self.model.add(ke.layers.Dense(self.numNeuronsInDenseLayers, activation=self.activationFunctions[self.numConvBlocks + i]))
      if batchNormalization:
        self.model.add(ke.layers.BatchNormalization())
    self.model.add(ke.layers.Dense(self.numOutputClasses))
    self.model.add(ke.layers.Activation(self.outputFunction))

  def trainCNN(self, x_train, y_train, x_valid, y_valid, numEpochs):
    filepath = './best_model_acc'
    self.model = ke.models.load_model(filepath, compile = True)
    filepath_acc = './best_model_acc'
    filepath_loss = './best_model_loss'
    checkpoint_acc = ke.callbacks.ModelCheckpoint(filepath_acc, save_weights_only=False, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    checkpoint_loss = ke.callbacks.ModelCheckpoint(filepath_loss, save_weights_only=False, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint_acc, checkpoint_loss]
    self.model.compile(
        optimizer=ke.optimizers.SGD(learning_rate=0.01),
        loss=ke.losses.CategoricalCrossentropy(),
        metrics=[ke.metrics.CategoricalAccuracy()],
    )
    self.model.fit(
        x_train,
        y_train,
        epochs = numEpochs,
        validation_data = (x_valid, y_valid),
        # callbacks=[WandbCallback()]
        callbacks=callbacks_list
    )
    preds = (self.model.predict(x_valid))
    for i in range(len(preds)):
        print(f"Prediction - {preds[i]}, Actual - {y_valid[i]}")
    ke.models.save_model(self.model, filepath)

  def evaluateCNN(self, x_test, y_test):
      filepath = './best_model/best_model_acc'
      self.model = ke.models.load_model(filepath, compile = True)
      results = self.model.evaluate(x_test, y_test)
      print(results)

  def generateGrads(self, inputs, labels, layerNo=0, neuNo = 0):
    filepath = './best_model/best_model_acc'
    self.model = ke.models.load_model(filepath, compile = True)
    inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
    loss = 0
    with tf.compat.v1.get_default_graph().gradient_override_map({'Relu': 'GuidedRelu'}):
      with tf.GradientTape() as tape:
          tape.watch(inputs)
          outputs = self.get_ith_layer_output(inputs, layerNo)
          # newOut = []
          # for k in range(outputs.shape[3]):
          #   if k != neuNo:
          #     newOut.append(np.zeros((1,outputs.shape[1],outputs.shape[2])))
          #   else:
          #     newOut.append(outputs[:, :, :, k])
          # outputs = tf.convert_to_tensor(newOut,dtype=tf.float32)
    grads = tape.gradient(outputs, inputs)
    np.save(f"./guidedBackPropGrads_layer{layerNo}_neuron{neuNo}.npy",grads)

  def guidedBackProp(self, inputs, labels, genGrads = False, randNeuNos = []):
    layerNo = 1   #Best model has only 1 CONV layer
    if randNeuNos == [] or genGrads:
      randNeuNos = []
      for i in range(10):
        neuNo = np.random.randint(32) #WKT number of filters in the only CONV layer is 32
        while neuNo in randNeuNos:
          neuNo = np.random.randint(32)
        randNeuNos.append(neuNo)
      self.modifyBackProp()
      for neuNo in randNeuNos:
        self.generateGrads(inputs, labels, layerNo, neuNo)
    for i in range(5):
      fig, axs = plt.subplots(2, 2)
      for j in range(2):
        allGrads = np.load(f"./guidedBackPropGrads_layer{layerNo}_neuron{randNeuNos[i]}.npy",allow_pickle=True)
        allGrads -= np.min(allGrads)
        allGrads /= np.max(allGrads)
        maxGrads = [np.max(-allGrads[k,:,:]) for k in range(allGrads.shape[0])]
        maxOut = np.argmax(maxGrads)
        maxOut = np.random.randint(999)
        grads = allGrads[maxOut]
        guidedBackprop = np.dstack((
                grads[:, :, 0],
                grads[:, :, 1],
                grads[:, :, 2],
            ))

        guidedBackprop -= np.min(guidedBackprop)
        guidedBackprop /= np.max(guidedBackprop)

        axs[j,0].set_title(f"Layer {layerNo}, Neuron {randNeuNos[2*i+j]}")
        axs[j,0].imshow(inputs[maxOut])
        axs[j,0].axis("off")
        axs[j,1].imshow(guidedBackprop)
        axs[j,1].axis("off")
      wandb.log({f"guidedBackProp{i}":wandb.Image(plt)})
    

  def get_ith_layer_output(self, inputs, i=0):
    if i > len(self.model.layers):
        return None
    filepath = './best_model/best_model_acc'
    self.model = ke.models.load_model(filepath, compile = True)
    partial = ke.Model(self.model.inputs, self.model.layers[i].output)
    out = partial([inputs], training=False)
    return out

  def visualiseFilters(self, inputs, outputs):
    filepath = './best_model/best_model_acc'
    self.model = ke.models.load_model(filepath, compile = True)
    wandb.log({"inputImage":wandb.Image(inputs[0])})
    fig, axs = plt.subplots(8,4)
    counter, row = 0, 0
    outputs = self.get_ith_layer_output(inputs)
    for i in range(32):
        out = outputs[...,i][0]
        axs[row,counter].imshow(out)
        axs[row,counter].axis("off")
        counter += 1
        if counter%4 == 0:
            row += 1
            counter = 0
    plt.show()
    wandb.log({"filtersVisualisation":wandb.Image(plt)})

  def predictedClasses(self, inputs, outputs):
    filepath = './best_model/best_model_acc'
    self.model = ke.models.load_model(filepath, compile = True)

    fig = plt.figure(figsize=(10,8))
    outer = gridspec.GridSpec(10, 1, wspace=0.2, hspace=0.2)

    for i in range(10):
        inner = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        tc = np.argmax(outputs[i])
        pcp = self.model.predict(inputs)[i]
        pcp /= np.sum(pcp) 
        pc = np.argmax(pcp) 
        pcp = np.max(pcp)
        ax = plt.Subplot(fig, inner[0])
        ax.imshow(inputs[i])
        fig.add_subplot(ax)
        ax = plt.Subplot(fig, inner[1])
        t = ax.text(0.5, 0.5, "trueClass=%d" % (tc))
        t.set_ha("center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
        ax = plt.Subplot(fig, inner[2])
        # t = ax.text(0.5, 0.5, "crossEntropy=%f" % (-1*tc*np.log(pcp)))
        t = ax.text(0.5, 0.5, "predictedClass=%d" % (pc))
        t.set_ha("center")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)

    wandb.log({"predictedClasses":wandb.Image(plt)})

  def modifyBackProp(self):
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({"Relu":"GuidedRelu"}):
      layers = [layer for layer in self.model.layers[1:] if hasattr(layer, "activation")]
      
      for layer in layers:
        if layer.activation == ke.activations.relu:
          layer.activation = tf.nn.relu
      
      filepath = './best_model/best_model_acc'
      self.model = ke.models.load_model(filepath, compile = True)
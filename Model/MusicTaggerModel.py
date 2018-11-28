import tensorflow as tf
import numpy as np
from batcher import Batcher

class MusicTaggerModel(object):

    def __init__(self, config):
        self.config = config

    def trainClassifier(self):
        """
            Train CnnClassifier

        :param
        :return:
        """
        session = tf.Session()

        print("Initializing TF Graph ... ")
        #forward pass
        self.cnnClassifier()
        #backward pass
        self.backwardPass()
        session.run(tf.global_variables_initializer())
        print("Graph initialized ...")
        #Start Training
        batcher = Batcher(self.config)
        print("Starting training ... ")
        for epochId in range(self.config.numEpochs):
            iterations =0
            trainImages, trainLabels = batcher.getNextTrainBatch(self.config.batchSize)
            while len(trainImages)>0:

                feedDict = {}
                feedDict[self.melSpectrogramImages] = trainImages
                feedDict[self.tags] = trainLabels
                feedDict[self.keepProb] = self.config.keepProb
                feedDict[self.learningRate] = self.config.initLearningRate

                _, trainLoss, trainLogits = session.run([self.trainStep, self.loss, self.logits], feedDict)

                trainPrecision = self.precision(trainLogits, trainLabels)
                trainRecall = self.recall(trainLogits, trainLabels)
                trainF1 = self.f1Score(trainLogits, trainLabels)

                print("TRAIN# Epoch:{} - Loss:{} - Prec:{} - Rec:{} - f1:{}".format(epochId,
                                                                                    trainLoss,
                                                                                    trainPrecision,
                                                                                    trainRecall,
                                                                                    trainF1))

                if self.config.saveModel and iterations%self.config.saveModelIteration ==0:
                    print("Running Validation ")
                    validImages, validLabels = batcher.getNextValidBatch(self.config.batchSize)
                    while len(validImages)>0:

                        feedDict = {}
                        feedDict[self.melSpectrogramImages] = validImages
                        feedDict[self.tags] = validLabels
                        feedDict[self.keepProb] = 1.0
                        # feedDict[self.learningRate] = self.config.initLearningRate

                        validLoss, validPrediction = session.run([self.loss, self.predictions], feedDict)
                        validPrecision = self.precision(validPrediction, validLabels)
                        validRecall = self.recall(validPrediction, validLabels)
                        validF1 = self.f1Score(validPrediction, validLabels)

                        print("VALID# Loss:{} - Prec:{} - Rec:{} - f1:{}".format(validLoss,
                                                                                validPrecision,
                                                                                validRecall,
                                                                                validF1))
                        validImages, validLabels = batcher.getNextValidBatch(self.config.batchSize)

                trainImages, trainLabels = batcher.getNextTrainBatch(self.config.batchSize)
                iterations+=1



    def initPlaceholders(self):
        self.melSpectrogramImages = tf.placeholder(shape=[None, self.config.melBins, self.config.timeSteps, self.config.channels],
                                                  dtype=tf.float32)
        self.tags = tf.placeholder(shape=[None, self.config.maxLabels], dtype=tf.float32)

        self.keepProb = tf.placeholder(dtype=tf.float32)
        self.learningRate = tf.placeholder(dtype=tf.float32)

    def cnnClassifier(self, finalLayerSize=None):

        self.initPlaceholders()
        self.melSpectrogramEmbedding = self.getConvolutionalEmbedding(self.melSpectrogramImages,
                                                   self.config.filterShapes,
                                                   self.config.poolWindowShapes)

        # Flatten final result
        self.flatMelSpectrogramEmbedding = tf.reshape(self.melSpectrogramEmbedding,
                                                      shape=[-1, 2048])

        #Fully Connected Layers - None

        # Output layer
        if finalLayerSize==None:
            finalLayerSize = self.config.maxLabels
        self.weight1 = tf.get_variable("weight1",
                                       dtype=tf.float32,
                                       shape=[2048, finalLayerSize])
        self.bias1 = tf.get_variable("bias1",
                                     dtype=tf.float32,
                                     shape=[finalLayerSize])
        self.fcResult = tf.matmul(self.flatMelSpectrogramEmbedding, self.weight1)+self.bias1

        # Predictions using sigmoid
        self.logits = tf.nn.sigmoid(self.fcResult)
        self.predictions = tf.cast(self.logits+0.5, tf.int32)

        # calculate loss(logistic loss)
        self.loss = tf.reduce_mean(tf.negative(tf.multiply(self.tags, tf.log(self.logits + 0.000000000001)) +
                                tf.multiply((1-self.tags), tf.log(1-self.logits))))

    def backwardPass(self):

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learningRate)
        self.grads_vars = self.optimizer.compute_gradients(self.loss)
        self.trainStep = self.optimizer.apply_gradients(self.grads_vars)
        return self.trainStep

    def getConvolutionalEmbedding(self, inputImages, filterShapes, poolWindowShapes):
        """

        :param inputImages: [BatchSize, mel-bins, timeFrame, Channels]
        :param filters: [ list of filter([filter_height, filter_width, in_channels, out_channels]) ]
        :return: Convolutional Embedding
        """
        input = inputImages
        self.results = []
        self.means = []
        self.variances = []
        for filterId, filter in enumerate(filterShapes):

            filter = tf.get_variable("filter_{}".format(filterId),
                                     dtype=tf.float32,
                                     shape=filterShapes[filterId],
                                     initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32))

            # bias = tf.get_variable("bias_{}",format(filterId),
            #                        dtype=tf.float32,
            #                        initializer=tf.zeros_initializer)

            result = tf.nn.conv2d(input,
                                  filter,
                                  strides=self.config.convStrides[filterId],
                                  padding="SAME")
            mean, var = tf.nn.moments(result, axes=-1, keep_dims=True)
            self.means.append(mean)
            self.variances.append(var)

            normalizedResult = tf.nn.batch_normalization(result,
                                                         mean=mean,
                                                         variance=var,
                                                         offset=0,
                                                         scale=0.99,
                                                         variance_epsilon=0.00001)
            resultMaxPooled = tf.nn.max_pool(normalizedResult,
                           ksize=poolWindowShapes[filterId],
                           strides= self.config.poolStrides[filterId],
                           padding="VALID")

            resultActivated = tf.nn.relu(resultMaxPooled)
            self.results.append(resultActivated)
            input = tf.nn.dropout(resultActivated, keep_prob= self.keepProb)

        return self.results[-1]

    def getAccuracy(self, predictions, labels):
        predictions += 0.5
        predictions[predictions<1] = 0
        return float(np.sum(np.logical_and(predictions, labels)))/np.sum(labels)

    def precision(self, predictions, labels):
        predictions += 0.5
        predictions[predictions<1] = 0
        return float(np.sum(np.logical_and(predictions, labels)))/np.sum(predictions)

    def recall(self, predictions, labels):
        predictions += 0.5
        predictions[predictions<1] = 0
        return float(np.sum(np.logical_and(predictions, labels)))/np.sum(labels)

    def f1Score(self, predictions, labels):
        prec = self.precision(predictions, labels)
        recall = self.recall(predictions, labels)
        return 2*prec*recall/(prec+recall)

import tensorflow as tf
import numpy as np
from batcher import Batcher
import os
from sklearn import metrics

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
        # forward pass
        self.cnnClassifier()
        # backward pass
        self.backwardPass()
        #Summary for grpahs

        self.initSummary(session)
        saver = tf.train.Saver()
        iterations = None
        if self.config.restoreModel and os.path.exists(self.config.savedModelPath):
            print(" - Restoring Saved Model ...")
            saver.restore(session, tf.train.latest_checkpoint(self.config.savedModelPath))
            iterations=tf.train.global_step(session, self.globalStep)
        else:
            print(" - Initializing with fresh parameters ...")
            session.run(tf.global_variables_initializer())
            iterations = 0
        ans = input("Do you want to continue : ")
        if (ans != 1):
            exit(1)

        print("Graph initialized ...")

        # Start Training
        self.batcher = Batcher(self.config)

        print("Starting training ... ")
        maxValidF1 = -1.0

        maxTrainF1 = -1.0

        # logger = open("log.txt", "w")

        for epochId in range(self.config.numEpochs):
            self.batcher.resetTrainBatcher()
            trainImages, trainLabels = self.batcher.getNextTrainBatch(self.config.batchSize)
            while iterations<self.config.maxIterations and len(trainImages) > 0:

                feedDict = {}
                feedDict[self.melSpectrogramImages] = trainImages
                feedDict[self.tags] = trainLabels
                feedDict[self.keepProb] = self.config.keepProb
                # feedDict[self.learningRate] = self.config.initLearningRate

                _, trainLoss, trainLogits, iterations, trainSummary = session.run([self.trainStep, self.loss, self.logits, self.globalStep, self.summariesOp], feedDict)
                # print(trainLogits)
                # print(trainLabels)
                # print(fc1)
                # logger.write(str(trainLogits))
                # logger.write(str(trainLabels))

                self.trainSummaryWriter.add_summary(trainSummary, global_step=iterations)

                trainPrecision = self.precision(trainLogits, trainLabels)
                trainRecall = self.recall(trainLogits, trainLabels)
                trainF1 = self.f1Score(trainLogits, trainLabels)

                if trainF1 > maxTrainF1:
                    maxTrainF1 = trainF1
                    print("Max train F1: {}".format(maxTrainF1))


                print("TRAIN# Epoch:{} - Iter:{} - Loss:{} - Prec:{} - Rec:{} - f1:{}".format(epochId,
                                                                                              iterations,
                                                                                              trainLoss,
                                                                                              trainPrecision,
                                                                                              trainRecall,
                                                                                              trainF1))

                if self.config.saveModel and iterations % self.config.saveModelIteration == 0:
                # if iterations ==1:
                    print("Running Validation ")
                    self.batcher.resetValidBatcher()
                    avgValidLoss = 0.0
                    avgValidPrec = 0.0
                    avgValidRecall = 0.0
                    avgValidF1 = 0.0

                    validImages, validLabels = self.batcher.getNextValidBatch(self.config.batchSize)
                    validLogitsList = []
                    validLabelsList = []

                    while len(validImages) > 0:

                        feedDict = {}
                        feedDict[self.melSpectrogramImages] = validImages
                        feedDict[self.tags] = validLabels
                        feedDict[self.keepProb] = 1.0
                        # feedDict[self.learningRate] = self.config.initLearningRate

                        validLoss, validLogits, summary = session.run([self.loss, self.logits, self.summariesOp], feedDict)
                        if self.config.summary ==True:
                            self.validSummaryWriter.add_summary(summary, iterations)

                        validLogitsList.extend(validLogits)
                        validLabelsList.extend(validLabels)

                        validPrecision = self.precision(validLogits, validLabels)
                        validRecall = self.recall(validLogits, validLabels)
                        validF1 = self.f1Score(validLogits, validLabels)

                        validImages, validLabels = self.batcher.getNextValidBatch(self.config.batchSize)

                        avgValidLoss = (avgValidLoss * (self.batcher.currentValidId-self.config.batchSize) + validLoss)/self.batcher.currentValidId
                        avgValidPrec = (avgValidPrec * (self.batcher.currentValidId - self.config.batchSize) + validPrecision) / self.batcher.currentValidId
                        avgValidRecall = (avgValidRecall * (self.batcher.currentValidId - self.config.batchSize) + validRecall) / self.batcher.currentValidId
                        avgValidF1 = (avgValidF1 * (self.batcher.currentValidId - self.config.batchSize) + validF1) / self.batcher.currentValidId


                    avgAUC, AUC = self.aucRoc(validLogitsList, validLabelsList)
                    print("VALID# Loss:{} - Prec:{} - Rec:{} - f1:{} - avgAUC:{}".format(avgValidLoss,
                                                                                         avgValidPrec,
                                                                                         avgValidRecall,
                                                                                         avgValidF1,
                                                                                         avgAUC))

                    if avgValidF1 > maxValidF1:
                        print("Saving Model to : {}".format(self.config.savedModelPath))
                        saver.save(session, os.path.join(self.config.savedModelPath, self.config.modelName)+"{}.ckpt".format(iterations))
                        print("--- Saved Model to : {}".format(self.config.savedModelPath))

                trainImages, trainLabels = self.batcher.getNextTrainBatch(self.config.batchSize)
        print("Max Train F1: {}".format(maxTrainF1))
        # logger.close()

    def initPlaceholders(self):
        self.melSpectrogramImages = tf.placeholder(
            shape=[None, self.config.melBins, self.config.timeSteps, self.config.channels],
            dtype=tf.float32)
        self.tags = tf.placeholder(shape=[None, self.config.maxLabels], dtype=tf.float32)

        self.keepProb = tf.placeholder(dtype=tf.float32)
        # self.learningRate = tf.placeholder(dtype=tf.float32)

    def cnnClassifier(self, finalLayerSize=None):

        self.initPlaceholders()
        self.melSpectrogramEmbedding = self.getConvolutionalEmbedding(self.melSpectrogramImages,
                                                                      self.config.filterShapes,
                                                                      self.config.poolWindowShapes)

        # Flatten final result
        self.flatMelSpectrogramEmbedding = tf.reshape(self.melSpectrogramEmbedding,
                                                      shape=[-1, 2048])

        # Fully Connected Layers - None
        # self.weight1 = tf.get_variable("weightFC1",
        #                                initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
        #                                                                                 dtype=tf.float32),
        #                                dtype=tf.float32,
        #                                shape=[2048, 2048])
        # self.bias1 = tf.get_variable("biasFC1",
        #                              initializer=tf.zeros_initializer,
        #                              dtype=tf.float32,
        #                              shape=[2048])
        # self.fc1 = tf.matmul(self.flatMelSpectrogramEmbedding, self.weight1) + self.bias1

        # Output layer
        if finalLayerSize == None:
            finalLayerSize = self.config.maxLabels
        self.weightOutput = tf.get_variable("weightOutput",
                                        initializer= tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                                                                           dtype=tf.float32),
                                       dtype=tf.float32,
                                       shape=[2048, finalLayerSize])
        self.biasOutput = tf.get_variable("biasOutput",
                                     initializer=tf.zeros_initializer,
                                     dtype=tf.float32,
                                     shape=[finalLayerSize])
        # self.fcResult = tf.matmul(self.flatMelSpectrogramEmbedding, self.weight1) + self.bias1
        self.fcResult = tf.matmul(self.flatMelSpectrogramEmbedding, self.weightOutput) + self.biasOutput
        # Predictions using sigmoid
        self.logits = tf.nn.sigmoid(self.fcResult)
        # self.predictions = tf.cast(self.logits + 0.5, tf.int32)
        #
        # # calculate loss(logistic loss)
        # self.loss = tf.reduce_mean(tf.negative(tf.multiply(self.tags, tf.log(self.logits + 0.000000000001)) +
        #                                        tf.multiply((1 - self.tags), tf.log(1 - self.logits + 0.000000000001))))
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fcResult, labels=self.tags))

    def backwardPass(self):

        self.globalStep = tf.get_variable("globalStep", shape=[], initializer=tf.zeros_initializer, dtype=tf.int32,
                                          trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.config.initLearningRate)
        self.grads_vars = self.optimizer.compute_gradients(self.loss)
        self.trainStep = self.optimizer.apply_gradients(self.grads_vars, global_step=self.globalStep)
        return self.trainStep

    def initSummary(self, session):

        tf.summary.scalar('cross_entropy', self.loss)


        l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
        for gradient, variable in self.grads_vars:
            tf.summary.histogram("gradients/" + variable.name, l2_norm(gradient))
            tf.summary.histogram("variables/" + variable.name, l2_norm(variable))

        self.summariesOp = tf.summary.merge_all()
        self.trainSummaryWriter = tf.summary.FileWriter(self.config.summariesPath+"/train", session.graph)
        self.validSummaryWriter = tf.summary.FileWriter(self.config.summariesPath+"/valid", session.graph)


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
                                     # initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None,
                                     #                                                       dtype=tf.float32)
                                     initializer=tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
                                     )

            # bias = tf.get_variable("bias_{}",format(filterId),
            #                        dtype=tf.float32,
            #                        initializer=tf.zeros(shape=[]))

            result = tf.nn.conv2d(input,
                                  filter,
                                  strides=self.config.convStrides[filterId],
                                  padding="SAME")
            # result = tf.nn.bias_add(result, bias)
            mean, var = tf.nn.moments(result, axes=-1, keep_dims=True)
            self.variable_summaries(name='mean_{}'.format(filterId), var=mean)
            self.variable_summaries(name='var_{}'.format(filterId), var = var)

            self.means.append(mean)
            self.variances.append(var)

            normalizedResult = tf.nn.batch_normalization(result,
                                                         mean=mean,
                                                         variance=var,
                                                         offset=0,
                                                         scale=0.99,
                                                         variance_epsilon=0.00001)

            tf.summary.histogram('convBNPre-Activations_{}'.format(filterId), normalizedResult)

            resultMaxPooled = tf.nn.max_pool(normalizedResult,
                                             ksize=poolWindowShapes[filterId],
                                             strides=self.config.poolStrides[filterId],
                                             padding="VALID")

            resultActivated = tf.nn.relu(resultMaxPooled)
            tf.summary.histogram('Activations_{}'.format(filterId), resultActivated)

            input = tf.nn.dropout(resultActivated, keep_prob=self.keepProb)
            self.results.append(resultActivated)


        return self.results[-1]

    # def getAccuracy(self, predictions, labels):
    #     predictions += 0.5
    #     predictions[predictions < 1] = 0
    #     return float(np.sum(np.logical_and(predictions, labels))) / np.sum(labels)

    def variable_summaries(self, name , var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def precision(self, predictions, labels):
        if np.sum(predictions)==0: return 0
        predictions += 0.5
        predictions[predictions < 1] = 0
        return float(np.sum(np.logical_and(predictions, labels))) / np.sum(predictions)

    def recall(self, predictions, labels):
        predictions += 0.5
        predictions[predictions < 1] = 0
        return float(np.sum(np.logical_and(predictions, labels))) / np.sum(labels)

    def f1Score(self, predictions, labels):
        prec = self.precision(predictions, labels)
        recall = self.recall(predictions, labels)
        if prec == 0 or recall == 0: return 0
        return 2 * prec * recall / (prec + recall)

    def aucRoc(self, logitsList, labelsList):

        yTrue = np.reshape(np.array(labelsList), newshape=[-1, self.config.maxLabels])
        yScore =  np.reshape(np.array(logitsList), newshape=[-1, self.config.maxLabels])
        AUC = []
        # aucList = []
        for i in range(self.config.maxLabels):
            AUC.append(metrics.roc_auc_score(yTrue[i,:], yScore[i,:]))

        avgAUC = np.mean(AUC)
        return avgAUC, AUC
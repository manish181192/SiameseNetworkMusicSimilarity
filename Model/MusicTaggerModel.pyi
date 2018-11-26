import tensorflow as tf
class MusicTaggerModel(object):

    def __init__(self, config):
        self.config = config


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
                                                      shape=[-1])

        #Fully Connected Layers - None

        # Output layer
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
        self.loss = tf.negative(tf.multiply(self.tags, self.logits) +
                                tf.multiply((1-self.tags), tf.log(1-self.logits)))

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
            mean, var = tf.nn.moments(result, axes=-1)
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
                           padding="VAID")

            resultActivated = tf.nn.relu(resultMaxPooled)

            input = tf.nn.dropout(resultActivated, keep_prob= self.keepProb)

        return result
class ModelConfig(object):

    def __init__(self):


        # Data config
        self.trainSongListFile = "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/MTAT_split/train_list_pub.cP"
        self.trainLabelsFile = "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/MTAT_split/y_train_pub.npy"

        self.validSongListFile = "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/MTAT_split/valid_list_pub.cP"
        self.validLabelsFile = "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/MTAT_split/y_valid_pub.npy"

        self.testSongListFile = "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/MTAT_split/test_list_pub.cP"
        self.testLabelsFile = "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/MTAT_split/y_test_pub.npy"

        self.imagesFolder = "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/iMelSpectrograms"

        self.maxLabelsFile = "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/MTAT_split/top50_tags.txt"


        self.labelsList = open(self.maxLabelsFile).readlines()
        self.maxLabels = len(self.labelsList)

        self.melBins = 96
        self.timeSteps = 1366
        self.channels = 1

        # Training
        self.initLearningRate = 0.05
        self.keepProb = 0.5
        self.batchSize = 4
        self.numEpochs = 2
        self.saveModelIteration = 100
        self.saveModel = False

        # Model configuration

        self.filterShapes = [[3,3,1,128],
                             [3,3,128,384],
                             [3,3,384,768],
                             [1,1,768,2048]]
        self.convStrides =[[1,1,1,1],
                           [1,1,1,1],
                           [1,1,1,1],
                           [1,1,1,1]]

        self.poolWindowShapes = [[1, 2, 4, 1],
                                 [1, 4, 5, 1],
                                 [1, 3, 8, 1],
                                 [1, 4, 8, 1]]


        self.poolStrides = [[1, 2, 4, 1],
                            [1, 4, 5, 1],
                            [1, 3, 8, 1],
                            [1, 4, 8, 1]]


class ModelConfig(object):

    def __init__(self):
        self.melBins = 96
        self.timeSteps = 1366
        self.channels =1
        self.maxLabels = 50
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


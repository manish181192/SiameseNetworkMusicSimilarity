import numpy as np
# import pandas as pd
import os
class Batcher(object):

    def __init__(self, config):

        self.dataConfig = config

        self.trainSongs = np.load(self.dataConfig.trainSongListFile)
        self.trainLabels = np.load(self.dataConfig.trainLabelsFile)
        self.trainSize = len(self.trainLabels)
        self.trainIds = np.arange(0, self.trainSize)

        self.validSongs = np.load(self.dataConfig.validSongListFile)
        self.validLabels = np.load(self.dataConfig.validLabelsFile)
        self.validSize = len(self.validLabels)
        self.validIds = np.arange(0, self.validSize)

        self.testSongs = np.load(self.dataConfig.testSongListFile)
        self.testLabels = np.load(self.dataConfig.testLabelsFile)
        self.testSize = len(self.testLabels)

        self.currentTrainId = 0
        self.trainEpochsCompleted = 0
        self.currentValidId = 0
        self.currentTestId = 0

    def resetTrainBatcher(self):

        self.currentTrainId = 0
        self.currentValidId = 0
        np.random.shuffle(self.trainIds)
        np.random.shuffle(self.validIds)
        self.trainEpochsCompleted+=1

    def resetTestBatcher(self):

        self.currentTestId =0

    def getNextTrainBatch(self, batchSize):

        if self.currentTrainId>=self.trainSize:
            self.resetTrainBatcher()

        trainBatchIds = self.trainIds[self.currentTrainId: self.currentTrainId+batchSize]
        songsBatch = [ self.trainSongs[i] for i in trainBatchIds]
        labelsBatch = [ self.trainLabels[i] for i in trainBatchIds]

        songsMelSpectrogram = self.getMelSpectrogram(songsBatch)

        self.currentTrainId += batchSize

        return songsMelSpectrogram, np.array(labelsBatch)

    def getNextValidBatch(self, batchSize):

        if self.currentValidId>=self.validSize:
            return [],[]

        validBatchIds = self.validIds[self.currentValidId: self.currentValidId + batchSize]
        songsBatch = [self.trainSongs[i] for i in validBatchIds]
        labelsBatch = [self.trainLabels[i] for i in validBatchIds]

        songsMelSpectrogram = self.getMelSpectrogram(songsBatch)

        self.currentValidId += batchSize

        return songsMelSpectrogram, np.array(labelsBatch)

    def getNextTestBatch(self, batchSize):

        if self.currentTestId>=self.testSize:
            return [], []

        songsBatch = [self.testSongs[i] for i in range(self.currentTestId, self.currentTestId+batchSize)]
        labelsBatch = [self.testLabels[i] for i in range(self.currentTestId, self.currentTestId+batchSize)]

        songsMelSpectrogram = self.getMelSpectrogram(songsBatch)

        self.currentTestId += batchSize

        return songsMelSpectrogram, np.array(labelsBatch)

    def getMelSpectrogram(self, songsBatch):

        songsMelSpectrogramsList =[]
        for song in songsBatch:
            split = song.split("/")
            folder = split[0]
            song = split[1]
            filepath = "###"+song.replace(".npy", ".mp3.npy")
            if folder!="3":
                filepath = folder+filepath
            songImage = np.load(os.path.join(self.dataConfig.imagesFolder, filepath))
            songImage = np.expand_dims(np.squeeze(songImage), axis=-1)
            songsMelSpectrogramsList.append(songImage)
        songsMelSpectrograms = np.array(songsMelSpectrogramsList)
        return songsMelSpectrograms

if __name__ =="__main__":
    from modelConfig import ModelConfig

    config = ModelConfig()
    batcher = Batcher(config)
    batcher.resetTrainBatcher()
    trainImages, trainLabels = batcher.getNextTrainBatch(batchSize=5)
    validImages, validLabels = batcher.getNextValidBatch(batchSize=5)
    testImages, testLabels = batcher.getNextTestBatch(batchSize=5)
    print("Done")
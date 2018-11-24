from multiprocessing import Pool
import audio_preprocessor as ap
import sys
import os
class MultiProcessing(object):

    def __init__(self, targetFunction, num_processes=1):
        self.numProcesses = num_processes
        self.targetFunction = targetFunction
        self.pool = Pool(self.numProcesses)

    def runProcesses(self, arguments):
        self.pool.map(self.targetFunction, arguments)


if __name__=="__main__":

    def listAllDirectories(dirPrefix):
        listArgsDict = []
        return [ {'directoryName': dir,"directoryPrefix": dirPrefix} for dir in os.listdir(dirPrefix)]


    # sampleDirectories = ["/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/mp3/sample_0",
    #             "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/mp3/sample_1"]
    if len(sys.argv)<3:
        print("Usage : python Multiprocessing.py <super_directory> <numThreads>")

    superDirectory = sys.argv[1]
    numThreads = sys.argv[2]

    argsFeed = listAllDirectories(superDirectory)

    multiProcessInstance = MultiProcessing(ap.processAudioFolder , 2)
    # multiProcessInstance.runProcesses(sampleDirectories)
    multiProcessInstance.runProcesses(argsFeed)


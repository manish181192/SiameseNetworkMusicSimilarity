import os
import numpy as np
from MultiProcessing import MultiProcessing
import time

def processFolder(melSpectrogramFile):

    print("Processing File : {}".format(melSpectrogramFile))
    start_time = time.time()
    outFile = "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/iMelSpectrograms/"
    folderName = melSpectrogramFile.split("/")[-1].strip("mp3").strip(".npy")
    songsImageDict = np.load(melSpectrogramFile).item()
    print("File {} loaded - time_taken : {}".format(melSpectrogramFile, (time.time()-start_time)/60))
    for song in songsImageDict.keys():
        npyFileName = folderName+"###"+song.strip(".mp3")
        np.save(outFile+npyFileName, songsImageDict[song])

    print("---- Finished Processing : {} ".format(melSpectrogramFile))
    
if __name__=="__main__":

    def listAllFiles(directory): return [os.path.join(directory, file) for file in os.listdir(directory) ]

    melSpectrogramFolder = "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/melSpectograms"
    #processFolder(melSpectrogramFolder)
    argsFeed = listAllFiles(melSpectrogramFolder)

    multiProcessInstance = MultiProcessing(processFolder, 2)
    multiProcessInstance.runProcesses(argsFeed)

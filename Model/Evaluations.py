import numpy as np
from batcher import Batcher
from modelConfig import ModelConfig
import heapq

class priorityQueue(object):
    def __init__(self, maxSize):

        self.heap = []
        self.heapSize = 0
        self.maxHeapSize = maxSize
        self.smallestValue = 100000

    def heapInsert(self, item, value):

        if self.heapSize< self.maxHeapSize:
            heapq.heappush(self.heap, (value, item))
            self.heapSize+=1
            self.smallestValue = self.heap[0][0]
        else:
            if value> self.smallestValue:
                print(value)
                self.heapPop()
                heapq.heappush(self.heap, (value, item))
                self.heapSize+=1
                self.smallestValue = self.heap[0][0]
    def heapPop(self):

        if self.heapSize == 0:
            return None
        else:
            element = heapq.heappop(self.heap)
            self.heapSize-=1
        return element


config = ModelConfig()
testEmbeddings = np.load("/home/manish/CS543/MusicSimilarity/Model/testEmbedings.npy")
batcher = Batcher(config)
print(len(batcher.testLabels))
print(testEmbeddings.shape)

dictScores = {}

i=0
heap = priorityQueue(maxSize=5)

while i< len(batcher.testSongs):
    j=i+1
    while j<len(batcher.testSongs):
        score = np.dot(testEmbeddings[i], testEmbeddings[j])
        # print(score)
        dictScores[(i,j)] = score
        heap.heapInsert((i,j), score)
        j+=1
    i+=1


print(heap.heap)
for tuple in heap.heap:
    score = tuple[0]
    pairSong = tuple[1]
    print("{} \n {} \n\nLabels: {} \n {} score:{}\n\n".format(batcher.testSongs[pairSong[0]],
                                                              batcher.testSongs[pairSong[1]],
                                                              batcher.testLabels[pairSong[0]],
                                                              batcher.testLabels[pairSong[0]],
                                                              score))

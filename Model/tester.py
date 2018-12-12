import tensorflow as tf
from MusicTaggerModel import MusicTaggerModel
from modelConfig import ModelConfig
if __name__ == "__main__":

    config = ModelConfig()
    musicClassifier = MusicTaggerModel(config)
    musicClassifier.testClassifier()
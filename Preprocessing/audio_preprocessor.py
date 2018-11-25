import os
from os import listdir
import librosa
import numpy as np
import time
def compute_melgram(audio_path):
    ''' Compute a mel-spectrogram and returns it in a shape of (1,1,96,1366), where
    96 == #mel-bins and 1366 == #time frame

    parameters
    ----------
    audio_path: path for the audio file.
                Any format supported by audioread will work.
    More info: http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load

    '''

    # mel-spectrogram parameters
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # to make it 1366 frame..

    src, sr = librosa.load(audio_path, sr=SR)  # whole signal
    n_sample = src.shape[0]
    n_sample_fit = int(DURA*SR)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA*SR) - n_sample,))))
    elif n_sample > n_sample_fit:  # if too long
        src = src[(n_sample-n_sample_fit)/2:(n_sample+n_sample_fit)/2]
    logam = librosa.amplitude_to_db
    melgram = librosa.feature.melspectrogram
    ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                        n_fft=N_FFT, n_mels=N_MELS)**2)
    ret = ret[np.newaxis, np.newaxis, :]
    # librosa.display.specshow(ret, sr=sr, hop_length=512, x_axis='time', y_axis='mel')
    return ret

def processAudioFolder(argsFeed):
    directoryPrefix = argsFeed['directoryPrefix']
    directoryName = argsFeed['directoryName']
    directoryPath = os.path.join(directoryPrefix, directoryName)
    print("Processing started on Directory Path : {}".format(directoryPath))
    start_time = time.time()
    audioNameToMelSpec = {}
    failCount =0
    for i, f in enumerate(listdir(directoryPath)):
        if i % 100 ==0:
            print("\nFolder Name : {} | Files Processed : {}\n".format(directoryName, i))
        try:
            melSpectogram = compute_melgram(os.path.join(directoryPath,f))
        except:
            failCount+=1
            continue
        audioNameToMelSpec[f] = melSpectogram

    print("Finished Directory Name : {} | Fail Count : {} | Time taken : {} minutes".format(directoryName, failCount, float(time.time()-start_time)/60))
    print("Saving Numpy array for Directory Name : {}".format(os.path.join(directoryPrefix+directoryName+".npy")))
    np.save(os.path.join(directoryPrefix+directoryName+".npy"), audioNameToMelSpec)
    print(" ---- Finish Saved for Directory Name : {}".format(directoryName))
    return 1

if __name__=="__main__":
    # audioPath = "/home/manish/CS543/MusicSimilarity/Dataset/000/000002.mp3"
    print(processAudioFolder({"directoryPrefix": "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/mp3",
                              "directoryName": "6"}))

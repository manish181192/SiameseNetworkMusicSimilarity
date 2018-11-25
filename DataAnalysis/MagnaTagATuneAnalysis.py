import pandas as pd
import numpy as np
dataFile = "/home/manish/CS543/MusicSimilarity/Datasets/MagnaTagATune/annotations_final.csv"
df = pd.read_csv(dataFile, delimiter="\t")
print("No of classes : {}".format(len(df.columns)-2))

data = []
for colId, col in enumerate(df.columns):
    # print(df[col].value_counts())
    if colId in [0, len(df.columns)-1]:
        continue
    data.append([df[col].value_counts()[1], df[col].value_counts()[0]])
classFrequency = pd.DataFrame(data, index = df.columns[1:-1],columns=["1s", "0s"])
print(classFrequency)

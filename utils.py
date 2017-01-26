import pandas as pd
import numpy as np
import glob, os
from scipy import ndimage


def read_txts_and_combine(folder="img_labeled/logs/"):
    #Read all .txt files, put in one dataframe with name of webpage in one column, and save again as csv
    all_dfs = []
    for f in os.listdir(folder):
        if f.endswith(".txt"):        
            df = pd.read_csv(folder+f,sep=",")
            df["filename"] = ".".join(f.split(".")[:-1])
            all_dfs.append(df)
    df = pd.concat(all_dfs).reset_index(drop=1)
    df.to_csv(folder+"log_all.csv", sep=",", index=0)
    
def blur(path, radius):
    im = Image.open(path)    
    im1 = im.filter(ImageFilter.GaussianBlur(radius=radius))
    return im1
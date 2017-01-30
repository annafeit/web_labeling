import pandas as pd
import numpy as np
import glob, os
import matplotlib.pyplot as plt
from scipy import ndimage
import cPickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from skimage.measure import block_reduce
from scipy import misc


label_numbers =  {"Button":0, "Icon":1, "Image":2, "Input field":3,  "Text":4, "Navigation Menu":5}
number_label = {0:"Button", 1:"Icon", 2:"Image", 3:"Input field",  4:"Text", 5: "Navigation Menu"}

def read_txts_and_combine(folder="../img_labeled/logs/"):
    """
    Reads all .txt files, put in one dataframe with name of webpage in one column, and save again as csv
    """
    all_dfs = []
    for f in os.listdir(folder):
        if f.endswith(".txt"):        
            df = pd.read_csv(folder+f,sep=",")
            df["filename"] = ".".join(f.split(".")[:-1])
            all_dfs.append(df)
    df = pd.concat(all_dfs).reset_index(drop=1)
    df.to_csv(folder+"log_all.csv", sep=",", index=0)
    return df

def filter_top_100(df):
    """
    Only keeps those entries whose labels have more than 100 entries and replaces the label names by numbers
    """
    
    #Filter those out with less than 100 labels
    counts = df.groupby("label").count()
    labels_100 = list(counts[counts.id>100].index)
    df = df[df.label.isin(labels_100)]
    df = df[df.label.isin(label_numbers)]
    #convert labels to numbers
    df.loc[:,"label"] = df.label.apply(lambda x:label_numbers[x])
    return df
    
    

def read_transposed_image(path):
    """
    reads the given path into a RGB image and takes care of transposing it as expected by the labeling algorithm.
    """
    img = ndimage.imread(path, mode="RGB")
    img = img.transpose(2,0,1)
    return img


def pickle_images(path, df, image_size=""):
    """
    reads all png images stored in the given folder, categorizes them according to the given data frame, and saves a pickled dictionary 
    of images and labels.
    """
    all_imgs = []
    labels = []
    folder = path
    for f in os.listdir(folder):        
        if f.endswith(".png"):        
            img = read_transposed_image(folder+f)
            i = f.split("_")[0]
            name = ".".join("_".join(f.split("_")[1:]).split(".")[:-1])
            label = df[(df.id == i) & (df.filename == name)].label
            if len(label) > 0:
                label = label.iloc[0]
                if image_size != "":
                    new_image = resize_and_pad(img, image_size)
                    new_dir = folder+str(image_size)
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    misc.toimage(new_image).save(new_dir+"/"+f)
                    all_imgs.append(new_image)
                else:
                    all_imgs.append(img)                        
                labels.append(label)
    d = {"data":all_imgs, "labels":labels}
    with open('data_'+str(image_size)+'.pkl', 'wb') as handle:
        cPickle.dump(d, handle, protocol=cPickle.HIGHEST_PROTOCOL)
    
def load_pickle(path):
    """
    reads in a pickled dictionary and returns an array of images and an np.array of labels
    """
    with open(path, 'rb') as f:
        data = cPickle.load(f)
    images = np.asarray([i/np.float32(255) for i in data['data']])
    labels = np.asarray(data['labels'], dtype='int32')
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)
    return X_train, y_train, X_test, y_test

def resize_and_pad(img, target_size):
    """
    downsamples or interpolates the image such that the largest dimension is at the given size. 
    Then fills the rest with whitespace so that we have a rectangular image
    """    
    max_shape = np.max(img.shape)
    max_index = np.argmax(img.shape)
    if img.shape[1] < target_size and img.shape[2] < target_size:    
        #interpolate
        factor = float(target_size)/max_shape
        scaled_img = misc.imresize(img, (max(1,int(img.shape[1]*factor)), max(1,int(img.shape[2]*factor))), interp="nearest")
    else:
        #downsample maximum dimension to target size        
        factor = max_shape/float(target_size)
        scaled_img  = misc.imresize(img, (max(1,int(img.shape[1]/factor)), max(1,int(img.shape[2]/factor))), interp="nearest")    

    #fill up   
    padded_image = np.stack([np.lib.pad(scaled_img[:,:,0], 
                                        ((0,target_size-scaled_img.shape[0]),
                                         (0,target_size-scaled_img.shape[1])), 
                                        'constant', constant_values=(255)),
                            np.lib.pad(scaled_img[:,:,1], 
                                       ((0,target_size-scaled_img.shape[0]),
                                        (0,target_size-scaled_img.shape[1])), 
                                       'constant', constant_values=(255)),
                            np.lib.pad(scaled_img[:,:,2], 
                                       ((0,target_size-scaled_img.shape[0]),
                                        (0,target_size-scaled_img.shape[1])), 
                                       'constant', constant_values=(255))])
  
    
    return padded_image

def plot_image(img, label=""):
    """
    takes care of transposing the image such that it can be plotted, and plots it. 
    """ 
    if img.shape[0] == 3:
        img = img.transpose(1,2,0)
    fig,ax = plt.subplots(1)
    sns.set_style('white')
    ax.imshow(np.asarray(img))
    if label!="":
        plt.title(number_label[label])
    return fig,ax

def plot_grey_image(img):
    sns.set_style('white')
    fig,ax = plt.subplots(1)
    ax.imshow(img, cmap = plt.cm.Greys_r)
    
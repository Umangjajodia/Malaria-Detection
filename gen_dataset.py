import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
#from skimage import data, color
from skimage.transform import resize
import random 
from PIL import Image

path = os.getcwd()
imgsize = 50
data_set = []
categories = ['Uninfected', 'Parasitized']

for i in categories:
    path1 = os.path.join(path, i)
    templist = os.listdir(path1)
    for j in templist:
        try:
            img = cv2.imread(os.path.join(path1, j))
            img_arr = Image.fromarray(img, 'RGB')
            resized_img = img_arr.resize((50, 50))
            img45 = resized_img.rotate(45)
            img75 = resized_img.rotate(75)
            blur = cv2.blur(np.array(resized_img), (10, 10))
            data_set.append([np.array(resized_img), categories.index(i)])
            data_set.append([np.array(img45), categories.index(i)])
            data_set.append([np.array(img75), categories.index(i)])
            data_set.append([np.array(blur), categories.index(i)])
        except Exception as e:
                pass
            
random.shuffle(data_set)

features = []
labels = []
for i,j in data_set:
    features.append(i)
    labels.append(j)
    
features = np.array(features)
labels = np.array(labels)

print('Featuress : {} | labels : {}'.format(features.shape , labels.shape))

plt.figure(1 , figsize = (15 , 9))
n = 0 
for i in range(49):
    n += 1 
    r = np.random.randint(0 , features.shape[0] , 1)
    plt.subplot(7 , 7 , n)
    plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
    plt.imshow(features[r[0]])
    plt.title('{} : {}'.format('Parasitized' if labels[r[0]] == 1 else 'Uninfected' ,
                               labels[r[0]]) )
    plt.xticks([]) , plt.yticks([])
    
plt.show()


np.save('Features', features)
np.save("Labels", labels)

"""
import pickle
pickle_out = open("Features.pickle", "wb")
pickle.dump(features,pickle_out)
pickle_out.close()

pickle_out = open("Labels.pickle", "wb")
pickle.dump(labels,pickle_out)
pickle_out.close()
"""



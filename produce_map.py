import torch
import numpy as np
import os
from skimage.transform import resize
from skimage.io import imshow, imsave
import pickle
import matplotlib.pyplot as plt


from predict import predict, predictResnet, image_loader, test_loader
from cam.network.utils import SaveFeatures, imshow_transform

from model import model 
image_file = './temp-images/0e387530-1398-495a-b2c3-0717c5ca4a25.jpeg'


pos = []
neg = []

for image in os.listdir('./temp-images/pos/'): 
    image_file = f"./temp-images/pos/{image}"

    pred, prob = predictResnet(model,image_file,image[0:-5],generate_map=False)
    if pred == 1:
        pos.append([np.random.randint(low=20,high=50),prob.tolist()])
    else:
        neg.append([np.random.randint(low=-50,high=-20),prob.tolist()])


for image in os.listdir('./temp-images/neg/'): 
    image_file = f"./temp-images/neg/{image}"

    pred, prob = predictResnet(model,image_file,image[0:-5],generate_map=False)

    if pred == 0:
        neg.append([np.random.randint(low=20,high=50),prob.tolist()])
    else:
        pos.append([np.random.randint(low=-50,high=-20),prob.tolist()])

with open("./pos-list.txt", "wb") as fp:
    pickle.dump(pos, fp)

with open("./neg-list.txt", "wb") as fp:
    pickle.dump(neg, fp)

"""
with open("./pos-list.txt", "rb") as fp:
    pos = pickle.load(fp)
    
with open("./neg-list.txt", "rb") as fp:
    neg = pickle.load(fp)
 """

plt.figure()
pos = np.asarray(pos)
neg = np.asarray(neg)


plt.scatter(pos[:,1], pos[:,0], color=['green'], label='positive classification')
plt.legend(prop={'size': 10})

my_image = plt.savefig('./pos.png')

plt.figure()

plt.scatter(neg[:,1], neg[:,0], color=['red'], label='negative classification')
plt.legend(prop={'size': 10})

my_image = plt.savefig('./neg.png')


import torch
import numpy as np
import os
from skimage.transform import resize
from skimage.io import imshow, imsave

import matplotlib.pyplot as plt


from predict import predict, image_loader, test_loader
from cam.network.utils import SaveFeatures, imshow_transform

from model import model 
image_file = './temp-images/0e387530-1398-495a-b2c3-0717c5ca4a25.jpeg'

for image in os.listdir('./temp-images/'): 
    image_file = f'./temp-images/{image}'
    print(predict(model,image_file,image[0:-5],generate_map=True))



"""
res = torch.argmax(outputs.data).cpu().detach().numpy()
labels = ["healthy", "pneumonia"]

print(f'result: {res} - {labels[res]} ')

sf.remove()
arr = sf.features.cpu().detach().numpy()
features_data = arr[0]

ans_0 = np.dot(np.rollaxis(features_data,0,3), [1,0])
ans_1 = np.dot(np.rollaxis(features_data,0,3), [0,1])

if(res ==1):
    ans_1 = resize(ans_1, (224,224))
    ans1_int = ans_1.astype(np.uint8)
    plt.imsave('cam.png', ans_1, cmap='jet')

im = image_loader(test_loader, image_file)

plt.figure()
plt.subplots(figsize=(4,4))
plt.imshow(imshow_transform(im))
plt.imshow(ans_1, alpha=.4, cmap='jet')
my_image = plt.savefig('camcam.png')"""
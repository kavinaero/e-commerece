# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import os
import logging
import webbrowser
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

#initialize logging
logging.basicConfig(filename='abc_logistics.log',level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filemode='w')

rgb_color_dictionary={"green":(33, 191, 96),'yellow':(255,255,0),'pink':(232, 111, 132),'purple':(128, 0, 128),'blue':(84, 149, 240),'orange':(255,165,0),'gray':(112, 103, 103),
                      'black':(26, 21, 22),'white':(255, 255, 255),'red':(247, 22, 49)}

threshold_value={"green":97,'yellow':97,'pink':30,'purple':97,'blue':97,'orange':97,'gray':20,
                      'black':97,'white':80,'red':97}


#Rest API service


app=Flask(__name__)


# function to load images from folders
def load_images_from_folder(folder):
    images = []
    i = 1
    
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
           # make uniform size images
           images.append(img)
           i = i + 1
    return (images)
#function to extract foreground
def extract_background(image):
    rectangle = (0, 10, 60, 80)
    # Create initial mask
    mask = np.zeros(image.shape[:2], np.uint8)

    # Create temporary arrays used by grabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run grabCut
    cv2.grabCut(image, # Our image
                mask, # The Mask
                rectangle, # Our rectangle
                bgdModel, # Temporary array for background
                fgdModel, # Temporary array for background
                5, # Number of iterations
                cv2.GC_INIT_WITH_RECT) # Initiative using our rectangle

    # Create mask where sure and likely backgrounds set to 0, otherwise 1
    mask_2 = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')

    # Multiply image with new mask to subtract background
    image_rgb_nobg = image * mask_2[:, :, np.newaxis]
    return image_rgb_nobg

data_folder = r"F:/ABC_ecomerce_praveen_aerofolic/static/img/DATA"

# reading the image
images = load_images_from_folder(data_folder)
imagenames=os.listdir(r"F:/ABC_ecomerce_praveen_aerofolic/static/img/DATA")

@app.route('/homepage.html',methods=['GET','POST'])
def home():
        return render_template('homepage.html')

@app.route('/product.html', methods=['GET','POST'])
def product():
    image_names=os.listdir("./static/img/DATA")
    
    return render_template('product.html',image_names=image_names)

@app.route('/color.html',methods=['GET','POST'])
def colour_page():
    filtered_image_names=[]
    
        
    for i in range(len(imagenames)):
        
        img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        if request.args.get("type")=='white' or request.args.get("type")=='gray':
            
            img=extract_background(img)
        img = cv2.resize(img, (128,128))
        #reshaping pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = 2)
        kmeans.fit(img)
        #the cluster centers are our dominant colors.
        colors = kmeans.cluster_centers_
        colour_choice_rgb=rgb_color_dictionary[request.args.get("type")]
        match = False
        x = colour_choice_rgb
        
        for y in colors:
            dist = np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)
            
            if dist < threshold_value[request.args.get("type")]:
                match = True
        # Evaluate the decision:
        if match == True:
            filtered_image_names.append(imagenames[i])
            
        else:
            pass
    print(filtered_image_names)
    return render_template('color.html',file_names=filtered_image_names)
if __name__=='__main__':
     webbrowser.open_new('http://127.0.0.1:5000/homepage.html')
     app.run(port=5000,debug=True,use_reloader=False)
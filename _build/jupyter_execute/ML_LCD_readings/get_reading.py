#!/usr/bin/env python
# coding: utf-8

# # Get liquid display readings from video using OpenCV-Python
# Written by Lei Lei<br>
# Faculty of Engineering, University of Nottingham

# 1. Import packages

# In[1]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['image.cmap'] = 'gray'


# 2. The reading updates once per second. We only need to extract one frame per second from the video file. 

# In[2]:


def rescaleFrame(frame, scale = 0.3):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)
    dimensions = (width, height)
    
    return cv2.resize(frame, dimensions, interpolation = cv2.INTER_AREA)


# In[3]:


# Reading Video
capture=cv2.VideoCapture('/mnt/c/users/stcik/scire/papers/muon/80C.MOV')
count = 0
success = True
fps = int(capture.get(cv2.CAP_PROP_FPS))
out_folder='/mnt/c/users/stcik/scire/papers/muon/80C/frames'
os.makedirs(out_folder,exist_ok=True)
while success:
    success, image = capture.read()
#     print('read a new frame:', success)
    if count%(1*fps) == 0 :
        image = rescaleFrame(image)
        cv2.imwrite(os.path.join(out_folder,'frame%d.jpg'%count),image)
#         print('Successfully written frame%d!'%count)
    count+=1


# 3. We need to crop the image to focus on the liquid crystal diplay area, this will save computational work.<br>
# Define a crop function. Reshape and crop the image to save computation efforts.

# In[4]:


def crop(img, y=200, h=400, x=160, w=300):
    new_image = img[y:h, x:w]
    cv2.imwrite("Images/new_image.jpg", new_image)
    return new_image


# Then, we define a plot function to save a little bit of input work.

# In[5]:


def plot(image, cmap=None):
    plt.axis('off')
    plt.imshow(image)


# We can first use the first image in the frame directory to get an idea on what parameters we should use.

# In[6]:


image_path=r'/mnt/c/users/stcik/scire/papers/muon/80C/frames/frame6438.jpg'
image=cv2.imread(image_path)
new_image=crop(image, y=70, h=280, x=30, w=270)
plot(new_image)


# After we get proper parameters, we can batch crop the frames.

# In[7]:


import re
rDir = r'/mnt/c/users/stcik/scire/papers/muon/80C/frames'
out_folder='/mnt/c/users/stcik/scire/papers/muon/80C/cropped'
os.makedirs(out_folder,exist_ok=True)
savs=[]
for file in os.listdir(rDir):
    if file.endswith(".jpg"):
        savs.append(os.path.join(rDir, file))

for sav in savs:
    image_path=sav
    image=cv2.imread(image_path)
    new_image=crop(image, y=70, h=280, x=30, w=270)
    name = image_path.split('/')[-1]
    cv2.imwrite(os.path.join(out_folder, f'80C_{name}.jpg'), new_image)


# We need to have a look at the cropped images. For frames where camera moved a lot, we may need to mannually crop them.

# In[8]:


image_path=r'/mnt/c/users/stcik/scire/papers/muon/80C/frames/frame4350.jpg'
image=cv2.imread(image_path)
new_image = crop(image, y=70, h=280, x=36, w=276)
name = image_path.split('/')[-1]
cv2.imwrite(os.path.join(/mnt/c/users/stcik/scire/papers/muon/80C/frames/, f'80C_{name}'), new_image)


# 4. We now need to make binary image from the cropped image.

# In[10]:


def make_bin(img, gmin=150, gmax=255):
    """Make a binary image from the cropped image."""
    # Drop the color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Thresholding the image
    thresh, img_bin = cv2.threshold(img, gmin, gmax, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Invert the image
    img_bin = ~img_bin
    cv2.imwrite("Images/Image_bin.jpg", img_bin)
    return img_bin


# In[11]:


img_bin=make_bin(new_image)
plot(img_bin)


# 5. We now need to make final bin iamge for cv2 to find contours.<br>
# Firstly, we define parameters to resolve vertical and horizontal lines.

# In[12]:


line_min_width = 8
kernal_h = np.ones((4,line_min_width), np.uint8)
kernal_v = np.ones((line_min_width,4), np.uint8)


# In[13]:


img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
plot(img_bin_h)


# In[14]:


img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
plot(img_bin_v)


# In[15]:


img_bin_final=img_bin_h|img_bin_v
plot(img_bin_final)


# We can make the lines thicker so that the contours are easier to resolve.

# In[16]:


final_kernel = np.ones((3,3), np.uint8)
img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=1)
plot(img_bin_final)


# Now we can fin the contours and visualise the contours found.

# In[17]:


#Retrieve contours 
contours, hierarchy = cv2.findContours(img_bin_final, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#Create box-list
box = []
# Get position (x,y), width and height for every contour 
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    box.append([x,y,w,h])
print(box)


# In[18]:


img_box = new_image.copy()
for n in box:
    img_box = cv2.rectangle(img_box, (n[0],n[1]), (n[0]+n[2],n[1]+n[3]), (255,0,0), 2)
plt.axis('off')
plt.imshow(img_box)


# 6. We can now set up parameters to sort the contours found so that we keep our readings only.

# Firstly, lets sort our contours, so that the boxes are well ordered.

# In[19]:


# We will use the size and position of the contourbox to filter the boxes.
def get_reading(img, contours):
    # Get the width and height of the image
    width, height,_ = img.shape
    textcont=[]
    for c in contours:
        # Returns the location and width, height for every contour
        x, y, w, h = cv2.boundingRect(c)
#         print(h/height, y/(height-y-h))
        # We will only use the height of the numbers, because all numbers have about the same height but 1 is much narrower than other numbers.
        # We will only use the y positions, because the x positions has a larger distribution.
        if 0.5 > h/height > 0.1 and 1.4 > y/(height-y-h) > 0.4:
            textcont.append(c)
    return textcont


# In[20]:


textcont = get_reading(new_image, contours)


# In[21]:


textbox = []
for c in textcont:
    x, y, w, h = cv2.boundingRect(c)
    textbox.append([x,y,w,h])
print(textbox)


# In[22]:


img_box = new_image.copy()
for n in textbox:
    img_box = cv2.rectangle(img_box, (n[0],n[1]), (n[0]+n[2],n[1]+n[3]), (255,0,0), 2)
plot(img_box)


# 7. Now we can crop the image according to the textboxes. 

# In[23]:


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


# In[24]:


textcont, textboxes = sort_contours(textcont, method="left-to-right")
print(textboxes)


# In[25]:


new_img = new_image[textboxes[0][1]:textboxes[0][1]+textboxes[0][3], textboxes[0][0]:textboxes[-1][0]+textboxes[-1][2]]
plot(new_img)
cv2.imwrite('outs/80C_frame6438.jpg.jpg', new_img)


# 8. To batch process the cropped frames, we need to make functions.

# In[26]:


def make_bin(img, gmin=150, gmax=255):
    """Make a binary image from the cropped image."""
    # Drop the color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Thresholding the image
    thresh, img_bin = cv2.threshold(img, gmin, gmax, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Invert the image
    img_bin = ~img_bin
    cv2.imwrite("Images/Image_bin.jpg", img_bin)
    return img_bin

def get_lines(img, line_min_width = 8, kernal_h = np.ones((4,line_min_width), np.uint8), kernal_v = np.ones((line_min_width,4), np.uint8)):
    # Detect vertical lines from an image
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_v)
    # Detect horizontal lines
    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernal_h)
      
    img_bin_final=img_bin_h|img_bin_v
    
    final_kernel = np.ones((4, 4), np.uint8)
    img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=1)
    return img_bin_final

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def get_reading(img, img_path, contours, folder):
        # Get the width and height of the image
    width, height,_ = img.shape
    textcont=[]
    for c in contours:
        # Returns the location and width, height for every contour
        x, y, w, h = cv2.boundingRect(c)
#         print(h/height, y/(height-y-h))
        # We will only use the height of the numbers, because all numbers have about the same height but 1 is much narrower than other numbers.
        # We will only use the y positions, because the x positions has a larger distribution.
        if 0.50 > h/height > 0.15 and 1.2 > y/(height-y-h) > 0.12:
            textcont.append(c)
    textcont, textboxes = sort_contours(textcont, method="left-to-right")
    new_img = img[textboxes[0][1]:textboxes[0][1]+textboxes[0][3], textboxes[0][0]:textboxes[-1][0]+textboxes[-1][2]]
    name = img_path.split('/')[-1]
    cv2.imwrite(os.path.join(folder, f'{name}'), new_img)


# 9. Finally, we can use the above defined functions to batch process the frames and save the detected region into ./outs directory.

# In[27]:


savs=[]
rDir = r'/mnt/c/users/stcik/scire/papers/muon/80C/cropped'
out_folder=r'/mnt/c/users/stcik/scire/papers/muon/80C/outs'
os.makedirs(out_folder,exist_ok=True)

for file in os.listdir(rDir):
    if file.endswith(".jpg"):
        savs.append(os.path.join(rDir, file))
for sav in savs:
    image=cv2.imread(sav)
    img_bin = make_bin(image)
    img_bin = get_lines(img_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    get_reading(image, img_path=sav, contours = contours, folder = out_folder)


# 10. Now go to the ./outs directory, check the outputs. Change parameters and try again.<br>
# I have 250 frames, only 4 of them are not correctly detect. The success rate is 98.4 %.

# 11. If you have problem with certain images, you can go back to the manual mode and see what happened.<br>
# I have problem with the following file, which turned out to be the reading in this image is not clear. The number '3' is broken which makes the contour box split into two small one and filtered by the conditions I set in the get_reading function. If we set the lower height bound smaller, it will work.

# 12. It is useful to make the detected readings binary so that it is easier to be recognised by OCR packages. 

# In[303]:


rDir = r'/mnt/c/users/stcik/scire/papers/muon/90C/outs'
out_folder='/mnt/c/users/stcik/scire/papers/muon/90C/binary'
os.makedirs(out_folder, exist_ok=True)
savs=[]

for file in os.listdir(rDir):
    if file.endswith(".jpg"):
        savs.append(os.path.join(rDir,file))

for sav in savs:
    image_path=sav
    image=cv2.imread(sav)
    img_bin=make_bin(image, gmin=150, gmax=255)
    name = image_path.split('/')[-1]
    cv2.imwrite(os.path.join(out_folder, f'{name}'), img_bin)


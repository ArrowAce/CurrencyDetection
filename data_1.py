#Calling the necessary library
import os
#from picamera import PiCamera
from time import sleep
from imutils import paths
import pyttsx3
import cv2
from utils import *
#Importing the necessary library functions
import subprocess
import numpy as np
#Image acquisition using RasPi camera
#camera = PiCamera()
camera=cv2.VideoCapture(0)
#camera.start_preview()
count=0
while True:
    global img
    ret, img=camera.read()
    #gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Test", img)
    if not ret:
        break

    k=cv2.waitKey(1)
    if k%256==27:
       #For Esc Key
       print("Close")
       break
    elif k%256==32:
       #For Space Key
       print("Image "+str(count)+"saved")
       file= os.path.join(r"C:\Users\HP.DESKTOP-MRN8MIT\Desktop", str(count) +'.jpg')
       cv2.imwrite(file, img)
       count +=1
       break
#sleep(5)
#camera.capture('test.jpg')
#camera.stop_preview()
camera.release
cv2.destroyAllWindows
max_val = 8
max_pt = -1
max_kp = 0
orb = cv2.ORB_create()
#Importing the captured image into this program
#test_img = cv2.imread('files/test.jpg')
#test_img = cv2.imread(r'D:\Users\ajayshankar\Desktop\AJAY\Currency-of-VND-Recognition-master\Data\500.jpg')
#original = resize_img(test_img, 0.4)
original = resize_img(img, 0.4)
display('original', original)
#(kp1, des1) = orb.detectAndCompute(test_img, None)
(kp1, des1) = orb.detectAndCompute(img, None)
#Declaring the training set
#training_set=os.listdir(r"D:\Users\ajayshankar\Desktop\AJAY\Currency-of-VND-Recognition-master\New folder\New folder\Indian Currencies zip\Indian Currencies\10")
#training_set=list(paths.list_images(r"D:\Users\ajayshankar\Desktop\AJAY\Currency-of-VND-Recognition-master\New folder\New folder\Indian Currencies zip\Data"))
training_set=list(paths.list_images(r"D:\Users\ajayshankar\Desktop\AJAY\Currency Detection\New folder\New folder\Indian Currencies zip\Semi"))
print(training_set,end='\n')
#for i in range(0, len(training_set)):
c=0
for trainingData in training_set:

# train image
    #train_img = cv2.imread(training_set[i])
    train_img = cv2.imread(trainingData)
    cv2.imshow("frame", train_img)
    (kp2, des2) = orb.detectAndCompute(train_img, None)
# brute force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    all_matches = bf.knnMatch(des1,des2,k=2)
    #all_matches = bf.match(des1,des2)
    #all_matches = sorted(all_matches, key = lambda x:x.distance)
    #img3 = cv2.drawMatchesKnn(img,kp1,train_img, kp2, all_matches[:10], None, flags=2)
    #plt.imshow(img3)
    #plt.show()
    good = []
# if good then append to list of good matches 
    for m,n in (all_matches):
        #if m.distance< 0.789*c.distance :
        #if n< len(all_matches) - 1 and m.distance < 0.789 * all_matches[n+1].distance:
        if m.distance < 0.789 *n.distance:   
            good.append(m)
        if len(good) >max_val:
            max_val = len(good)
            max_pt = c
            max_kp = kp2
    print(c, ' ', trainingData, ' ', len(good))
    c+=1
    print (max_val)
if max_val !=8 :
    print(training_set[max_pt])
    print('good matches ', max_val)
    global note
    #note = str(training_set[max_pt])[6:-6][6:]
    note = rename((training_set[max_pt])[-12:-8])
    print('\nDetected denomination: Rs. '+ note)
    #Speech synthesis
    engine = pyttsx3.init()
    engine.say("Detected denomination is ")
    engine.say(note)
    engine.say("Dee huyi mudra ")
    engine.say("{}Rupaye ki Haye".format(note))
    engine.setProperty('volume', 1.0)
    engine.runAndWait()
else:
    print('No Matches')
#Speech synthesis
    engine = pyttsx3.init()
    engine.say("No match found.Try again.")
    engine.setProperty('volume', 0.9)
    engine.runAndWait()

    #playsound(r'D:/Users/HP.DESKTOP-MRN8MIT/Desktop/Data/'+ str(note)+'.mp3')
#os.system('aplay /home/pi/Desktop/project/'+str(note)+'.wav')
#os.system(r'D:\Users\HP.DESKTOP-MRN8MIT\Desktop\Data'+ '\\'+str(note)+'.mp3')
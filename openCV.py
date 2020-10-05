## 臉部辨識
import cv2
picpath=r"/Users/dongguan-cheng/PycharmProjects/pythonProject/haarcascade_frontalface_default.xml"
face_cascade=cv2.CascadeClassifier(picpath)
img=cv2.imread("96364051_2973651016064638_5160046036663664640_o.jpg",1)
faces=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=3,minSize=(10,10))
#標註右下角底色是黃色
cv2.rectangle(img,(img.shape[1]-113,img.shape[0]-20),(img.shape[1],img.shape[0]),(0,255,255),-1)
#標注找到多少的人臉
cv2.putText(img,"Finding"+str(len(faces))+"face",(img.shape[1]-113,img.shape[0]-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
##將人臉框起來，由於有可能找到好幾個臉所以用迴圈繪出來
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
cv2.namedWindow("face",cv2.WINDOW_NORMAL)
cv2.imshow("face",img)
cv2.waitKey(0)
#cv2.destroyWindow("face")

## 臉部存檔
from PIL import Image
num=1 #把檔名進行編號
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #用藍色框先框住臉
    filename="face"+str(num)+".jpg" #建立檔名
    image=Image.open("96364051_2973651016064638_5160046036663664640_o.jpg") #PIL模組開啟
    imageCrop=image.crop((x,y,x+w,y+h)) #進行裁切
    imageResize=imageCrop.resize((150,150),Image.ANTIALIAS) #高品質重置圖片大小
    imageResize.save(filename) #儲存大小
    num+=1
## 攝像機是否開啟
import cv2

capture=cv2.VideoCapture(0)
if capture.isOpened():
    while True:
        success,img =capture.read() #讀取攝影機所拍影像_success為布林值，img影像物件
        if success:
            cv2.imshow("framw",img)
            k=cv2.waitKey(100000)

        if k==ord("s") or k==ord('S'):
            cv2.imwrite('test.jpg',img)
            print('以儲存')

        if k == ord("q") or k == ord('Q'):
            print('離開')
            cv2.destroyAllWindows()
            capture.release() #關閉攝影機
            break
else:
    print('關閉攝影機失敗')
##讀取攝像機所拍畫面
import cv2
from PIL import Image
picpath=r"/Users/dongguan-cheng/PycharmProjects/pythonProject/haarcascade_frontalface_default.xml"
face_cascade=cv2.CascadeClassifier(picpath)
cv2.namedWindow("photo")
cap=cv2.VideoCapture(0)
while(cap.isOpened()):
    success, img = cap.read()  # 讀取攝影機所拍影像_success為布林值，img影像物件
    cv2.imshow("photo",img)
    if success==True:
        key=cv2.waitKey(20000)
        if key == ord("s") or k == ord('S'):
            cv2.imwrite('photo.jpg', img)
            print('以儲存圖片')
            break
cap.release() #關閉攝影機
faces=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=3,minSize=(20,20))

#標註右下角底色是黃色
cv2.rectangle(img,(img.shape[1]-113,img.shape[0]-20),(img.shape[1],img.shape[0]),(0,255,255),-1)
#標注找到多少的人臉
cv2.putText(img,"Finding"+str(len(faces))+"face",(img.shape[1]-113,img.shape[0]-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #用藍色框先框住臉
    image=Image.open("photo.jpg") #PIL模組開啟
    imageCrop=image.crop((x,y,x+w,y+h)) #進行裁切
    imageResize=imageCrop.resize((150,150),Image.ANTIALIAS) #高品質重置圖片大小
    imageResize.save("faceout.jpg") #儲存大小

cv2.namedWindow("face_recognize",cv2.WINDOW_NORMAL)
cv2.imshow("face_recognize",img)

## 臉型比對_利用RMS值兩圖一樣RMS=0,值越大代表圖差異越大
from functools import reduce
from PIL import Image
import math, operator
h1 =Image.open("face1.jpg").histogram()
h2=Image.open("face14.jpg").histogram()
RMS=math.sqrt(reduce(operator.add,list(map(lambda a,b:(a-b)**2,h1,h2)))/len(h1))
print("RMS=", RMS)
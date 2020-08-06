#cv2 to read out frames
import cv2
#urllib to make http request to flask server
import urllib.request
import numpy as np

# get request to webcam stream and saving response to 'stream'
stream = urllib.request.urlopen('http://localhost:8080/video_feed')
bytes = bytes()
while True:
    # reads 1024 bytes from stream response
    bytes += stream.read(1024)
    # get the start and end frame binary code \xff is start of jpeg image marker (https://docs.fileformat.com/image/jpeg/)
    a = bytes.find(b'\xff\xd8')
    #frame end binary code
    b = bytes.find(b'\xff\xd9')


    if a != -1 and b != -1:
        #save frames in jpeg bytearray
        jpg = bytes[a:b+2] # add offset of 2 because end range not included
        bytes = bytes[b+2:]
        #read out bytearray with cv2 as jpg into 'i' and display color image using imshow
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

##### CODE FOR SSPE ######


#use i as jpeg input for SSPE 


#########################
        cv2.imshow('i', i)
        #display a frame every 1ms  
        #cv2.waitKey(1)
        #include button to exit app 27 == esc (http://www.asciitable.com/)
        if cv2.waitKey(1) == 27:
            exit(0)  
        
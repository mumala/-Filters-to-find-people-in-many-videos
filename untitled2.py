import threading
import face_recognition
import cv2
import os
import numpy as np
import time

sem = threading.Semaphore(3)
   
def execute(number):
    sem.acquire()
  
    videofilename = videofiles[number]
    videoname, videoext = os.path.splitext(videofilename)
    
    if videoext == '.mp4' or '.mpeg' or '.avi' or '.mpg' or '.wmv':
    
        videopathname = os.path.join(videodirname, videofilename)
    
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

    #여기부터는 동영상에서 찾는 얼굴데이터에 대한 정보를 저장할것임
    #위의 4개 변수는 이를 위한 변수
        
        cap = cv2.VideoCapture(videopathname)

        fps = int(cap.get(cv2.CAP_PROP_FPS))            #fps계산
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        count = 0

        while True:
            ret, frame = cap.read()
            
            if (cap.get(cv2.CAP_PROP_POS_FRAMES)==cap.get(cv2.CAP_PROP_FRAME_COUNT)): break
 
            if(int(cap.get(1)) % fps == 1 or int(cap.get(1)) % fps == 2 or int(cap.get(1)) % fps == 3):
                count+=1
                      
                if height<360:
                    small_frame = frame
                elif 360<=height<720 :
                    small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)
                elif 720<=height<1080:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                else :
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                if process_this_frame:
                    
                    rgb_small_frame = small_frame[:, :, ::-1]
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    if len(face_locations) == 0 : continue
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                        #빼낸 프레임에서 얼굴영역, 얼굴 특징을 추출, 저장
                   
                
                    face_names = []
                        
                    for face_encoding in face_encodings:
                        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                                #동영상에서 추출한 얼굴과 아까 디렉토리에서 추출한 얼굴의 distance(유사도) 저장
                        min_value = min(distances)
                            
                        name = "Unknown"
                        if min_value < 0.45:                #유사도에 따라서 인식할지 안할지 결정
                            index = np.argmin(distances)
                            name = known_face_names[index]

                            face_names.append(name)
                            sec = count/3
                            minute, sec = divmod(sec,60)
                            hour, minute = divmod(minute,60)
                            cv2.imwrite("C:\pictureresult26/%s_%dh%02dm%02ds.jpg" %(videofilename,hour,minute,sec), frame)         #인식한 프레임 저장       
                         
                process_this_frame = not process_this_frame         
        cap.release()     
    
    cv2.destroyAllWindows()
    print('finish')         
    print(threading.currentThread().getName(), number)
    sem.release()
    
if __name__ == '__main__':
    start = time.time()
    known_face_encodings = []
    known_face_names = []
     
    dirname = 'knowns'
    files = os.listdir(dirname)
    for filename in files:
        name, ext = os.path.splitext(filename)
            
        if ext == '.jpg':
            known_face_names.append(name) 
            pathname = os.path.join(dirname, filename)
            img = face_recognition.load_image_file(pathname)
            face_encoding = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(face_encoding)

    videodirname = 'videodir'
    videofiles = os.listdir(videodirname)
    
    length = 0
    count = 0
    threads = []
    
    for length in range(len(videofiles)):
        my_thread = threading.Thread(target = execute, args=(length,))
        threads.append(my_thread)

    for th in threads :
        th.start()
    for th in threads :
        th.join()
    end = time.time()
    print(str(end-start))

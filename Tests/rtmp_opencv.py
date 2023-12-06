# import the opencv library 
import cv2 
  
  
# define a video capture object 
rmtp = "rtmp://127.0.0.1/live/SyWPfOiBa"
vid = cv2.VideoCapture(rmtp) 
  
while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 
  
    # Display the resulting frame 
    if ret:
        cv2.imshow('frame', frame)
        print(frame.shape)
    else: break
      
    # the 'q' button is set as the 
    # quitting button you may use any 
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
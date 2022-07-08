import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import FDNet

# Loading the fall detection model
model = FDNet()
model.load_state_dict(torch.load('./train_model/fdnet.pt'))
model.eval()
classes_dict = {0:'fall', 1:'no fall'}

#parameters for motion history image
MHI_DURATION = 30
DEFAULT_THRESHOLD = 32

# define a video capture object
vid = cv2.VideoCapture(0)
ret, frame = vid.read()
h, w = frame.shape[:2]
prev_frame = frame.copy()
motion_history = np.zeros((480,640), np.float32)
timestamp = 0
count = 0

# Start reading frames from the webcam
while(True):
    ret, frame = vid.read()
    frame_diff = cv2.absdiff(frame, prev_frame)
    gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)
    timestamp += 1
    # update motion history
    cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)
    # normalize motion history
    mh = np.uint8(np.clip((motion_history-(timestamp-MHI_DURATION)) / MHI_DURATION, 0, 1)*255)
    # Prepare input to the pytorch model
    temp_mh = cv2.cvtColor(mh,cv2.COLOR_GRAY2RGB)
    temp_mh = cv2.resize(temp_mh,(224, 224))
    temp_mh = torch.FloatTensor(temp_mh)
    temp_mh = temp_mh.reshape((1,3,224,224))
    # Get model prediction
    outputs = model(temp_mh)
    output = torch.max(F.softmax(outputs, dim=1), dim=1)[1]
    label_name = classes_dict[int(output)]
    print(label_name)
    # Inserting output label text on video
    cv2.putText(frame, label_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_4)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    prev_frame = frame.copy()
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
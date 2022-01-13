from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from PIL import Image
import color_uti as cu
import subfuncs
import os

class yolov5():
    def __init__(self, yolo_type, confThreshold=0.5, nmsThreshold=0.5, objThreshold=0.5):
        anchors = [[4,5,  8,10,  13,16], [23,29,  43,55,  73,105], [146,217,  231,300,  335,433]]
        num_classes = 1
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.no = num_classes + 5 + 10
        self.grid = [np.zeros(1)] * self.nl
        self.stride = np.array([8., 16., 32.])
        self.anchor_grid = np.asarray(anchors, dtype=np.float32).reshape(self.nl, -1, 2)
        self.inpWidth = 640
        self.inpHeight = 640
        self.net = cv2.dnn.readNet(yolo_type+'-face.onnx')
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.objThreshold = objThreshold
        self.face_counter = 0
        
    def _make_grid(self, nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((-1, 2)).astype(np.float32)

    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        ratioh, ratiow = frameHeight / self.inpHeight, frameWidth / self.inpWidth
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.

        confidences = []
        boxes = []
        landmarks = []
        for detection in outs:
            confidence = detection[15]
            # if confidence > self.confThreshold and detection[4] > self.objThreshold:
            if detection[4] > self.objThreshold:
                center_x = int(detection[0] * ratiow)
                center_y = int(detection[1] * ratioh)
                width = int(detection[2] * ratiow)
                height = int(detection[3] * ratioh)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)

                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                landmark = detection[5:15] * np.tile(np.float32([ratiow,ratioh]), 5)
                landmarks.append(landmark.astype(np.int32))
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            landmark = landmarks[i]
            frame = self.drawPred(frame, confidences[i], left, top, left + width, top + height, landmark)
        self.face_counter = 0
        return frame
        
    def drawPred(self, frame, conf, left, top, right, bottom, landmark):
        ## 
        try:
            ## add croped roi calculation
            roi = frame[top:bottom, left:right, :]
            ## stats
            #roi_sRGB, roi_Lab, roi_LCH, roi_sRGB_std, roi_sRGB_entropy = subfuncs.cal_stats(roi)
            #print('roi_sRGB:', roi_sRGB)
            roi_sRGB = np.array([np.mean(roi[:,:,2]), np.mean(roi[:,:,1]), np.mean(roi[:,:,0])])
            roi_Lab = cu.sRGB2Lab(roi_sRGB)
            print('roi_sRGB_np:', roi_sRGB )
            print('roi_Lab_np:', roi_Lab )
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (int(roi.shape[1]/10), int(roi.shape[0]/4*3))
            fontScale = roi.shape[1] / 100.0
            color = (0, 0, 255)
            thickness = int(roi.shape[1]/30)
            roi = cv2.putText(roi, '' "{:.2f}".format(roi_Lab[0]), org, font, fontScale, color, thickness, cv2.LINE_AA)
            roi = cv2.putText(roi, 'L*', (org[0], int(org[1]- roi.shape[0] / 2)), font, fontScale, color, thickness, cv2.LINE_AA)
            #cv2.namedWindow('face rect roi' + str(self.face_counter), 0)
            #cv2.imshow('face rect roi' + str(self.face_counter), roi)
            self.face_counter = self.face_counter + 1
        except:
            pass
        
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        # label = '%.2f' % conf
        # Display the label at the top of the bounding box
        # labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # top = max(top, labelSize[1])
        # cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        for i in range(5):
            cv2.circle(frame, (landmark[i*2], landmark[i*2+1]), 1, (0,255,0), thickness=-1)
        return frame
        
    def detect(self, srcimg):
        blob = cv2.dnn.blobFromImage(srcimg, 1 / 255.0, (self.inpWidth, self.inpHeight), [0, 0, 0], swapRB=True, crop=False)
        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())[0]

        # inference output
        outs[..., [0,1,2,3,4,15]] = 1 / (1 + np.exp(-outs[..., [0,1,2,3,4,15]]))   ###sigmoid
        row_ind = 0
        for i in range(self.nl):
            h, w = int(self.inpHeight/self.stride[i]), int(self.inpWidth/self.stride[i])
            length = int(self.na * h * w)
            if self.grid[i].shape[2:4] != (h,w):
                self.grid[i] = self._make_grid(w, h)
            
            g_i = np.tile(self.grid[i], (self.na, 1))
            a_g_i = np.repeat(self.anchor_grid[i], h * w, axis=0)
            outs[row_ind:row_ind + length, 0:2] = (outs[row_ind:row_ind + length, 0:2] * 2. - 0.5 + g_i) * int(self.stride[i])
            outs[row_ind:row_ind + length, 2:4] = (outs[row_ind:row_ind + length, 2:4] * 2) ** 2 * a_g_i

            outs[row_ind:row_ind + length, 5:7] = outs[row_ind:row_ind + length, 5:7] * a_g_i + g_i * int(self.stride[i])   # landmark x1 y1
            outs[row_ind:row_ind + length, 7:9] = outs[row_ind:row_ind + length, 7:9] * a_g_i + g_i * int(self.stride[i])  # landmark x2 y2
            outs[row_ind:row_ind + length, 9:11] = outs[row_ind:row_ind + length, 9:11] * a_g_i + g_i * int(self.stride[i])  # landmark x3 y3
            outs[row_ind:row_ind + length, 11:13] = outs[row_ind:row_ind + length, 11:13] * a_g_i + g_i * int(self.stride[i])  # landmark x4 y4
            outs[row_ind:row_ind + length, 13:15] = outs[row_ind:row_ind + length, 13:15] * a_g_i + g_i * int(self.stride[i])  # landmark x5 y5
            row_ind += length
        return outs

def get_score(img):

    #
    yolonet = yolov5(yolo_type='yolov5m', confThreshold = 0.3, nmsThreshold = 0.5, objThreshold = 0.3)
    # Display the resulting frame
    
    img = Image.open(img)
    img = np.array(img)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    dets = yolonet.detect(img)
    frame = yolonet.postprocess(img, dets)
    
    return frame
    #print('det: ', len(dets))
    #cv2.namedWindow('face rect', 0)
    #cv2.imshow('face rect', frame)
    
            
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def flask_main():
    if request.method == 'GET':
        return render_template('index.html', value='hi')
    if request.method == 'POST':

        # try to delete old result file
        try:
            files = os.listdir( os.path.join(os.path.dirname(__file__), \
                                             'static/' ) )
            for file in files:
                if 'results' in file and file.endswith('.png'):
                    os.remove( os.path.join(os.path.dirname(__file__), \
                                             'static/', \
                                             file ))
        except:
            pass
        file = request.files['file']
        result = get_score(file)
        new_graph_name = "results" + str(time.time()) + ".png"
        cv2.imwrite('static/' + new_graph_name, result)
   
        return render_template('result.html', r_cal = '1', results=new_graph_name)
        



if __name__ == '__main__':
    app.run(debug=True)

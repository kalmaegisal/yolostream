from django.http import StreamingHttpResponse
from django.shortcuts import render
import torch
import cv2
import numpy as np
import os
import sys

# YOLOv5 모델 로드
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'stream', 'yolo'))
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_boxes
from utils.dataloaders import letterbox

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load('yolov5s.pt')
model.eval()
names = model.module.names if hasattr(model, 'module') else model.names

camera = cv2.VideoCapture(0)  # 웹캠 사용


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # YOLOv5 전처리
            img = letterbox(frame, new_shape=640)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # 객체 탐지
            pred = model(img, augment=False)[0]
            pred = non_max_suppression(pred, 0.4, 0.5)

            # 탐지된 객체 표시
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{names[int(cls)]} {conf:.2f}'
                        cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                        cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def index(request):
    return render(request, 'stream/index.html')

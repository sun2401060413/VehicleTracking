#!/bin/bash
# Download common models

activate CV
python download_weights.py
# python -c "
# import sys
# sys.path.append(r‘E:/Project/CV/yolov5-master’)
# from utils.google_utils import *;
# attempt_download('weights/yolov5s.pt');
# attempt_download('weights/yolov5m.pt');
# attempt_download('weights/yolov5l.pt');
# attempt_download('weights/yolov5x.pt')
# "

'''
VideoMaker.
Written by sunzhu, 2019-04-02, version 1.0
'''
import cv2
import os
import glob

img_root = r"E:\Project\CV\trajectory\VehicleTracking\results\sct\004"
video_root = r"E:\Project\CV\trajectory\VehicleTracking\results\sct\004"

def VideoMaker(input_root, output_root, height=1080, width=1920, fps=30):
    # Four-Character Codes
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    path_split = img_root.split('\\')
    videoWriter = cv2.VideoWriter(os.path.join(output_root, path_split[-1]+'_trace.mp4'), fourcc, fps, (width, height))
    os.chdir(input_root)
    filelist = glob.glob("*.jpg")
    filelist.sort(key=lambda x: int(x[:-4]))
    for elem in filelist:
        img = cv2.imread(os.path.join(input_root, elem))
        if img is not None:
            img = cv2.resize(img, (width, height))
            videoWriter.write(img) 
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)
            cv2.waitKey(1)
    videoWriter.release()

if __name__=="__main__":
    VideoMaker(img_root, video_root, fps=25)
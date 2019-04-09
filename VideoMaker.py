'''
VideoMaker.
Written by sunzhu, 2019-04-02, version 1.0
'''
import cv2
import os
import glob

img_root = r"D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data\resutls\loc_predicate_multi\12f"
video_root = r"D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data\resutls\loc_predicate_multi\12f"

def VideoMaker(input_root,output_root,height=1080,width=1920,fps=5):
    # Four-Character Codes
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter(os.path.join(output_root,'trace.mp4'), fourcc, fps, (width,height))
    os.chdir(input_root)
    filelist = glob.glob("*.jpg")
    filelist.sort(key= lambda x:int(x[:-4]))
    for elem in filelist:
        img = cv2.imread(os.path.join(input_root,elem))
        if img is not None:
            img = cv2.resize(img,(width,height))
            videoWriter.write(img) 
            cv2.namedWindow('img',cv2.WINDOW_NORMAL)
            cv2.imshow('img',img)
            cv2.waitKey(1)
    videoWriter.release()

if __name__=="__main__":
    VideoMaker(img_root,video_root,)
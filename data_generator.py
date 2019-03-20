'''
Data generator for triplet loss method.
crop objects from original image
written by sunzhu, 2019-03-19, version 1.0
'''

import os,sys
import pandas as pd
import cv2
import json

from Single_camera_track import IOU_tracker

# root path
dataset_root = r"E:\DataSet\trajectory\concatVD"

# save root
save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data'

device_info = [
    ['wuqi1B','wuqiyuce1.csv'],
    ['wuqi2B','wuqiyuce2.csv'],
    ['wuqi3B','wuqiyuce3.csv'],
    ['wuqi4B','wuqiyuce4.csv'],
]

def load_tracking_info(savepath):
    tracking_json = os.path.join(savepath,'tracking_info.json')
    with open(tracking_json,'r') as doc:
        data = json.load(doc)
    return data

def get_tracking_info(device_id):
    img_filepath = os.path.join(dataset_root,device_info[device_id][0]) 
    img_savepath = os.path.join(save_root,device_info[device_id][0])
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)
    tracker = IOU_tracker()
    data = pd.read_csv(os.path.join(dataset_root,device_info[device_id][1]))
    prev_fps = -1
    id_in_frame = 0
    for index, row in data[["x1", "y1", "x2", "y2","fps"]].iterrows():
        # print(row['x1'],row['y1'],row['x2'],row['y2'],row['fps'])
        tracker.update([row['x1'],row['y1'],row['x2'],row['y2'],row['fps']])
        # filename
        filename = str(row['fps'])+'.jpg'
        img_filename = os.path.join(img_filepath,filename)
        img = cv2.imread(img_filename)
        tracker.draw_trajectory(img)
    tracker.save_data(img_savepath)
    
def get_crop_img(device_id):
    '''Get and save images in box with same object id
        input: device_id: camera id
    '''
    img_filepath = os.path.join(dataset_root,device_info[device_id][0]) 
    img_savepath = os.path.join(save_root,device_info[device_id][0])
    tracking_info = load_tracking_info(img_savepath)
    crop_img_info_json = 'crop_img_info.json'
    crop_img_info = {}
    for elem in tracking_info:
        obj_id = tracking_info[elem]['id']
        trace = tracking_info[elem]['list']
        crop_img_list = []
        for box_info in trace:
            frame = box_info[1]
            box = box_info[0]
            filename = str(frame)+'.jpg'
            img_filename = os.path.join(img_filepath,filename)
            cv2.namedWindow('img',cv2.WINDOW_NORMAL)
            img = cv2.imread(img_filename)
            crop_img = img[box[1]:box[3],box[0]:box[2]]
            save_img_filename = os.path.join(img_savepath,str(frame)+'_'+str(obj_id)+'.jpg')
            crop_img_list.append(save_img_filename)
            print(save_img_filename)
            cv2.imwrite(save_img_filename,crop_img)
            cv2.imshow('img',crop_img)
            cv2.waitKey(1)
        crop_img_info[elem] = crop_img_list
    print(crop_img_info)
    with open(os.path.join(img_savepath,crop_img_info_json),'w') as doc:
        json.dump(crop_img_info,doc)
    
if __name__=="__main__":
    device_id = 0
    # ----- tracker test -----
    # get_tracking_info(device_id)
    
    # ------ crop img generate -------
    # get_crop_img(device_id)
    print("Mission accomplished!")
    # tracking_info = load_tracking_info(img_savepath)
    # crop_img_info_json = 'crop_img_info.json'
    # crop_img_info = {}
    # for elem in tracking_info:
        # obj_id = tracking_info[elem]['id']
        # trace = tracking_info[elem]['list']
        # crop_img_list = []
        # for box_info in trace:
            # frame = box_info[1]
            # box = box_info[0]
            # filename = str(frame)+'.jpg'
            # img_filename = os.path.join(img_filepath,filename)
            # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
            # img = cv2.imread(img_filename)
            # crop_img = img[box[1]:box[3],box[0]:box[2]]
            # save_img_filename = os.path.join(img_savepath,str(frame)+'_'+str(obj_id)+'.jpg')
            # crop_img_list.append(save_img_filename)
            # print(save_img_filename)
            # cv2.imwrite(save_img_filename,crop_img)
            # cv2.imshow('img',crop_img)
            # cv2.waitKey(1)
        # crop_img_info[elem] = crop_img_list
    # with open(os.path.join(img_savepath,crop_img_info_json),'w') as doc:
        # json.dump(crop_img_info,doc)
        
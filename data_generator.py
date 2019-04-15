'''
Data generator for triplet loss method.
crop objects from original image
written by sunzhu, 2019-03-19, version 1.0
'''

import os,sys
import pandas as pd
import numpy as np
import cv2
import json

from Single_camera_track import IOU_tracker

# images root path
dataset_root = r"E:\DataSet\trajectory\concatVD"
# tracking info path
tracking_root = r"D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data"
# perspective transformer root path
pt_trans_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\ROItools\data'
# save root
crop_image_save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data'

device_info = [
    ['wuqi1B','wuqiyuce1.csv','ROI_cam_1_transformer.json'],
    ['wuqi2B','wuqiyuce2.csv','ROI_cam_2_transformer.json'],
    ['wuqi3B','wuqiyuce3.csv','ROI_cam_3_transformer.json'],
    ['wuqi4B','wuqiyuce4.csv','ROI_cam_4_transformer.json'],
]
# # ===== CLASS =====
class files_info(object):
    def __init__(self,
                dataset_root,
                tracking_root,
                pt_trans_root,
                device_info):
        self.dataset_root = dataset_root
        self.tracking_root = tracking_root
        self.pt_trans_root = pt_trans_root
        self.device_info = device_info
    
    def get_filepath(self,id):
        output_dict = {}
        output_dict['img_filepath'] = os.path.join(self.dataset_root,self.device_info[id][0])
        output_dict['box_info_filepath'] = os.path.join(self.dataset_root,self.device_info[id][1])
        output_dict['tracking_info_filepath'] = os.path.join(self.tracking_root,self.device_info[id][0])
        output_dict['pt_savepath'] = os.path.join(self.pt_trans_root,self.device_info[id][2])
        return output_dict

class data_generator(object):
    def __init__(self,
                csv_filename,
                image_fileroot):
        
        self.csv_filename = csv_filename
        self.image_fileroot = image_fileroot

    def data_gen(self,time_interval=1):
        data = pd.read_csv(self.csv_filename)
        unique_id = np.unique(data['fps'])
        for v,elem in enumerate(unique_id):
            if v%time_interval == 0:
                img_data = cv2.imread(os.path.join(self.image_fileroot,str(elem)+'.jpg'))
                frame_data = get_frame_data(data,elem)
                yield img_data,frame_data
        return False

class two_cameras_simulator(object):
    def __init__(self,
                csv_filename_1,
                csv_filename_2,
                image_fileroot_1,
                image_fileroot_2):
        self.csv_filename_1 = csv_filename_1
        self.csv_filename_2 = csv_filename_2
        self.image_fileroot_1 = image_fileroot_1
        self.image_fileroot_2 = image_fileroot_2
        
    def data_gen(self,time_interval=1,shape=(1080,1920,3)):
        data_1 = pd.read_csv(self.csv_filename_1)
        data_2 = pd.read_csv(self.csv_filename_2)
        
        filelist_1 = os.listdir(self.image_fileroot_1)
        filelist_2 = os.listdir(self.image_fileroot_2)
        filelist_1.sort(key = lambda x:int(x[:-4]))
        filelist_2.sort(key = lambda x:int(x[:-4]))
        
        total = max(len(filelist_1),len(filelist_2))
        
        
        for i in range(0,total):
            if i%time_interval==0:
                filename_1 = os.path.join(self.image_fileroot_1,"{}.jpg".format(i))
                filename_2 = os.path.join(self.image_fileroot_2,"{}.jpg".format(i))
                if os.path.exists(filename_1):
                    img_1 = cv2.imread(filename_1)
                else:
                    img_1 = np.zeros(shape,np.uint8)
                if os.path.exists(filename_2):
                    img_2 = cv2.imread(filename_2)
                else:
                    img_2 = np.zeros(shape,np.uint8)
                    
                frame_data_1 = get_frame_data(data_1,i)
                frame_data_2 = get_frame_data(data_2,i)
                
                yield img_1,img_2,frame_data_1,frame_data_2
        # print(data_1[data_1['fps']==-1].empty)
        '''
        # unique_id_1 = np.unique(data_1['fps'])
        # unique_id_2 = np.unique(data_2['fps'])
        
        # shape_1 = cv2.imread(os.path.join(self.image_fileroot_1,str(unique_id_1[0])+'.jpg')).shape
        # shape_2 = cv2.imread(os.path.join(self.image_fileroot_2,str(unique_id_2[0])+'.jpg')).shape
        
        # for v,elem in enumerate(np.unique(np.append(unique_id_1,unique_id_2,axis=0))):
            # if v%time_interval == 0:
                # # image 1 
                # if elem in unique_id_1:
                    # img_1 = cv2.imread(os.path.join(self.image_fileroot_1,str(elem)+'.jpg'))
                # else:
                    # img_1 = np.zeros(shape_1)
                    
                # # image 2
                # if elem in unique_id_2:
                    # img_2 = cv2.imread(os.path.join(self.image_fileroot_2,str(elem)+'.jpg'))
                # else:
                    # img_2 = np.zeros(shape_1)
                    
                # frame_data_1 = get_frame_data(data_1,elem)
                # frame_data_2 = get_frame_data(data_2,elem)
                
                # yield img_1,img_2,frame_data_1,frame_data_2
        '''

    
# # ===== UTILS FUNCTIONS =====  
def get_files_info():
    obj = files_info(dataset_root,
                    tracking_root,
                    pt_trans_root,
                    device_info)
    return obj

def get_frame_data(data,frame_id):
    slice_data = data[data['fps'] == frame_id]
    if slice_data.empty:
        frame_data = None
    else:
        frame_data = list(zip(*[slice_data['x1'],slice_data['y1'],slice_data['x2'],slice_data['y2'],slice_data['fps']]))
    return frame_data   # output tuple
    
def load_tracking_info(savepath):
    tracking_json = os.path.join(savepath,'tracking_info.json')
    with open(tracking_json,'r') as doc:
        data = json.load(doc)
    return data

def get_tracking_info(device_id):
    img_filepath = os.path.join(dataset_root,device_info[device_id][0]) 
    img_savepath = os.path.join(crop_image_save_root,device_info[device_id][0])
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
        cv2.namedWindow('img',cv2.WINDOW_NORMAL)
        cv2.imshow('img',tracker.draw_trajectory(img))
        cv2.waitKey(1)
    tracker.save_data(img_savepath)
    
def get_crop_img(device_id):
    '''Get and save images in box with same object id
        input: device_id: camera id
    '''
    img_filepath = os.path.join(dataset_root,device_info[device_id][0]) 
    img_savepath = os.path.join(crop_image_save_root,device_info[device_id][0])
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
# # ===== TEST FUNCTIONS =====
def crop_img_generator():
    device_id = 0
    get_crop_img(device_id)
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
    with open(os.path.join(img_savepath,crop_img_info_json),'w') as doc:
        json.dump(crop_img_info,doc)
    
def files_info_test():
    obj = get_files_info()
    print("====Get in data_generator_test! ====")
    print(obj.dataset_root)
    print(obj.tracking_root)
    print(obj.pt_trans_root)
    print(obj.device_info)
    print(obj.get_filepath(0))
    print(obj.get_filepath(1))
    return

def data_generator_test():
    files_gen = get_files_info()
    obj = data_generator(
            files_gen.get_filepath(0)['box_info_filepath'],
            files_gen.get_filepath(0)['img_filepath'])
    datagen = obj.data_gen(time_interval=1)
    try:
        while(True):
            img,d = datagen.__next__()
            cv2.namedWindow('img',cv2.WINDOW_NORMAL)
            cv2.imshow('img',img)
            for elem in d:
                print(elem)
            cv2.waitKey(1)
    except StopIteration:
        pass
    return
    
def two_cameras_simulator_test():
    files_gen = get_files_info()
    obj_sim_gen = two_cameras_simulator(
                    files_gen.get_filepath(0)['box_info_filepath'],
                    files_gen.get_filepath(1)['box_info_filepath'],
                    files_gen.get_filepath(0)['img_filepath'],
                    files_gen.get_filepath(1)['img_filepath'])
    func = obj_sim_gen.data_gen(time_interval=5)
    try:
        while(True):
            img_1,img_2,frame_data_1,frame_data_2 = func.__next__()
            cv2.namedWindow("img_1",cv2.WINDOW_NORMAL)
            cv2.namedWindow("img_2",cv2.WINDOW_NORMAL)
            cv2.imshow("img_1",img_1)
            cv2.imshow("img_2",img_2)
            print("-"*25)
            if frame_data_1 is not None:
                for elem in frame_data_1:
                    print("CAM_1:",elem)
            if frame_data_2 is not None:
                for elem in frame_data_2:
                    print("CAM_2:",elem)
            cv2.waitKey(1)
    except StopIteration:
        pass
    return
    
if __name__=="__main__":
    # files_info_test()
    # data_generator_test()
    two_cameras_simulator_test()
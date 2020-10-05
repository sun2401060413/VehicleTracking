'''
Data generator for triplet loss method.
crop objects from original image
written by sunzhu, 2019-03-19, version 1.0
Updated by sunzhu on Sep 29, 2020, version 1.1
'''

import os
import sys
import pandas as pd
import numpy as np
import cv2
import json


from Common import SRC_IMAGES, DET_RESULT, ROI_RESULT, SCT_RESULT, MCT_RESULT

# ==== Files path setting, you could comment the code line below and set your own path ====
from Common import cam_names, data_path, box_info, roi_info, track_info, save_path


# # ===== CLASS =====


class FilesInfo(object):
    """
    Files setting: set the include image path, bounding box info path, tracking info path, pt_trans info path
    """
    def __init__(self,
                dataset_root=None,      # image root path
                boxinfo_root=None,      # bound box info root path
                tracking_root=None,     # tracking info root path
                pt_trans_root=None      # pt_trans info root path
                 ):
        self.dataset_root = dataset_root
        self.boxinfo_root = boxinfo_root
        self.tracking_root = tracking_root
        self.pt_trans_root = pt_trans_root
    
    def get_filepath(self, device_id):
        """Create specific path """
        output_dict = {}
        output_dict['img_filepath'] = self.dataset_root[device_id]
        output_dict['box_info_filepath'] = self.boxinfo_root[device_id]
        output_dict['tracking_info_filepath'] = self.tracking_root[device_id]
        output_dict['pt_savepath'] = self.pt_trans_root[device_id]
        return output_dict


class DataGenerator(object):
    """
    Data generator
    csv_filename: bbox info path
    image_fileroot: src image path
    """
    def __init__(self,
                csv_filename=None,
                image_fileroot=None):
        self.csv_filename = csv_filename
        self.image_fileroot = image_fileroot
        self.data = pd.read_csv(self.csv_filename)
        self.unique_id = np.unique(self.data['fps'])

    def data_gen(self, time_interval=1):
        """Simulate detection output"""
        for v, elem in enumerate(self.unique_id):
            if v % time_interval == 0:
                img_data = cv2.imread(os.path.join(self.image_fileroot, '{:0>4d}.jpg'.format(elem)))
                frame_data = get_frame_data(self.data, elem)
                yield img_data, frame_data  # image, detected objs
        return False


class TwoCamerasSimulator(object):
    """
    input data like decoded video stream
    """
    def __init__(self,
                csv_filename_1=None,
                csv_filename_2=None,
                image_fileroot_1=None,
                image_fileroot_2=None):
        self.csv_filename_1 = csv_filename_1
        self.csv_filename_2 = csv_filename_2
        self.image_fileroot_1 = image_fileroot_1
        self.image_fileroot_2 = image_fileroot_2
        
    def data_gen(self, time_interval=1, shape=(1080, 1920, 3)):
        data_1 = pd.read_csv(self.csv_filename_1)
        data_2 = pd.read_csv(self.csv_filename_2)
        
        filelist_1 = os.listdir(self.image_fileroot_1)
        filelist_2 = os.listdir(self.image_fileroot_2)
        filelist_1.sort(key=lambda x: int(x[:-4]))
        filelist_2.sort(key=lambda x: int(x[:-4]))
        
        total = max(len(filelist_1), len(filelist_2))

        for i in range(0, total):
            if i % time_interval == 0:
                filename_1 = os.path.join(self.image_fileroot_1, "{}.jpg".format(i))
                filename_2 = os.path.join(self.image_fileroot_2, "{}.jpg".format(i))
                if os.path.exists(filename_1):
                    img_1 = cv2.imread(filename_1)
                else:
                    img_1 = np.zeros(shape, np.uint8)
                if os.path.exists(filename_2):
                    img_2 = cv2.imread(filename_2)
                else:
                    img_2 = np.zeros(shape, np.uint8)
                    
                frame_data_1 = get_frame_data(data_1, i)
                frame_data_2 = get_frame_data(data_2, i)
                
                yield img_1, img_2, frame_data_1, frame_data_2


# # ===== UTILS FUNCTIONS =====  
def get_files_info(dataset_root=data_path,
                   boxinfo_root=box_info,
                   tracking_root=track_info,
                   pt_trans_root=roi_info):
    obj = FilesInfo(dataset_root=dataset_root,
                    boxinfo_root=boxinfo_root,
                    tracking_root=tracking_root,
                    pt_trans_root=pt_trans_root)
    return obj

def get_frame_data(data, frame_id):
    slice_data = data[data['fps'] == frame_id]
    if slice_data.empty:
        frame_data = None
    else:
        frame_data = list(zip(*[slice_data['x1'], slice_data['y1'], slice_data['x2'], slice_data['y2'], slice_data['fps']]))
    return frame_data   # output tuple
    
def load_tracking_info(filepath):
    tracking_json = filepath
    with open(tracking_json, 'r') as doc:
        data = json.load(doc)
    return data

def get_tracking_info(device_id):
    from Single_camera_track import IOUTracker
    img_filepath = data_path[device_id]
    img_savepath = save_path[device_id]
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)
    tracker = IOUTracker()
    data = pd.read_csv(box_info[device_id])
    prev_fps = -1
    id_in_frame = 0
    for index, row in data[["x1", "y1", "x2", "y2", "fps"]].iterrows():
        # print(row['x1'],row['y1'],row['x2'],row['y2'],row['fps'])
        tracker.update([row['x1'], row['y1'], row['x2'], row['y2'], row['fps']])
        # filename
        filename = str(row['fps'])+'.jpg'
        img_filename = os.path.join(img_filepath, filename)
        img = cv2.imread(img_filename)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', tracker.draw_trajectory(img))
        cv2.waitKey(1)
    tracker.save_data(img_savepath)
    
def get_crop_img(device_id):
    '''Get and save images in box with same object id
        input: device_id: camera id
    '''
    img_filepath = data_path[device_id]
    img_savepath = save_path[device_id]
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
            img_filename = os.path.join(img_filepath, filename)
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            img = cv2.imread(img_filename)
            crop_img = img[box[1]:box[3], box[0]:box[2]]
            save_img_filename = os.path.join(img_savepath, str(frame)+'_'+str(obj_id)+'.jpg')
            crop_img_list.append(save_img_filename)
            print(save_img_filename)
            cv2.imwrite(save_img_filename, crop_img)
            cv2.imshow('img', crop_img)
            cv2.waitKey(1)
        crop_img_info[elem] = crop_img_list
    print(crop_img_info)
    with open(os.path.join(img_savepath, crop_img_info_json), 'w') as doc:
        json.dump(crop_img_info, doc)


# # ===== TEST FUNCTIONS =====


def crop_img_generator(device_id=0):
    get_crop_img(device_id)
    tracking_info = load_tracking_info(data_path[device_id])
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
            img_filename = os.path.join(data_path[device_id], filename)
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            img = cv2.imread(img_filename)
            crop_img = img[box[1]:box[3], box[0]:box[2]]
            save_img_filename = os.path.join(save_path[device_id], str(frame)+'_'+str(obj_id)+'.jpg')
            crop_img_list.append(save_img_filename)
            print(save_img_filename)
            cv2.imwrite(save_img_filename, crop_img)
            cv2.imshow('img', crop_img)
            cv2.waitKey(1)
        crop_img_info[elem] = crop_img_list
    with open(os.path.join(save_path[device_id], crop_img_info_json), 'w') as doc:
        json.dump(crop_img_info, doc)


def files_info_test():
    obj = get_files_info()
    print("====Get in data_generator_test! ====")
    print(obj.dataset_root)
    print(obj.tracking_root)
    print(obj.pt_trans_root)
    print(obj.get_filepath(0))
    print(obj.get_filepath(1))
    return


def data_generator_test():
    files_gen = get_files_info()
    obj = DataGenerator(
            files_gen.get_filepath(0)['box_info_filepath'],
            files_gen.get_filepath(0)['img_filepath'])
    datagen = obj.data_gen(time_interval=1)
    try:
        while True:
            img, d = datagen.__next__()
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)
            for elem in d:
                print(elem)
            cv2.waitKey(1)
    except StopIteration:
        pass
    return


def two_cameras_simulator_test():
    files_gen = get_files_info()
    str1 = files_gen.get_filepath(0)['box_info_filepath']
    str2 = files_gen.get_filepath(1)['box_info_filepath']
    str3 = files_gen.get_filepath(0)['img_filepath']
    str4 = files_gen.get_filepath(1)['img_filepath']
    obj_sim_gen = TwoCamerasSimulator(
                    files_gen.get_filepath(0)['box_info_filepath'],
                    files_gen.get_filepath(1)['box_info_filepath'],
                    files_gen.get_filepath(0)['img_filepath'],
                    files_gen.get_filepath(1)['img_filepath'])
    func = obj_sim_gen.data_gen(time_interval=5)
    try:
        while True:
            img_1, img_2, frame_data_1, frame_data_2 = func.__next__()
            cv2.namedWindow("img_1", cv2.WINDOW_NORMAL)
            cv2.namedWindow("img_2", cv2.WINDOW_NORMAL)
            cv2.imshow("img_1", img_1)
            cv2.imshow("img_2", img_2)
            print("-"*25)
            if frame_data_1 is not None:
                for elem in frame_data_1:
                    print("CAM_1:", elem)
            if frame_data_2 is not None:
                for elem in frame_data_2:
                    print("CAM_2:", elem)
            cv2.waitKey(1)
    except StopIteration:
        pass
    return


if __name__ == "__main__":
    # ==== Files_Info_test ====
    # files_info_test()

    # ==== Data generator test ====
    data_generator_test()

    # ==== Two cameras simulator test ====
    # two_cameras_simulator_test()

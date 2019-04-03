'''
MCT: Multi cameras tracking.
Multi-objects tracking in multi-cameras.
written by sunzhu, 2019-03-20, version 1.0
'''

import os,sys
sys.path.append(r'D:\Project\tensorflow_model\VehicleTracking\AIC2018_iamai\ReID\ReID_CNN')
import pandas as pd
import cv2
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import compute_VeRi_dis as dist
from Model_Wrapper import ResNet_Loader

# from Single_camera_track import STP_tracker


# # ===== UTILS FUNCTIONS =====     
def load_crop_img_list(save_path):
    json_file_path = os.path.join(save_path,'crop_img_info.json')
    with open(json_file_path,'r') as doc:
        data = json.load(doc)
    return data

# # ===== CLASS =====
class Cameras_Topology(object):
    '''Topology of Cameras'''
    def __init__(self,trackers_dict={},STP_dict={},associate_dict={}):
        '''
        trackers_dict: tracker with CAM_id
        STP_dict: spatial-temporal-prior between CAM_id and next_CAM_id 
        associate_dict: prev_CAM_id: CAM_id
        '''
        self.trackers_dict = trackers_dict
        self.STP_dict = STP_dict
        self.associate_dict = associate_dict
        
    def add_new_element(self,obj_tracker,obj_STP,id,prev_id=None):
        self.trackers_dict[id] = tracker
        if prev_id is not None:
            self.STP_dict[prev_id] = obj_STP
            self.associate_dict[prev_id] = id       # record next cam id
        self.trackers_dict[id] = obj_tracker
        
    def get_reverse_associate_dict(self):
        '''Find prev cam id'''
        return {self.associate_dict[elem]:elem for elem in self.associate_dict}

class MCT_STP_tracker(object):
    def __init__(self,
                frame_space_dist = 100,
                obj_cameras_topology = None,
                match_mode = 'Prob'):
        # time range
        self.frame_space_dist = frame_space_dist

        
        # Single camera tracker
        self.obj_STP_tracker = obj_STP_tracker
        # Multi cameras 
        self.obj_Multi_cameras_STP = obj_Multi_cameras_STP
        
        self.match_mode = match_mode
        
        # save vehicle object out from cam_1
        self.objects_pool = {}
        
        # threshold of tracking
        self.thresh_probability = 0.001
        self.thresh_distance = 5
        
        # image information
        self.img_height = 1080
        self.img_width = 1920
        self.obj_pool_display_height = 100
        self.obj_pool_display_width = 100
        self.obj_pool_display_channel = 3
        
        # # display setting
        # self.display_monitor_region = False
        
    def version(self):
        return print("===== Written by sunzhu, 2019-04-03, Version 1.0 =====")
        
    def get_available_id(self):
        pass
        
    def get_available_color(self):
        pass
        
    def isTrackFinish(self,frame):
        pass
        
    def update(self,box,img=None):
        pass
        
    def match(self,box,frame):
        pass

    def rank(self,objs_list):
        pass
    
    
# # ===== TEST FUNCTIONS =====
def Devices_Topology_test():
    from data_generator import get_files_info
    obj_data_generator = get_files_info()
    
    time_interval = 25
    # file path 1
    file_dict_1 = obj_data_generator.get_filepath(0)
    file_dict_2 = obj_data_generator.get_filepath(1)
        
    from data_generator import load_tracking_info
    tracker_record_1 = load_tracking_info(file_dict_1['tracking_info_filepath'])
    tracker_record_2 = load_tracking_info(file_dict_2['tracking_info_filepath'])
    
    from Perspective_transform import Perspective_transformer
    pt_obj_1 = Perspective_transformer(file_dict_1['pt_savepath'])
    pt_obj_2 = Perspective_transformer(file_dict_2['pt_savepath'])

    from cameras_associate import Single_camera_STP
    STP_Predictor_1 = Single_camera_STP(
                        tracker_record_1,
                        pt_obj_1,
                        time_interval = time_interval)
    STP_Predictor_1.var_beta_x = int(25*25/time_interval)
    STP_Predictor_1.var_beta_y = int(25/time_interval)+1
    STP_Predictor_1.update_predictor()            
                        
    STP_Predictor_2 = Single_camera_STP(
                        tracker_record_2,
                        pt_obj_2,
                        time_interval = time_interval)
    STP_Predictor_2.var_beta_x = int(25*25/time_interval)
    STP_Predictor_2.var_beta_y = int(25/time_interval)+1
    STP_Predictor_2.update_predictor()        
    
    from Single_camera_track import STP_tracker
    obj_tracker_1 = STP_tracker(frame_space_dist=50,obj_STP=STP_Predictor_1)
    obj_tracker_1.match_mode = 'Prob'
    obj_tracker_1.display_monitor_region = True
    obj_tracker_2 = STP_tracker(frame_space_dist=50,obj_STP=STP_Predictor_2)
    obj_tracker_2.region_top = 250
    obj_tracker_2.match_mode = 'Prob'
    obj_tracker_2.display_monitor_region = True
    
    from data_generator import two_cameras_simulator    # Just for test
    cameras_simulator = two_cameras_simulator(
                            file_dict_1['box_info_filepath'],
                            file_dict_2['box_info_filepath'],
                            file_dict_1['img_filepath'],
                            file_dict_2['img_filepath'])
    datagen = cameras_simulator.data_gen(time_interval=time_interval)
    try:
        while(True):
            img_1,img_2,d_1,d_2 = datagen.__next__()
            
            if d_1 is not None:
                for elem in d_1:
                    cp_img = img_1[elem[1]:elem[3],elem[0]:elem[2]].copy()
                    obj_tracker_1.update(elem,cp_img)
            if d_2 is not None:
                for elem in d_2:
                    cp_img = img_2[elem[1]:elem[3],elem[0]:elem[2]].copy()
                    obj_tracker_2.update(elem,cp_img)
            tray_img_1 = obj_tracker_1.draw_trajectory(img_1)
            tray_img_2 = obj_tracker_2.draw_trajectory(img_2)
            
            cv2.namedWindow('img_1',cv2.WINDOW_NORMAL)
            cv2.namedWindow('img_2',cv2.WINDOW_NORMAL)
            cv2.imshow('img_1',tray_img_1)
            cv2.imshow('img_2',tray_img_2)
            cv2.waitKey()
    except StopIteration:
        pass
    
    # obj_cameras_topology = Cameras_Topology()
    return

def MCT_STP_tracker_test():
    return

def SimiliarityCalculateTest():
    # root path
    dataset_root = r"E:\DataSet\trajectory\concatVD"

    # save root
    save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data'

    # images
    cam_1 = 'wuqi1B'
    cam_2 = 'wuqi2B'
    cam_3 = 'wuqi3B'
    cam_4 = 'wuqi4B'

    # bbox information
    box_info_1 = 'wuqiyuce1.csv'
    box_info_2 = 'wuqiyuce2.csv'
    box_info_3 = 'wuqiyuce3.csv'
    box_info_4 = 'wuqiyuce4.csv'

    img_savepath_1 = os.path.join(save_root,cam_1)
    img_savepath_2 = os.path.join(save_root,cam_2)
    
    load_ckpt = r"D:\Project\tensorflow_model\VehicleTracking\models\model_880_base.ckpt"
    n_layer = 50
    gallery_txt = r"E:\DataSet\BoxCars\Reid_dataset\BoxCars\BoxCars_train.txt"
    query_txt = r"E:\DataSet\BoxCars\Reid_dataset\BoxCars\BoxCars_train.txt"
    dis_mat = r"E:\DataSet\BoxCars\Reid_dataset\BoxCars"
    batch_size = 1
    print('loading model....')
    model = ResNet_Loader(load_ckpt,n_layer,output_color=False,batch_size=1)

    data_1 = load_crop_img_list(img_savepath_1)
    data_2 = load_crop_img_list(img_savepath_2)
    list_1 = [data_1['0'][2],data_1['1'][2],data_1['2'][2],data_1['3'][2],data_1['4'][2]]
    list_2 = [data_2['0'][2],data_2['1'][2],data_2['2'][2],data_2['3'][2],data_2['4'][2]]
    
    plt.figure("Image") #
    for i,elem in enumerate(list_1):
        name,_ = os.path.splitext(os.path.split(elem)[1])
        img = Image.open(os.path.join(elem))
        plt.subplot(2,5,i+1)
        plt.imshow(img)
        print(name)
        plt.title(name)
    for i,elem in enumerate(list_2):
        name,_ = os.path.splitext(os.path.split(elem)[1])
        img = Image.open(os.path.join(elem))
        plt.subplot(2,5,i+6)
        plt.imshow(img)
        print(name)
        plt.title(name)

    q_features = model.inference(list_1)
    g_features = model.inference(list_2)
            
    q_features = nn.functional.normalize(q_features,dim=1).cuda()
    g_features = nn.functional.normalize(g_features,dim=1).transpose(0,1).cuda()

    print('compute distance')
    SimMat = -1 * torch.mm(q_features,g_features)
    SimMat = SimMat.cpu().transpose(0,1)

    SimMat = SimMat.numpy()
    data1 = pd.DataFrame(SimMat)
    data1.to_csv(os.path.join(r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data','data1.csv'))
    
    plt.show()
    
def useless():
    # with open(args.query_txt,'r') as f:
        # # query_txt = [q.strip() for q in f.readlines()]
        # # query_txt = query_txt[1:]
        # chosen_text = [[q.strip().split(' ')[0],q.strip().split(' ')[1]] for q in f.readlines() if q.strip()!='']
        
        # query_txt,id_txt = zip(*chosen_text)
        # query_txt = query_txt[1:21]
        # id_txt = id_txt[1:21]
    # with open(args.gallery_txt,'r') as f:
        # # gallery_txt = [q.strip() for q in f.readlines()]
        # # gallery_txt = gallery_txt[1:]
        # gallery_txt = [q.strip().split(' ')[0] for q in f.readlines() if q.strip()!='']
        # gallery_txt = query_txt[1:21]
    
    # print(query_txt)
    # print(id_txt)
    
    # plt.figure("Image") #
    # for i,elem in enumerate(query_txt):
        # img = Image.open(os.path.join(elem))
        # plt.subplot(4,5,i+1)
        # plt.imshow(img)
        # print(id_txt[i])
        # plt.title(id_txt[i])
    # plt.show()
        
    # print('inferencing q_features')
    # q_features = model.inference(query_txt)
    # print('inferencing g_features')
    # g_features = model.inference(query_txt)
    
    # q_features = nn.functional.normalize(q_features,dim=1).cuda()
    # g_features = nn.functional.normalize(g_features,dim=1).transpose(0,1).cuda()
    
    # print('compute distance')
    # SimMat = -1 * torch.mm(q_features,g_features)
    # SimMat = SimMat.cpu().transpose(0,1)

    # print(SimMat.size())
    
    # SimMat = SimMat.numpy()
    # # import scipy.io as sio
    # # sio.savemat(args.dis_mat,{'dist_CNN':SimMat})
    
    # data1 = pd.DataFrame(SimMat)
    # data1.to_csv(os.path.join(r'E:\DataSet\BoxCars\Reid_dataset','data1.csv'))
    # # doc = open(os.path.join(r'E:\DataSet\BoxCars\Reid_dataset','boxcar.txt'),'w')
    # # print(SimMat,file=doc)
    pass


if __name__=="__main__":
    # # ===== TEST: Calculate Similiarity test =====
    # SimiliarityCalculateTest()
    # # ===== TEST:Devices_Topology_test =====
    Devices_Topology_test()
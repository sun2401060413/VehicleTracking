'''
MCT: Multi cameras tracking.
Multi-objects tracking in multi-cameras.
Written by sunzhu, 2019-03-20, version 1.0
'''

import os,sys
sys.path.append(r'D:\Project\tensorflow_model\VehicleTracking\AIC2018_iamai\ReID\ReID_CNN')
import pandas as pd
import numpy as np
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

def get_box_center(box,mode="both"):
    if mode == "both":
        return int((box[0]+box[2])/2),int((box[1]+box[3])/2)
    if mode == "bottom":
        return int((box[0]+box[2])/2),int(box[3])
# # ===== CLASS =====
class TrackersArray(object):
    '''Array of Cameras'''
    def __init__(self,trackers_dict={},multi_tracker_dict={},associate_dict={}):
        '''
        trackers_dict: tracker with CAM_id
        STP_dict: spatial-temporal-prior between CAM_id and next_CAM_id 
        associate_dict: prev_CAM_id: CAM_id
        '''
        self.trackers_dict = trackers_dict
        self.multi_tracker_dict = multi_tracker_dict
        self.associate_dict = associate_dict
        self.hist_record_dict = {}
        self.run_record_dict = {}
        self.object_append_status = {}
        self.object_append_dict = {}
        
    def add_new_tracker(self,obj_single_tracker,obj_multi_tracker,current_id,prev_id=None):
        '''
        Parameters:
            obj_single_tracker: single camera tracker with id=current_id;
            obj_multi_tracker: multi-cameras tracker between camera with id=current_id and previous camera with id=prev_id;
            current_id: new added camera id;
            prev_id: associated camera id;
        '''
        self.trackers_dict[id] = obj_single_tracker
        if prev_id is not None:
            self.multi_tracker_dict[prev_id] = obj_multi_tracker
            self.associate_dict[prev_id] = current_id       # record mapping relation
        
    def update(self,img_dict,box_dict,frame_count=None):
        '''
        img_dict: images dict
        box_dict: box_dict
        frame_count: frame_count or time stamp
        '''
        # 注: 大坑预警！！！zip()生成的对象只能被访问一次.
        # 例：
        # ====== TEST CASE =====
        # CODE:
        # ----------------------
        # a = [[1,3],[2,4]]
        # b = zip(*a)
        # for elem in b:
            # print("First output:",elem)
        # for elem in b:
            # print("Second Output:",elem)
        # ----------------------
        # OUTPUT:
        # ----------------------
        # >>> First output: (1, 2)
        #     First output: (3, 4)
        # ======================
        # 原因：zip返回的可迭代对象是iterator类型的对象.\
        # 可以在迭代中被next()函数调用并不断返回下一个值\
        # 直至迭代完成.但迭代器指针只能前进,不能后退.\
        # 多次访问zip()对象时,数据区指针无法重置到数据区起始位置\
        # 因此访问结果为空...
        
        # Update all single-camera trackers
        for elem in self.trackers_dict:
            self.hist_record_dict[elem] = []
            self.object_append_status[elem] = False
            if box_dict[elem] is not None:
                self.object_append_dict[elem] = {}
                for bx in box_dict[elem]:
                    box_info = bx[0:4]
                    frame_info = bx[4]
                    cp_img = img_dict[elem][box_info[1]:box_info[3],box_info[0]:box_info[2]].copy()
                    
                    hist,run = self.trackers_dict[elem].update(bx,cp_img)
                    if self.multi_tracker_dict.__contains__(elem):
                        for obj in hist:
                            self.multi_tracker_dict[elem].objects_pool[obj] = hist[obj]
                    self.run_record_dict[elem] = run
                    if self.trackers_dict[elem].new_object_append_status:
                        self.object_append_status[elem] = True
                        self.object_append_dict[elem][bx] = self.trackers_dict[elem].last_append_id

            else:
                hist,run = self.trackers_dict[elem].update([frame_count])
                if self.multi_tracker_dict.__contains__(elem):
                    for obj in hist:
                        self.multi_tracker_dict[elem].objects_pool[obj] = hist[obj]
        # Update all multi-cameras trackers
        for elem in self.multi_tracker_dict:
            associated_elem = self.associate_dict[elem] # use the appended id in cam 2;

            # print(self.multi_tracker_dict[elem].cam2_new_object_id)
            # 问题：如果一副图像有多个目标同时进入监控场，如何工作？
            if box_dict[associated_elem] is not None:
                for bx in box_dict[associated_elem]:
                    box_info = bx[0:4]
                    frame_info = bx[4]
                    cp_img = img_dict[associated_elem][box_info[1]:box_info[3],box_info[0]:box_info[2]].copy()
                    cv2.imshow('cp_img',cp_img)
                    
                    if self.object_append_dict[associated_elem].__contains__(bx):
                        self.multi_tracker_dict[elem].cam2_new_object_id = self.object_append_dict[associated_elem].pop(bx)
                    
                    if self.multi_tracker_dict[elem].cam2_new_object_id is not None:
                        self.multi_tracker_dict[elem].update( box=bx,
                                img = cp_img)
                        self.object_append_status[elem] = False
                        self.multi_tracker_dict[elem].cam2_new_object_id = None
            else:
                hist,run = self.multi_tracker_dict[elem].update([frame_count])
            pass
        
    def get_reverse_associate_dict(self):
        '''Find prev cam id'''
        return {self.associate_dict[elem]:elem for elem in self.associate_dict}

class MCT_STP_tracker(object):
    def __init__(self,
                frame_space_dist = 300,
                obj_STP_tracker_1 = None,
                obj_STP_tracker_2 = None,
                obj_multi_cameras_STP = None,
                match_mode = 'Prob'):
        # time range
        self.frame_space_dist = frame_space_dist
        
        # Single camera tracker
        self.obj_STP_tracker_1 = obj_STP_tracker_1
        self.obj_STP_tracker_2 = obj_STP_tracker_2
        self.obj_multi_cameras_STP = obj_multi_cameras_STP
        
        # Multi cameras
        self.match_mode = match_mode
        
        # save vehicle object out from cam_1
        self.objects_pool = {}
        # 
        self.mapping_recorder = {}
        self.cam2_new_object_id = None
        
        # threshold of tracking
        self.thresh_probability = 0.0001
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
        del_id_list = []
        for elem in self.objects_pool:
            # print(elem,frame,self.objects_pool[elem].last_frame)
            if (frame - self.objects_pool[elem].last_frame) > self.frame_space_dist:
                self.objects_pool[elem].update_status = False
                del_id_list.append(elem)
                
        # can not delete dict key-value in iter
        del_id_dict = {}
        for elem in del_id_list:
            pop_obj = self.objects_pool.pop(elem)   # Delete elem in objects_pool
            del_id_dict[pop_obj.id] = pop_obj
        return del_id_dict
        
    def update(self,box,img=None):
        if len(box) == 1:       # box in cam 2
            return self.isTrackFinish(box[0]),self.objects_pool
            
        box_info = [box[0],box[1],box[2],box[3]]
        frame_info = box[4]
        if self.obj_STP_tracker_2.isBoxInRegion(box_info):
            matched_obj = self.match(box_info,frame_info)
            if matched_obj and self.cam2_new_object_id is not None:
                # self.objects_pool[matched_obj.id].update(box_info,frame_info)
                self.mapping_recorder[matched_obj.id] = self.cam2_new_object_id
        return self.isTrackFinish(box[4]),self.objects_pool
        
    def match(self,box,frame):
        possible_obj_list = []
        print("===== match =====")
        print("self.objects_pool:",self.objects_pool)
        for k,v in self.objects_pool.items():
            cmp_id = k
            # cmp_locat = v.last_box
            # cmp_frame = v.last_frame
            cmp_locat = v.first_box
            cmp_frame = v.first_frame
            # print("update_status:",v.id,v.update_status)
            
            center_x,center_y = get_box_center(box,mode='bottom')
            # print("center_x,center_y:",center_x,center_y)
            base_center_x,base_center_y = get_box_center(cmp_locat,mode='bottom')
            # print("base_center_x,base_center_y:",base_center_x,base_center_y)
            
            # perspective_transform of base point in cam 1
            pt_centers = self.obj_STP_tracker_2.obj_STP.perspective_transformer.get_pred_transform(np.array([[center_x,center_y]],np.float))
            pt_center_x,pt_center_y = pt_centers[0][0]
            
            
            # perspective_transform of point in cam 2
            pt_base_centers = self.obj_STP_tracker_1.obj_STP.perspective_transformer.get_pred_transform(np.array([[base_center_x,base_center_y]],np.float))
            
            pt_center_x,pt_center_y = pt_centers[0][0]
            pt_base_center_x,pt_base_center_y = pt_base_centers[0][0]
            # print("pt_center_x,pt_center_y:",pt_center_x,pt_center_y)
            # print("pt_base_center_x,pt_base_center_y:",pt_base_center_x,pt_base_center_y)
            
            # # ==== TEST: Display the probability map =====
            # img_3 = self.draw_color_probability_map(img_current,pt_base_center_x,pt_base_center_y)
            # cv2.namedWindow("img_current",cv2.WINDOW_NORMAL)
            # cv2.imshow("img_current",img_3)
            # cv2.waitKey(1)
            
            cmp_frame_dist = frame-cmp_frame
            if self.match_mode == 'Prob':
                cmp_result = self.obj_multi_cameras_STP.get_probability(pt_center_x,pt_center_y,pt_base_center_x,pt_base_center_y,cmp_frame_dist)
                
                print(v.id,cmp_result)
                cmp_result = cmp_result[2]
                if cmp_result>=self.thresh_probability and cmp_frame_dist<=self.frame_space_dist:
                    possible_obj_list.append([v,cmp_result,cmp_frame_dist*1./self.frame_space_dist])
            else:   # Dist mode
                cmp_result = self.obj_STP.get_distance(pt_center_x,pt_center_y,pt_base_center_x,pt_base_center_y)
                if cmp_result<=self.thresh_distance and cmp_frame_dist<=self.frame_space_dist:
                    possible_obj_list.append([v,cmp_result,cmp_frame_dist*1./self.frame_space_dist])
        matched_obj = self.rank(possible_obj_list)
        if matched_obj is not None:
            matched_obj.update_status = True
            print(matched_obj.id)
        return matched_obj

    def rank(self,objs_list):
        '''find the nearest object in spatial and temporal space, the default weights of two space is 0.5 and 0.5'''
        if objs_list == []:
            return None
        else:
            def takeSecond(elem):
                return elem[1]
            dist_list = []
            for elem in objs_list:
                dist = elem[1]
                dist_list.append([objs_list[0],dist])
            # dist_list.sort(key=takeSecond)
            if self.match_mode == 'Prob':
                sorted_objs_list = sorted(objs_list,key=lambda x:x[1])
                return sorted_objs_list[-1][0]
            else:
                sorted_objs_list = sorted(objs_list,key=lambda x:x[2]) 
                return sorted_objs_list[0][0]
                
    def draw_trajectory(self,img):
        self.draw_objects_pool()
        if self.display_monitor_region:
            cv2.line(img,(0,self.region_top),(img.shape[1]-1,self.region_top),(255,255,255),5)
            cv2.line(img,(0,img.shape[0]-self.region_bottom),(img.shape[1]-1,img.shape[0]-self.region_bottom),(255,255,255),5)
        for k,v in self.objects_pool.items():
            if v.update_status:
                if len(v.list)>0:
                    for i in range(len(v.list)-1):
                        center_1 = get_box_center(v.list[i][0],mode="bottom")
                        center_2 = get_box_center(v.list[i+1][0],mode="bottom")
                        cv2.line(img,center_1,center_2,v.color,5)
                    cv2.putText(img,"ID:{}".format(v.id),(v.last_box[2],v.last_box[3])
                    ,cv2.FONT_HERSHEY_COMPLEX,2,v.color,5)
        return img
        
    def draw_objects_pool(self):
        if len(self.objects_pool)>0:
            img_height = self.obj_pool_display_height
            img_width = self.obj_pool_display_width*len(self.objects_pool)
            disp_objs_pool_img = np.zeros((img_width,img_height,self.obj_pool_display_channel),np.uint8)
            obj_count = 0
            for k,v in self.objects_pool.items():
                chosen_img = cv2.resize(v.first_img,(self.obj_pool_display_width,self.obj_pool_display_height))
                disp_objs_pool_img[ self.obj_pool_display_width*obj_count:self.obj_pool_display_width*(obj_count+1),0:self.obj_pool_display_height] = chosen_img
                cv2.putText(disp_objs_pool_img,"ID:{}".format(v.id),(0,self.obj_pool_display_height*(obj_count+1)-3),cv2.FONT_HERSHEY_SIMPLEX,1,v.color,2)
                obj_count += 1
            return disp_objs_pool_img
        else:
            return None
            
    def draw_color_probability_map(self,img_current,pt_base_center_x,pt_base_center_y,alpha=0.5):
        # probability color map
        p_map = self.obj_multi_cameras_STP.get_probability_map(base_x=pt_base_center_x,base_y=pt_base_center_y,height=210,width=80)
        p_map = cv2.applyColorMap(p_map,cv2.COLORMAP_JET)
        color_p_map = cv2.resize(p_map,(int(self.obj_STP.perspective_transformer.transformed_width_for_disp),int(self.obj_STP.perspective_transformer.transformed_height_for_disp)))
        color_p_map = cv2.flip(color_p_map,0)   # 0:vertical flip
        pt_color_p_map = self.obj_STP.perspective_transformer.get_inverse_disp_transform(color_p_map)

        img_3 = cv2.addWeighted(img_current, alpha, pt_color_p_map, 1-alpha, 0)

        return img_3
            
# # ===== TEST FUNCTIONS =====
def TrackersArray_test():
    from data_generator import get_files_info
    obj_data_generator = get_files_info()
    
    time_interval = 5
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
    from cameras_associate import Multi_cameras_STP
    
    # objects mapping between cam_1 and cam_2
    associate_dict_c1_c2 = {
        0:0,
        1:1,
        2:2,
        3:3,
        4:4
    }
    Multi_STP_Predictor = Multi_cameras_STP(
                                    STP_Predictor_1,
                                    STP_Predictor_2,
                                    associate_dict_c1_c2)
    
    from Single_camera_track import STP_tracker
    obj_tracker_1 = STP_tracker(frame_space_dist=50,obj_STP=STP_Predictor_1)
    obj_tracker_1.match_mode = 'Prob'
    obj_tracker_1.display_monitor_region = True
    obj_tracker_2 = STP_tracker(frame_space_dist=50,obj_STP=STP_Predictor_2)
    obj_tracker_2.region_top = 250
    obj_tracker_2.match_mode = 'Prob'
    obj_tracker_2.display_monitor_region = True
    
    obj_MCT_tracker_1 = MCT_STP_tracker(
                            obj_STP_tracker_1 = obj_tracker_1,
                            obj_STP_tracker_2 = obj_tracker_2,
                            obj_multi_cameras_STP = Multi_STP_Predictor
                            )
    
    trackers_dict={
        0:obj_tracker_1,
        1:obj_tracker_2}
    multi_tracker_dict={
        0:obj_MCT_tracker_1}
    associate_dict={
        0:1}
    obj_TrackersArray = TrackersArray(
                                trackers_dict=trackers_dict,
                                multi_tracker_dict=multi_tracker_dict,
                                associate_dict=associate_dict)
                                
    from data_generator import two_cameras_simulator    # Just for test
    cameras_simulator = two_cameras_simulator(
                            file_dict_1['box_info_filepath'],
                            file_dict_2['box_info_filepath'],
                            file_dict_1['img_filepath'],
                            file_dict_2['img_filepath'])
    datagen = cameras_simulator.data_gen(time_interval=time_interval)
    
    frame_count = 0
    try:
        while(True):
            img_1,img_2,d_1,d_2 = datagen.__next__()
            img_dict = {
                0:img_1,
                1:img_2
            }
            box_dict = {
                0:d_1,
                1:d_2
            }
            obj_TrackersArray.update(
                                img_dict = img_dict,
                                box_dict = box_dict,
                                frame_count = frame_count
                                )
            frame_count+=time_interval
            # if d_1 is not None:
                # for elem in d_1:
                    # cp_img = img_1[elem[1]:elem[3],elem[0]:elem[2]].copy()
                    # hist_objs_1,_= obj_TrackersArray.trackers_dict[0].update(elem,cp_img)
            # if d_2 is not None:
                # for elem in d_2:
                    # cp_img = img_2[elem[1]:elem[3],elem[0]:elem[2]].copy()
                    # obj_TrackersArray.trackers_dict[1].update(elem,cp_img)
            tray_img_1 = obj_TrackersArray.trackers_dict[0].draw_trajectory(img_1)
            tray_img_2 = obj_TrackersArray.trackers_dict[1].draw_trajectory(img_2)
            
            obj_pool_img_1 = obj_TrackersArray.multi_tracker_dict[0].draw_objects_pool()
            if obj_pool_img_1 is not None:
                cv2.imshow('obj_pool_img_1',obj_pool_img_1)
            
            cv2.namedWindow('img_1',cv2.WINDOW_NORMAL)
            cv2.namedWindow('img_2',cv2.WINDOW_NORMAL)
            cv2.imshow('img_1',tray_img_1)
            cv2.imshow('img_2',tray_img_2)
            cv2.waitKey(1)
    except StopIteration:
        pass
    
    for elem in obj_TrackersArray.multi_tracker_dict:
        print(obj_TrackersArray.multi_tracker_dict[elem].mapping_recorder)
    return

def MCT_STP_tracker_test():
    from data_generator import get_files_info
    obj_data_generator = get_files_info()
    
    time_interval = 10
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
    single_camera_STP_1 = Single_camera_STP(
                        tracker_record_1,
                        pt_obj_1,
                        time_interval = time_interval)
    single_camera_STP_1.var_beta_x = int(25*25/time_interval)
    single_camera_STP_1.var_beta_y = int(25/time_interval)+1
    single_camera_STP_1.update_predictor()            
                        
    single_camera_STP_2 = Single_camera_STP(
                        tracker_record_2,
                        pt_obj_2,
                        time_interval = time_interval)
    single_camera_STP_2.var_beta_x = int(25*25/time_interval)
    single_camera_STP_2.var_beta_y = int(25/time_interval)+1
    single_camera_STP_2.update_predictor()
    
    # id association of same vehicle between cam_1 and cam_2
    associate_dict_c1_c2 = {
        0:0,
        1:1,
        2:2,
        3:3,
        4:4
    }
    
    from cameras_associate import Multi_cameras_STP
    obj_multi_cameras_STP = Multi_cameras_STP(
                                single_camera_STP_1,
                                single_camera_STP_2,
                                associate_dict_c1_c2
        )
    
    # Single camera STP_tracker object
    from Single_camera_track import STP_tracker
    obj_tracker_1 = STP_tracker(frame_space_dist=50,obj_STP=single_camera_STP_1)
    obj_tracker_1.frame_space_dist = 10
    obj_tracker_1.match_mode = 'Prob'
    obj_tracker_1.display_monitor_region = True
    obj_tracker_2 = STP_tracker(frame_space_dist=50,obj_STP=single_camera_STP_2)
    obj_tracker_2.frame_space_dist = 10
    obj_tracker_2.region_top = 250  # Optional
    obj_tracker_2.match_mode = 'Prob'
    obj_tracker_2.display_monitor_region = True
    
    # Multi_cameras STP_tracker object
    obj_MCT_STP_tracker = MCT_STP_tracker(
                            obj_STP_tracker_1 = obj_tracker_1,
                            obj_STP_tracker_2 = obj_tracker_2,
                            obj_multi_cameras_STP = obj_multi_cameras_STP
                            )
    
    from data_generator import two_cameras_simulator    # Just for test
    cameras_simulator = two_cameras_simulator(
                            file_dict_1['box_info_filepath'],
                            file_dict_2['box_info_filepath'],
                            file_dict_1['img_filepath'],
                            file_dict_2['img_filepath'])
    datagen = cameras_simulator.data_gen(time_interval=time_interval)
    
    frame_count = 0
    try:
        while(True):
            img_1,img_2,d_1,d_2 = datagen.__next__()
            
            if d_1 is not None:
                for elem in d_1:
                    cp_img = img_1[elem[1]:elem[3],elem[0]:elem[2]].copy()
                    hist_objs_1,_ = obj_MCT_STP_tracker.obj_STP_tracker_1.update(elem,cp_img)
                    for obj in hist_objs_1:
                        obj_MCT_STP_tracker.objects_pool[obj] = hist_objs_1[obj]
            else:
                hist_objs_1,_ = obj_MCT_STP_tracker.obj_STP_tracker_1.update([frame_count])
            # obj_MCT_STP_tracker.isTrackFinish(frame_count)
            
            if d_2 is not None:
                for elem in d_2:
                    cp_img = img_2[elem[1]:elem[3],elem[0]:elem[2]].copy()
                    hist_objs_2,run_objs_2 = obj_MCT_STP_tracker.obj_STP_tracker_2.update(elem,cp_img)
                    if obj_MCT_STP_tracker.obj_STP_tracker_2.new_object_append_status:
                        print("update")
                        obj_MCT_STP_tracker.cam2_new_object_id = obj_MCT_STP_tracker.obj_STP_tracker_2.last_append_id
                        obj_MCT_STP_tracker.update(
                                    box=elem,
                                    img=cp_img)
                        obj_MCT_STP_tracker.cam2_new_object_id = None
            else:
                hist_objs_2,_ = obj_MCT_STP_tracker.obj_STP_tracker_2.update([frame_count])
      
            tray_img_1 = obj_MCT_STP_tracker.obj_STP_tracker_1.draw_trajectory(img_1)
            tray_img_2 = obj_MCT_STP_tracker.obj_STP_tracker_2.draw_trajectory(img_2)
            obj_pools = obj_MCT_STP_tracker.draw_objects_pool()
            if obj_pools is not None:
                cv2.imshow('obj_pools',obj_pools)
            cv2.namedWindow('img_1',cv2.WINDOW_NORMAL)
            cv2.namedWindow('img_2',cv2.WINDOW_NORMAL)
            
            cv2.imshow('img_1',tray_img_1)
            cv2.imshow('img_2',tray_img_2)
            
            # height,width,channels = tray_img_1.shape
            # img_3 = np.zeros((int(height/2),width,channels),np.uint8)
            # concat_img_1 = cv2.resize(tray_img_1,(int(width/2),int(height/2)))
            # print(concat_img_1.shape)
            # concat_img_2 = cv2.resize(tray_img_2,(int(width/2),int(height/2)))
            # img_3[0:int(height/2),0:int(width/2),:]=concat_img_1[:,:,:]
            # img_3[0:int(height/2),int(width/2):width,:]=concat_img_2[:,:,:]
            
            # cv2.imwrite(os.path.join(r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data\resutls\multi_stp_tracking',str(frame_count)+'.jpg'),img_3)
            # cv2.namedWindow('img_3',cv2.WINDOW_NORMAL)
            # cv2.imshow('img_3',img_3)
            
            frame_count+=time_interval
            
            cv2.waitKey(1)
    except StopIteration:
        pass
    print(obj_MCT_STP_tracker.mapping_recorder)
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
    # # ===== TEST: TrackerArray_test =====
    TrackersArray_test()
    # # ===== TEST: Multi-cameras STP tracker test =====
    # MCT_STP_tracker_test()
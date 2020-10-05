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
import time

# import compute_VeRi_dis as dist
# from Model_Wrapper import ResNet_Loader


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
    def __init__(self, trackers_dict={}, multi_tracker_dict={}, associate_dict={}):
        '''
        trackers_dict: tracker with CAM_id
        STP_dict: spatial-temporal-prior between CAM_id and next_CAM_id
        associate_dict: prev_CAM_id: CAM_id
        '''
        self.trackers_dict = trackers_dict
        self.first_tracker_id = None
        self.last_tracker_id = None

        self.multi_tracker_dict = multi_tracker_dict
        self.associate_dict = associate_dict

        self.get_first_last_tracker()

        self.hist_record_dict = {}
        self.run_record_dict = {}
        self.object_append_status = {}
        self.object_append_dict = {}

        self.time_stamp = None

    def get_first_last_tracker(self):
        '''Get the first and last tracker id from associate_dict'''
        prev_ids = [elem for elem in self.associate_dict]
        next_ids = [self.associate_dict[elem] for elem in self.associate_dict]

        all_ids = prev_ids.copy()
        all_ids.extend(next_ids)   # Note: extend function returns None
        all_ids = np.unique(all_ids)

        self.first_tracker_id = [elem for elem in all_ids if elem not in next_ids][0]
        self.last_tracker_id = [elem for elem in all_ids if elem not in prev_ids][0]

        return self.first_tracker_id,self.last_tracker_id

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

    def update(self, img_dict, box_dict, frame_count=None):
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
        self.time_stamp = frame_count
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
                            self.multi_tracker_dict[elem].objects_pool[obj].update_status = True
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
                    # cv2.imshow('cp_img',cp_img)

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

    def draw_trajectory(self,img_dict=None,box_dict=None,mode='single'):
        new_img_dict = {}
        if mode == 'single':
            for elem in img_dict:
                new_img_dict[elem] = self.trackers_dict[elem].draw_trajectory(img_dict[elem])
        elif mode == 'multi':
            for elem in img_dict:
                for obj in self.trackers_dict[elem].objects_pool:
                    idx = self.get_global_id(elem,self.trackers_dict[elem].objects_pool[obj].id)
                    box = self.trackers_dict[elem].objects_pool[obj].last_box
                    new_img_dict[elem] = cv2.putText(img_dict[elem],"ID:{}".format(idx),(box[2],box[3]),cv2.FONT_HERSHEY_COMPLEX,2,self.trackers_dict[elem].objects_pool[obj].color,5)
                    bx_list = []
                    for bx in self.trackers_dict[elem].objects_pool[obj].list:
                        bx_list.append(get_box_center(bx[0],mode="bottom"))
                    for i in range(len(bx_list)-1):
                        new_img_dict[elem] = cv2.line(new_img_dict[elem],bx_list[i],bx_list[i+1],self.trackers_dict[elem].objects_pool[obj].color,5)
                    # draw monitor region
                    if self.trackers_dict[elem].display_monitor_region:
                        # region top
                        new_img_dict[elem] = cv2.line(new_img_dict[elem],(0,self.trackers_dict[elem].region_top),(img_dict[elem].shape[1],self.trackers_dict[elem].region_top),(255,255,255),5)
                        # region bottom
                        new_img_dict[elem] = cv2.line(new_img_dict[elem],(0,img_dict[elem].shape[0]-self.trackers_dict[elem].region_bottom),(img_dict[elem].shape[1],img_dict[elem].shape[0]-self.trackers_dict[elem].region_bottom),(255,255,255),5)

        return new_img_dict

    def draw_single_camera_objects_pool(self,mode='h',set_range=0):
        output_img_dict = {}
        for elem in self.trackers_dict:
            output_img_dict[elem] = self.trackers_dict[elem].draw_objects_pool(mode=mode,set_range=set_range)
        return output_img_dict

    def draw_inter_camera_objects_pool(self,mode='v',set_range=0):
        output_img_dict = {}
        for elem in self.multi_tracker_dict:
            output_img_dict[elem] = self.multi_tracker_dict[elem].draw_objects_pool(mode=mode,set_range=set_range)
        return output_img_dict

    def get_global_id(self,device_id,obj_id):
        output_id = None
        # Recursive condition
        if self.get_reverse_associate_dict().__contains__(device_id):
            if self.multi_tracker_dict[self.get_reverse_associate_dict()[device_id]].mapping_recorder.__contains__(obj_id):
                output_id = self.get_global_id(
                    self.get_reverse_associate_dict()[device_id],
                    self.multi_tracker_dict[self.get_reverse_associate_dict()[device_id]].get_reverse_mapping_recorder()[obj_id]
                )
            else:
                return None
        else:
            output_id = obj_id
        return output_id

    def get_reverse_associate_dict(self):
        '''Find prev cam id'''
        return {self.associate_dict[elem]:elem for elem in self.associate_dict}

class MCT_STP_tracker(object):
    def __init__(self,
                frame_space_dist = 250,
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
        self.reverse_mapping_recorder = {}
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
            if (frame - self.objects_pool[elem].last_frame) > self.frame_space_dist or self.mapping_recorder.__contains__(elem):
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
        if self.obj_STP_tracker_2.isBoxInRegion(box_info)==0:
            matched_obj = self.match(box_info,frame_info)
            if matched_obj and self.cam2_new_object_id is not None:
                # self.objects_pool[matched_obj.id].update(box_info,frame_info)
                self.mapping_recorder[matched_obj.id] = self.cam2_new_object_id
                self.get_reverse_mapping_recorder()

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
            # print("len(objs_list):",len(objs_list))
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

    def draw_objects_pool(self,mode='v',set_range=0):
        from Draw_trajectory import draw_objects_pool
        return draw_objects_pool(   self.objects_pool,
                                    self.obj_pool_display_height,
                                    self.obj_pool_display_width,
                                    self.obj_pool_display_channel,
                                    mode=mode,
                                    set_range=set_range)

    def draw_color_probability_map(self,img_current,pt_base_center_x,pt_base_center_y,alpha=0.5):
        # probability color map
        p_map = self.obj_multi_cameras_STP.get_probability_map(base_x=pt_base_center_x,base_y=pt_base_center_y,height=210,width=80)
        p_map = cv2.applyColorMap(p_map,cv2.COLORMAP_JET)
        color_p_map = cv2.resize(p_map,(int(self.obj_STP.perspective_transformer.transformed_width_for_disp),int(self.obj_STP.perspective_transformer.transformed_height_for_disp)))
        color_p_map = cv2.flip(color_p_map,0)   # 0:vertical flip
        pt_color_p_map = self.obj_STP.perspective_transformer.get_inverse_disp_transform(color_p_map)

        img_3 = cv2.addWeighted(img_current, alpha, pt_color_p_map, 1-alpha, 0)

        return img_3

    def get_reverse_mapping_recorder(self):
        self.reverse_mapping_recorder = {self.mapping_recorder[elem]:elem for elem in self.mapping_recorder}
        return self.reverse_mapping_recorder

# # ===== TEST FUNCTIONS =====
def TrackersArray_test():
    from data_generator import get_files_info
    obj_data_generator = get_files_info()

    time_interval = 2
    # file path 1
    file_dict_1 = obj_data_generator.get_filepath(0)
    file_dict_2 = obj_data_generator.get_filepath(1)

    from data_generator import load_tracking_info
    tracker_record_1 = load_tracking_info(file_dict_1['tracking_info_filepath'])
    tracker_record_2 = load_tracking_info(file_dict_2['tracking_info_filepath'])

    from Perspective_transform import Perspective_transformer
    pt_obj_1 = Perspective_transformer(file_dict_1['pt_savepath'])
    pt_obj_2 = Perspective_transformer(file_dict_2['pt_savepath'])

    from cameras_associate import SingleCameraSTP
    STP_Predictor_1 = SingleCameraSTP(
                        tracker_record_1,
                        pt_obj_1,
                        time_interval = time_interval)
    STP_Predictor_1.var_beta_x = int(25*25/time_interval)
    STP_Predictor_1.var_beta_y = int(25/time_interval)+1
    STP_Predictor_1.update_predictor()

    STP_Predictor_2 = SingleCameraSTP(
                        tracker_record_2,
                        pt_obj_2,
                        time_interval = time_interval)
    STP_Predictor_2.var_beta_x = int(25*25/time_interval)
    STP_Predictor_2.var_beta_y = int(25/time_interval)+1
    STP_Predictor_2.update_predictor()
    from cameras_associate import MultiCamerasSTP

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
    obj_tracker_1 = STP_tracker(frame_space_dist=75,obj_STP=STP_Predictor_1)
    obj_tracker_1.region_top = 450
    obj_tracker_1.match_mode = 'Prob'
    obj_tracker_1.display_monitor_region = True
    obj_tracker_2 = STP_tracker(frame_space_dist=75,obj_STP=STP_Predictor_2)
    obj_tracker_2.region_top = 450
    obj_tracker_2.match_mode = 'Prob'
    obj_tracker_2.display_monitor_region = True

    obj_MCT_tracker_1 = MCT_STP_tracker(
                            frame_space_dist = 300,
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

    from Draw_trajectory import Canvas
    obj_canvas = Canvas(obj_TrackersArray)

    from data_generator import two_cameras_simulator    # Just for test
    cameras_simulator = two_cameras_simulator(
                            file_dict_1['box_info_filepath'],
                            file_dict_2['box_info_filepath'],
                            file_dict_1['img_filepath'],
                            file_dict_2['img_filepath'])
    datagen = cameras_simulator.data_gen(time_interval=time_interval)

    frame_count = 0
    count = 0
    start = time.clock()
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
            count += 1

            # # ==== TEST:Draw object trajectory in single-camera =====
            # trajectory_img = obj_TrackersArray.draw_trajectory(img_dict=img_dict,mode='multi')
            # Display images
            trajectory_img = obj_TrackersArray.draw_trajectory(img_dict)
            # for elem in trajectory_img:
                # cv2.namedWindow('trajectory_img_'+str(elem),cv2.WINDOW_NORMAL)
                # cv2.imshow('trajectory_img_'+str(elem),trajectory_img[elem])
                # cv2.waitKey(1)
            # # ==== TEST:Draw objects pool in single-camera =====
            single_objs_pool_img = obj_TrackersArray.draw_single_camera_objects_pool(set_range=960)
            # # Display images
            # for elem in single_objs_pool_img:
                # if single_objs_pool_img[elem] is not None:
                    # cv2.imshow('single_objs_pool_img_'+str(elem),single_objs_pool_img[elem])
                # cv2.waitKey(1)
            # # ==== TEST:Draw objects pool in inter-camera =====
            multi_obj_pool_img = obj_TrackersArray.draw_inter_camera_objects_pool(set_range=540)
            # # Display images
            # if multi_obj_pool_img is not None:
                # for elem in multi_obj_pool_img:
                    # if multi_obj_pool_img[elem] is not None:
                        # cv2.imshow('multi_obj_pool_img_'+str(elem),multi_obj_pool_img[elem])
                    # cv2.waitKey(1)

            from Draw_trajectory import draw_objects_on_canvas
            canvas_img = draw_objects_on_canvas(obj_canvas,obj_TrackersArray)

            from Draw_trajectory import draw_all_results
            result_img = draw_all_results(trajectory_img,single_objs_pool_img,multi_obj_pool_img,canvas_img,[0,1],540,960,100,300)

            cv2.namedWindow("result_img", cv2.WINDOW_NORMAL)
            cv2.imshow("result_img", result_img)
            # cv2.imwrite(r"D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data\resutls\camera_array\ver2\{}\{}.jpg".format(time_interval,frame_count),result_img)
            cv2.waitKey(1)

    except StopIteration:
        pass
    for elem in obj_TrackersArray.multi_tracker_dict:
        elapsed = (time.clock() - start)
        print(elapsed,count,count/elapsed)
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

    from cameras_associate import SingleCameraSTP
    single_camera_STP_1 = SingleCameraSTP(
                        tracker_record_1,
                        pt_obj_1,
                        time_interval = time_interval)
    single_camera_STP_1.var_beta_x = int(25*25/time_interval)
    single_camera_STP_1.var_beta_y = int(25/time_interval)+1
    single_camera_STP_1.update_predictor()

    single_camera_STP_2 = SingleCameraSTP(
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
                        obj_MCT_STP_tracker.objects_pool[obj].update_status = True
            else:
                hist_objs_1,_ = obj_MCT_STP_tracker.obj_STP_tracker_1.update([frame_count])
            # obj_MCT_STP_tracker.isTrackFinish(frame_count)

            if d_2 is not None:
                for elem in d_2:
                    cp_img = img_2[elem[1]:elem[3],elem[0]:elem[2]].copy()
                    hist_objs_2,run_objs_2 = obj_MCT_STP_tracker.obj_STP_tracker_2.update(elem,cp_img)
                    if obj_MCT_STP_tracker.obj_STP_tracker_2.new_object_append_status:

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

            cv2.waitKey()
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


# ===============================================================================================
# ==== WARNING: The author is mad when he writing this part, code lines below may be useless ====
# ===============================================================================================

def i_completely_forget_save_me():
    """Just for showing, it doesn't work in reality"""
    from Common import MCT_RESULT
    from Common import cam_names, roi_info, save_path, track_info, associate_info
    from cameras_associate import get_associate_dict
    from Perspective_transform import Perspective_transformer


    # associate_dict: TEST PASS
    associate_dict = get_associate_dict(associate_info)

    pt_transformer_1 = Perspective_transformer(roi_info[1])
    pt_transformer_2 = Perspective_transformer(roi_info[2])
    pt_transformer_3 = Perspective_transformer(roi_info[3])
    pt_transformer_4 = Perspective_transformer(roi_info[4])

    with open(track_info[1], 'r') as doc:
        trace_1 = json.load(doc)
    with open(track_info[2], 'r') as doc:
        trace_2 = json.load(doc)
    with open(track_info[3], 'r') as doc:
        trace_3 = json.load(doc)
    with open(track_info[4], 'r') as doc:
        trace_4 = json.load(doc)

    # labeled img; cropped img; traces; transformers;
    cam_array = [
        [save_path[1], os.path.join(save_path[1], "images"), trace_1, pt_transformer_1],
        [save_path[2], os.path.join(save_path[2], "images"), trace_2, pt_transformer_2],
        [save_path[3], os.path.join(save_path[3], "images"), trace_3, pt_transformer_3],
        [save_path[4], os.path.join(save_path[4], "images"), trace_4, pt_transformer_4]
    ]

    dist_1, diff_1, spd_1 = estimate_distance(trace_1, trace_2, associate_dict["003"])
    dist_2, diff_2, spd_2 = estimate_distance(trace_2, trace_3, associate_dict["004"])
    dist_3, diff_3, spd_3 = estimate_distance(trace_3, trace_4, associate_dict["005"])

    # cam_2:1387+946;
    # cam_3:1388+156;   210;
    # cam_4:1388+324;   168; (210); 547; +337;
    # cam_5:1388+534;   210; 35;         -175;

    # print(dist_1, dist_2, dist_3)
    # print(diff_1, diff_2, diff_3)
    # print(spd_1, spd_2, spd_3)

    # 186.87489281155294    547.9742216846969       35.846546287736814  m
    # 166.5142857142857     528.875                 34.55263157894737   frames
    # 28.421919696601453    25.913013562801034      27.095261951284453  m/s
    # 210/30 = 7;7*25=175;  168/30 = 6; 6*25=150    210/30 = 7; 7*25=175
    #                       525 - 150 = 375         35 - 175 = 140

    # # get_cam_assoicate(trace_front=cam_array[0][2], trace_back=cam_array[1][2], associate_dict=associate_dict)

    f1_in, f2_in, f3_in, f4_in, f1_out, f2_out, f3_out = get_objectid_in_each_frame(
        trace_1=trace_1,
        trace_2=trace_2,
        trace_3=trace_3,
        trace_4=trace_4,
        assoc_dict_12=associate_dict["003"],
        assoc_dict_23=associate_dict["004"],
        assoc_dict_34=associate_dict["005"],
    )     # 003, 004, 005


    # 多摄像机跟踪路径绘制
    seg_setting ={'speed':[28, 25, 27], 'dist':[200, 600, 50]}

    # draw_canvas_with_objects(trace_list=[trace_1, trace_2, trace_3, trace_4],
    #                          assoc_dict=associate_dict,
    #                          transformer_list=[pt_transformer_1, pt_transformer_2, pt_transformer_3, pt_transformer_4],
    #                          seg_setting=seg_setting)


    # # 读入图片 PASS
    for i in range(1, 3001):
        filename = "{:0>4d}.jpg".format(i)
        imgs = [cv2.imread(os.path.join(elem[0], filename)) for elem in cam_array]

        in_scene_objs_1 = draw_in_scene_objs(trace_1, f1_in, i, cam_array[0][0])
        in_scene_objs_2 = draw_in_scene_objs(trace_2, f2_in, i, cam_array[1][0])
        in_scene_objs_3 = draw_in_scene_objs(trace_3, f3_in, i, cam_array[2][0])
        in_scene_objs_4 = draw_in_scene_objs(trace_4, f4_in, i, cam_array[3][0])
        out_scene_objs_1 = draw_in_scene_objs(trace_1, f1_out, i, cam_array[0][0], mode='v')
        out_scene_objs_2 = draw_in_scene_objs(trace_2, f2_out, i, cam_array[1][0], mode='v')
        out_scene_objs_3 = draw_in_scene_objs(trace_3, f3_out, i, cam_array[2][0], mode='v')
        if in_scene_objs_1 is None:
            in_scene_objs_1 = np.zeros((100, 700, 3), np.uint8)
        if in_scene_objs_2 is None:
            in_scene_objs_2 = np.zeros((100, 700, 3), np.uint8)
        if in_scene_objs_3 is None:
            in_scene_objs_3 = np.zeros((100, 700, 3), np.uint8)
        if in_scene_objs_4 is None:
            in_scene_objs_4 = np.zeros((100, 700, 3), np.uint8)
        if out_scene_objs_1 is None:
            out_scene_objs_1 = np.zeros((700, 100, 3), np.uint8)
        if out_scene_objs_2 is None:
            out_scene_objs_2 = np.zeros((700, 100, 3), np.uint8)
        if out_scene_objs_3 is None:
            out_scene_objs_3 = np.zeros((700, 100, 3), np.uint8)

        trace_img_1 = cv2.imread(os.path.join(MCT_RESULT, 'trace_1\\{:0>4d}.jpg'.format(i)))
        trace_img_2 = cv2.imread(os.path.join(MCT_RESULT, 'trace_2\\{:0>4d}.jpg'.format(i)))

        # cv2.namedWindow("002", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("003", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("004", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("005", cv2.WINDOW_NORMAL)
        # cv2.imshow("002", imgs[0])
        # cv2.imshow("003", imgs[1])
        # cv2.imshow("004", imgs[2])
        # cv2.imshow("005", imgs[3])
        # cv2.imshow("trace_1", trace_img_1)
        # cv2.imshow("trace_2", trace_img_2)
        #
        #
        # cv2.imshow("in_scene_objs_1", in_scene_objs_1)
        # cv2.imshow("in_scene_objs_2", in_scene_objs_2)
        # cv2.imshow("in_scene_objs_3", in_scene_objs_3)
        # cv2.imshow("in_scene_objs_4", in_scene_objs_4)
        # cv2.imshow("out_scene_objs_1", out_scene_objs_1)
        # cv2.imshow("out_scene_objs_2", out_scene_objs_2)
        # cv2.imshow("out_scene_objs_3", out_scene_objs_3)

        im_width, im_height = 275, 275
        pool_width, pool_height = 60, 60
        trace_height = 190

        width_setting = [im_width, pool_width, im_width, pool_width, im_width, pool_width, im_width]
        height_setting = [im_height, pool_height, trace_height, trace_height]

        width_mk = [0]
        for elem in width_setting:
            width_mk.append(width_mk[-1] + elem)
        print(width_mk)

        height_mk = [0]
        for elem in height_setting:
            height_mk.append(height_mk[-1] + elem)
        print(height_mk)

        result_image = np.zeros((720, 1280, 3), np.uint8)
        in_scene_objs = [in_scene_objs_1, in_scene_objs_2, in_scene_objs_3, in_scene_objs_4]
        for j in range(4):
            result_image[height_mk[0]:height_mk[1], width_mk[2*j]:width_mk[2*j+1]] = cv2.resize(imgs[j], (im_width, im_height), interpolation=cv2.INTER_LINEAR)
        for j in range(4):
            result_image[height_mk[1]:height_mk[2], width_mk[2 * j]:width_mk[2 * j + 1]] = cv2.resize(in_scene_objs[j],
                                                                                                      (im_width, pool_height),
                                                                                                      interpolation=cv2.INTER_LINEAR)
        out_scene_objs = [out_scene_objs_1, out_scene_objs_2, out_scene_objs_3]
        for j in range(3):
            result_image[height_mk[0]:height_mk[1], width_mk[2*j+1]:width_mk[2*(j + 1)]] = cv2.resize(out_scene_objs[j],
                                                                                                      (pool_width, im_height),
                                                                                                      interpolation=cv2.INTER_LINEAR)
        result_image[height_mk[2]:height_mk[3], 0:1280] = cv2.resize(
            trace_img_1,
            (1280, trace_height),
            interpolation=cv2.INTER_LINEAR)
        result_image[height_mk[3]+4:height_mk[4]+4, 0:1280] = cv2.resize(
            trace_img_2,
            (1280, trace_height),
            interpolation=cv2.INTER_LINEAR)

        # for i in range()
        cv2.namedWindow("result_image", cv2.WINDOW_NORMAL)
        cv2.imwrite(os.path.join(MCT_RESULT, "show\\{:0>4d}.jpg".format(i)), result_image)
        cv2.imshow("result_image", result_image)


        # if len(cropped_imgs[i]) > 0:
        #     scene_img = []
        #     for v, elem in enumerate(cropped_imgs[i]):
        #         fname = 'id_{:0>4d}.jpg'.format(int(elem))
        #         scene_img.append(cv2.imread(os.path.join(cam_array[0][1], fname)))
        #         cv2.imshow(str(v), scene_img[v])
        cv2.waitKey(1)
        # print(cam_array[0][2][str(i)])

    pass

class DrawObject(object):
    def __init__(self,
                 id=None,
                 color=None,
                 first_img=None):
        self.id = id
        self.color = color
        self.first_img = first_img
        self.list = None

def draw_in_scene_objs(trace, f, frame, save_path, mode='h'):
    from Draw_trajectory import draw_objects_pool
    objs = {}
    for elem in f[frame]:       # 帧为key，内容为id
        objs[elem] = DrawObject(id=elem, color=trace[elem]['color'], first_img=cv2.imread((os.path.join(save_path, "images\\id_{:0>4d}.jpg".format(int(elem))))))
    if mode == 'h':
        return draw_objects_pool(objs, set_height=100, set_width=100, set_channel=3, set_range=700, mode='h')
    else:
        return draw_objects_pool(objs, set_height=100, set_width=100, set_channel=3, set_range=800, mode='v')


def draw_canvas_with_objects(trace_list=None, assoc_dict=None, transformer_list=None, seg_setting=None):
    from Draw_trajectory import draw_objects_on_canvas, Vehicle, CanvasV2, Trajectory
    from Common import MCT_RESULT
    object_in_canvas = {}
    infer_chain = associate_chain(assoc_dict)
    rcd_chain = record_chain(trace_list=trace_list, transformer_list=transformer_list, infer_chain=infer_chain, seg_setting=seg_setting)
    canvas_obj = CanvasV2(rcd_chain=rcd_chain,
                          seg_setting=seg_setting,
                          reg_setting={"height": [elem.transformed_height_for_pred for elem in transformer_list],
                                       "width": [elem.transformed_width_for_pred for elem in transformer_list]},
                          device_count=4)
    img = canvas_obj.draw()

    # display chosen objs
    chosen_obj = ['1', '23', '37']

    record_json = {}

    print(rcd_chain['1'].list)
    for i in range(1, 3001):
        new_canvas = canvas_obj.draw()
        if len(chosen_obj) == 0:
            for elem in rcd_chain:
                tpobj = rcd_chain[elem]
                record_json[elem] = rcd_chain[elem].list
                tppts = tpobj.update(i)

                if tppts is not None:
                    if len(tppts) > 1:
                        for j in range(len(tppts)-1):
                            cv2.line(new_canvas,
                                     (int(tppts[j][1]*canvas_obj.scale["width"]), int(tppts[j][0]*canvas_obj.scale["height"])),
                                     (int(tppts[j+1][1]*canvas_obj.scale["width"]), int(tppts[j+1][0]*canvas_obj.scale["height"])),
                                     tpobj.color,
                                     2)

                    objV = Vehicle(type='car', color=tpobj.color, id=elem)
                    objV.draw(new_canvas, x=int(tpobj.list_dict[i][2][1]*canvas_obj.scale["width"]), y=int(tpobj.list_dict[i][2][0]*canvas_obj.scale["height"]))
        else:
            for elem in chosen_obj:
                tpobj = rcd_chain[elem]
                record_json[elem] = rcd_chain[elem].list
                tppts = tpobj.update(i)
                if tppts is not None:
                    if len(tppts) > 1:
                        for j in range(len(tppts)-1):
                            cv2.line(new_canvas,
                                     (int(tppts[j][1]*canvas_obj.scale["width"]), int(tppts[j][0]*canvas_obj.scale["height"])),
                                     (int(tppts[j+1][1]*canvas_obj.scale["width"]), int(tppts[j+1][0]*canvas_obj.scale["height"])),
                                     tpobj.color,
                                     2)

                    objV = Vehicle(type='car', color=tpobj.color, id=elem)
                    objV.draw(new_canvas, x=int(tpobj.list_dict[i][2][1]*canvas_obj.scale["width"]), y=int(tpobj.list_dict[i][2][0]*canvas_obj.scale["height"]))
        cv2.imshow("canvas", new_canvas)

        filesavepath = os.path.join(MCT_RESULT, 'show')
        # cv2.imwrite(os.path.join(filesavepath, "{:0>4d}.jpg".format(i)), new_canvas)
        cv2.waitKey(1)

    with open(os.path.join(MCT_RESULT, 'record.json'), 'w') as doc:
        json.dump(record_json, doc)

def associate_chain(assoc_dict=None):
    # ==========TODO===============
    assoc_dict_1 = assoc_dict["003"]
    assoc_dict_2 = assoc_dict["004"]
    assoc_dict_3 = assoc_dict["005"]
    # ==============================
    infer_chain = {}
    # 初始化链条
    # 2:3:4:5
    # 1-2-3
    for elem in assoc_dict_1:
        infer_chain[elem] = [elem, assoc_dict_1[elem]]
        if assoc_dict_2.__contains__(assoc_dict_1[elem]):
            infer_chain[elem].append(assoc_dict_2[assoc_dict_1[elem]])
            if assoc_dict_3.__contains__(assoc_dict_2[assoc_dict_1[elem]]):
                infer_chain[elem].append(assoc_dict_3[assoc_dict_2[assoc_dict_1[elem]]])
    print(infer_chain)
    return infer_chain


def record_chain(trace_list=None, transformer_list=None, infer_chain=None, seg_setting=None):
    from Draw_trajectory import TrajectoryV2
    from Single_camera_track import get_box_center

    frame_rate = 25

    seg_length_setting = [elem.transformed_height_for_pred for elem in transformer_list]

    start_dist = [0]
    for i in range(len(seg_length_setting)-1):
        start_dist.append(start_dist[-1] + seg_setting['dist'][i] + seg_length_setting[i])
    print("seg_dist", seg_setting['dist'])
    print("seg_length", seg_length_setting)
    print("start_dist", start_dist)

    rcd_chain = {}
    for elem in infer_chain:    # elem: 起始id '1',
        traj_obj = TrajectoryV2(id=elem, color=trace_list[0][elem]['color'])
        traj_pos = []
        for v, tr in enumerate(infer_chain[elem]):    # 路径各段id  v:0-3, tr:'1', '5', '13', '15'...
            if int(tr) > 0:         # 分段路径id
                tmp_info = trace_list[v][tr]
                for tmp_pos in tmp_info["list"]:
                    ct_tmp_pos = np.array([list(get_box_center(tmp_pos[0], mode="bottom"))]).astype(float).tolist()
                    pt_tmp_pos = transformer_list[v].get_pred_transform(ct_tmp_pos)
                    # print(elem, ";", v, ";", tmp_pos, pt_tmp_pos)   # 示例：1 ; 2 ; [[660, 538, 960, 807], 826] [[[7.97350411 0.88516923]]]
                    # 存储： id, cam, frame, world_pos
                    traj_pos.append([elem, tmp_pos[1], [pt_tmp_pos[0][0][0], pt_tmp_pos[0][0][1]+start_dist[v]], 1])
                    # print([elem, tr, tmp_pos[1], [pt_tmp_pos[0][0][0], pt_tmp_pos[0][0][1]+start_dist[v]], 1])
                    pass
                if v < len(infer_chain[elem])-1:
                    if int(infer_chain[elem][v+1]) > 0:
                        unmonitored_frame_begin = trace_list[v][tr]['last_frame']+1
                        unmonitored_frame_end = trace_list[v+1][infer_chain[elem][v+1]]['list'][0][1]
                        # print("unmonitored_frame_begin:", unmonitored_frame_begin, ";unmonitored_frame_end:",
                        #       unmonitored_frame_end)
                        for unmonitored_frame in range(unmonitored_frame_begin, unmonitored_frame_end):
                            # id, cam, frame, world_pos
                            traj_pos.append([elem, unmonitored_frame, [traj_pos[-1][2][0], traj_pos[-1][2][1]+(1.0/frame_rate)*seg_setting['speed'][v]], 0])



        # print("traj_pos:", traj_pos)
        traj_obj.list = traj_pos
        traj_obj.get_active_frame()

        rcd_chain[elem] = traj_obj
    return rcd_chain


def estimate_distance(trace_front=None, trace_back=None, associate_dict=None):
    """从匹配路径计算未监控区域的长度"""
    frame_rate = 25
    estimated_dist = []
    diff_frames = []
    speeds = []
    for elem in associate_dict:
        # 存在匹配的才算
        if int(associate_dict[elem]) > 0:
            first_frame = trace_front[elem]["last_frame"]
            last_frame = int(trace_back[associate_dict[elem]]["list"][0][1])
            diff_frame = last_frame - first_frame
            speed = trace_front[elem]["speed"]/3.6
            estimated_dist.append((diff_frame/frame_rate)*speed)
            diff_frames.append(diff_frame)
            speeds.append(speed)
    return np.mean(np.array(estimated_dist)), np.mean(np.array(diff_frames)), np.mean(np.array(speeds))


def get_useful_trace_data():
    pass


# 如何知道每一帧都有哪些目标？
def get_objectid_in_each_frame(trace_1=None,
                               trace_2=None,
                               trace_3=None,
                               trace_4=None,
                               assoc_dict_12=None,
                               assoc_dict_23=None,
                               assoc_dict_34=None):
    # 获取每一帧图像存在目标的编号
    # 应该按照帧号输出, 帧号没有0, 目标编号有0

    f1_in = get_in_scene_frame(trace=trace_1)
    f2_in = get_in_scene_frame(trace=trace_2)
    f3_in = get_in_scene_frame(trace=trace_3)
    f4_in = get_in_scene_frame(trace=trace_4)

    f1_out = get_out_scene_frame(trace_front=trace_1, trace_back=trace_2, assoc_dict=assoc_dict_12)
    f2_out = get_out_scene_frame(trace_front=trace_2, trace_back=trace_3, assoc_dict=assoc_dict_23)
    f3_out = get_out_scene_frame(trace_front=trace_3, trace_back=trace_4, assoc_dict=assoc_dict_34)

    return f1_in, f2_in, f3_in, f4_in, f1_out, f2_out, f3_out

def get_in_scene_frame(trace=None):
    rst_dict = {}
    for i in range(1, 3001):    # 自用
        rst_dict[i] = []        # 每帧初始化为list
    for elem in trace:          # trace的key为id
        boxlst = trace[elem]["list"]
        for bx in boxlst:
            rst_dict[bx[1]].append(elem)
    return rst_dict


def get_out_scene_frame(trace_front=None, trace_back=None, assoc_dict=None):
    rst_dict = {}
    for i in range(1, 3001):
        rst_dict[i] = []
    for elem in assoc_dict:     # elem是目标id
        # 想一下, -1,-2的标记是啥意思来， -1表示进入下一监控场,-2表示进入未监视场.
        start_frame = trace_front[elem]["last_frame"] + 1
        if int(assoc_dict[elem]) > 0:    # mapping id
            end_frame = trace_back[assoc_dict[elem]]['list'][0][1]
        else:
            end_frame = 3001
        # print("start_frame:", start_frame, ";end_frame:", end_frame)
        for i in range(start_frame, end_frame):
            rst_dict[i].append(elem)
    # print(rst_dict)
    return rst_dict


def fill_some_missing_imgs():
    """找到缺失的图像，重绘"""
    from Common import cam_names, data_path, save_path, roi_info, SCT_RESULT
    from Perspective_transform import Perspective_transformer

    device_id = 1
    transformer = Perspective_transformer(roi_info[device_id])

    filled_imgs_path = os.path.join(SCT_RESULT, 'filled')
    for i in range(1, 3001):
        fname = "{:0>4d}.jpg".format(i)
        if not os.path.exists(os.path.join(save_path[device_id], fname)):
            # print(fname)
            cv2.namedWindow("fill", cv2.WINDOW_NORMAL)
            img = cv2.imread(os.path.join(data_path[device_id], fname))
            """Draw the tracking results"""
            if isinstance(transformer.endpoints, list):
                pts_list = []
                pt_list = []
                for elem in transformer.endpoints:
                    pt_list = []
                    pt_list.append(elem)
                    pts_list.append(pt_list)
                polygonpts = np.array(pts_list).astype(int)
            else:
                pts_list = transformer.endpoints
                polygonpts = transformer.endpoints
            cv2.drawContours(img, [polygonpts], -1, (0, 0, 255), 3)
            cv2.imshow("fill", img)
            cv2.imwrite(os.path.join(filled_imgs_path, fname), img)
            cv2.waitKey(1)



def object_mapping_test():
    """测试素材完整性，对应性，PASS"""
    from Common import cam_names, track_info, associate_info, save_path

    with open(associate_info, 'r') as doc:
        associate_dict = json.load(doc)

    device_id = 2
    # trace_front = track_info[device_id]
    # trace_back = track_info[device_id+1]

    asso_dict = associate_dict[cam_names[device_id+1]]

    for elem in asso_dict:
        print("front:", elem, ";back:", asso_dict[elem])
        im_1 = cv2.imread(os.path.join(save_path[device_id], "images\\id_{:0>4d}.jpg".format(int(elem))))
        if int(asso_dict[elem]) > 0:
            im_2 = cv2.imread(os.path.join(save_path[device_id+1], "images\\id_{:0>4d}.jpg".format(int(asso_dict[elem]))))
        else:
            im_2 = np.zeros(im_1.shape, np.uint8)
        cv2.namedWindow("front", cv2.WINDOW_NORMAL)
        cv2.namedWindow("back", cv2.WINDOW_NORMAL)

        cv2.imshow("front", im_1)
        cv2.imshow("back", im_2)

        cv2.waitKey()




    pass


if __name__=="__main__":
    # # ===== TEST: Calculate Similiarity test =====
    # SimiliarityCalculateTest()
    # # ===== TEST: TrackerArray_test =====
    # TrackersArray_test()
    # # ===== TEST: Multi-cameras STP tracker test =====
    # MCT_STP_tracker_test()

    # ====== TEST: Ah~~~, Kill me!!! ======
    i_completely_forget_save_me()
    # fill_some_missing_imgs()
    # object_mapping_test()
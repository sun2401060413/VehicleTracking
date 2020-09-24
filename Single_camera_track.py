'''
SCT: Single camera tracking.
Multi-objects tracking in single camera.
written by sunzhu on 2019-03-19, version 1.0
Updated by sunzhu on 2020-08-18, version 1.1
'''

import os, sys
import pandas as pd
import cv2
import json
import numpy as np
import operator

# SETTINGS
# 1. color setting for displaying
colors = [[31, 0, 255], [0, 159, 255], [255, 95, 0], [255, 19, 0],
    [255, 0, 0], [255, 38, 0], [0, 255, 25], [255, 0, 133],
    [255, 172, 0], [108, 0, 255], [0, 82, 255], [0, 255, 6],
    [255, 0, 152], [223, 0, 255], [12, 0, 255], [0, 255, 178]]

# 2. frame rate
frame_rate = 20

# # ===== UTILS FUNCTIONS =====     
def isContactWithImage(img_height, img_width, box_x1, box_y1, box_x2, box_y2, thresh=5):
    ''' check whether the box have contact with image boundaries or not '''
    if box_x1 <= thresh or box_x2 >= img_width-thresh or box_y1 <= thresh or box_y2 >= img_height-thresh:
        return True
    else:
        return False

def IOU(box1,box2):
    '''intersection over union: IOU'''
    x1, y1, x3, y3, width1, height1 = box1[0], box1[1], box1[2], box1[3], box1[2]-box1[0], box1[3]-box1[1]
    x2, y2, x4, y4, width2, height2 = box2[0], box2[1], box2[2], box2[3], box2[2]-box2[0], box2[3]-box2[1]
    # Intersection
    i_width = width1+width2-(max(x3, x4)-min(x1, x2))
    i_height = height1+height2-(max(y3, y4)-min(y1, y2))

    if i_width <= 0 or i_height <= 0:
        IOU = 0
    else:
        i_area = i_width*i_height       # intersection area
        area1 = width1*height1
        area2 = width2*height2
        o_area = area1 + area2 - i_area # union area
        IOU = i_area*1./o_area        # intersection over union
    return IOU
    
def get_box_center(box, mode="both", ratio_x = 0.5, ratio_y = 0.5):
    '''Updated on 2020-08-19'''
    if mode == "both":
        return int((box[0]+box[2])/2), int((box[1]+box[3])/2)
    if mode == "bottom":
        return int((box[0]+box[2])/2), int(box[3])
    if mode == "ratio":
        return int(box[0] + ratio_x*abs(box[2] - box[0])), int(box[1] + ratio_y*abs(box[3] - box[1]))

def get_offset_box(box, offset=0):
    '''resize bounding box, offset refers the distance'''
    return [box[0]-offset, box[1]-offset, box[2]+offset, box[3]+offset]

# # ===== CLASS ======
class vehicle_object(object):
    def __init__(self, id):
        self.list = []
        self.image = []
        self.id = id

        self.first_box = None               # image space position
        self.first_box_ct_in_world = None   # world space position
        self.first_frame = None             # frame number
        self.first_img = None

        self.last_box = None                # image space position
        self.last_box_ct_in_world = None    # world space position
        self.last_frame = None              # frame number

        self.update_status = True
        self.speed = 0                      # speed in world space
        self.color = None
        
    def set_first_frame(self, box=None, frame=None, img=None, transformed_box_ct=None):
        self.first_box = box
        self.first_frame = frame
        self.first_img = img
        if transformed_box_ct is not None:
            self.first_box_ct_in_world = transformed_box_ct
        
    def update(self, box=None, frame=None, img=None, transformed_box_ct=None):
        self.list.append([box, frame])
        if img is not None:
            self.image.append(img)
        self.last_box = box
        self.last_frame = frame
        if transformed_box_ct is not None:
            self.last_box_ct_in_world = transformed_box_ct

        if not operator.eq(self.first_box_ct_in_world, self.last_box_ct_in_world).all():
            self.speed = self.get_avgspeed()

        
    def set_color(self, color):
        self.color = color

    def get_avgspeed(self):
        if self.last_frame != self.first_frame:
            # position instance: v1: [[[x1,y1]]], v2: [[[x2,y2]]]
            s_distance = np.linalg.norm(self.last_box_ct_in_world.flatten() - self.first_box_ct_in_world.flatten())
            t_interval = self.last_frame - self.first_frame
            self.speed = (s_distance/t_interval)*frame_rate*3.6  # km/h
            return self.speed
        else:
            return 0
         
class IOU_tracker(object):
    def __init__(self,  
                    frame_space_dist=5,
                    region_top=450,
                    region_bottom=50,
                    region_left=50,
                    region_right=50):
        # objects pool, all tracked objects are saved in this dict
        self.objects_pool = {}
        self.hist_objects_pool = {}
        self.objects_count = 0
        self.hist_objects_reord_flag = False        # Record the history information or not
        
        # threshvalue for tracking
        self.frame_space_dist = frame_space_dist    # ignore the object with long time interval
        self.region_top = region_top                # ignore the object outside the certain region
        self.region_bottom = region_bottom
        self.region_left = region_left
        self.region_right = region_right
        self.thresh_iou = 0.3                       # ignore the object with low iou
        
        # image info
        self.img_height = 1080
        self.img_width = 1920
        self.obj_pool_display_height = 100
        self.obj_pool_display_width = 100
        self.obj_pool_display_channel = 3
        
        # display setting
        self.display_monitor_region = False
        
    def get_available_id(self):
        out_put = self.objects_count
        if self.objects_count < 100000:
            self.objects_count += 1
        else:
            self.objects_count = 0
        return out_put     # The max id is 99999
        
    def get_available_color(self, id):
        i = id%len(colors)
        return colors[i]
        
    def isTrackFinish(self, frame):
        delete_obj_list = []
        for elem in self.objects_pool:
            # print(elem,frame,self.objects_pool[elem].last_frame)
            if (frame - self.objects_pool[elem].last_frame) > self.frame_space_dist:
                self.objects_pool[elem].update_status = False
                delete_obj_list.append(elem)
        delete_obj_dict = {}
        for elem in delete_obj_list:
            del_obj = self.objects_pool.pop(elem)
            delete_obj_dict[del_obj.id] = del_obj
        return delete_obj_dict
           
    def update(self, box, img=None):
        box_info = [box[0], box[1], box[2], box[3]]
        if self.isBoxInRegion(box_info):
            frame_info = box[4]
            matched_obj = self.match(box_info, frame_info)
            if matched_obj:
                self.objects_pool[matched_obj.id].update(box_info, frame_info)
            else:
                obj_id = self.get_available_id()
                obj = vehicle_object(obj_id)                # create a new vehicle object
                obj.set_first_frame(box_info, frame_info, img)
                obj.set_color(self.get_available_color(obj_id))  # set color for displaying
                obj.update(box_info, frame_info)
                self.objects_pool[obj_id] = obj
        del_objs = self.isTrackFinish(box[4])
        if self.hist_objects_reord_flag and del_objs:
            for elem in del_objs:
                self.hist_objects_pool[elem] = del_objs[elem]
        
    def match(self, box, frame):
        possible_obj_list = []
        for k, v in self.objects_pool.items():
            cmp_id = k
            cmp_locat = v.last_box
            cmp_frame = v.last_frame
            cmp_iou = IOU(box, cmp_locat)
            cmp_frame_dist = frame-cmp_frame
            if cmp_iou >= self.thresh_iou and cmp_frame_dist <= self.frame_space_dist:
                possible_obj_list.append([v, cmp_iou, cmp_frame_dist*1./self.frame_space_dist])
        matched_obj = self.rank(possible_obj_list, 0.6, 0.4)
        return matched_obj
        
    def rank(self, objs_list, weight_spatial=0.5, weight_temporal=0.5):
        ''' find the nearest object in spatial and temporal space, the default weights of two space are 0.5 and 0.5 '''
        if objs_list == []:
            return None
        else:
            def takeSecond(elem):
                return elem[1]
            dist_list = []
            for elem in objs_list:
                dist = elem[1]*weight_spatial+elem[2]*weight_temporal
                dist_list.append([objs_list[0], dist])
            
            dist_list.sort(key=takeSecond)
            return dist_list[-1][0][0]
        
    def isBoxInRegion(self, box):
        '''Check a object in the setting region or not
            Note: In vertical direction, we take the box bottom as a reference to check the present of object.
        '''
        if box[0] > self.region_left and box[2] < self.img_width - self.region_right and box[3] > self.region_top and box[3] < self.img_height - self.region_bottom:
            return True
        else:
            return False
      
    def draw_trajectory(self, img):
        if self.display_monitor_region:
            cv2.line(img, (0, self.region_top), (img.shape[1]-1, self.region_top), (255, 255, 255), 5)
            cv2.line(img, (0, img.shape[0]-self.region_bottom), (img.shape[1]-1, img.shape[0]-self.region_bottom), (255, 255, 255), 5)
        for k, v in self.objects_pool.items():
            if v.update_status:
                if len(v.list) > 1:
                    for i in range(len(v.list)-1):
                        center_1 = get_box_center(v.list[i][0], mode="bottom")
                        center_2 = get_box_center(v.list[i+1][0], mode="bottom")
                        cv2.line(img, center_1, center_2, v.color, 5)
                    cv2.putText(img, "ID:{}".format(v.id), (v.last_box[2], v.last_box[3])
                    , cv2.FONT_HERSHEY_COMPLEX, 2, v.color, 5)
        return img
        
    def draw_objects_pool(self):
        if len(self.objects_pool)>0:
            img_height = self.obj_pool_display_height
            img_width = self.obj_pool_display_width*len(self.objects_pool)
            disp_objs_pool_img = np.zeros((img_width, img_height, self.obj_pool_display_channel), np.uint8)
            obj_count = 0
            for k, v in self.objects_pool.items():
                chosen_img = cv2.resize(v.first_img, (self.obj_pool_display_width, self.obj_pool_display_height))
                disp_objs_pool_img[ self.obj_pool_display_width*obj_count:self.obj_pool_display_width*(obj_count+1), 0:self.obj_pool_display_height] = chosen_img
                cv2.putText(disp_objs_pool_img, "ID:{}".format(v.id), (0, self.obj_pool_display_height*(obj_count+1)-3), cv2.FONT_HERSHEY_SIMPLEX, 1, v.color, 2)
                obj_count += 1
            return disp_objs_pool_img
        else:
            return None
        
    def save_data(self, filepath):
        if self.hist_objects_reord_flag:
            filename = r"tracking_info.json"
            saved_info = {}
            for elem in self.hist_objects_pool:
                tmp = {}
                tmp['list'] = [[[int(v[0][0]), int(v[0][1]), int(v[0][2]), int(v[0][3])], int(v[1])] for v in self.hist_objects_pool[elem].list]
                tmp['id'] = self.hist_objects_pool[elem].id
                tmp['last_box'] = [int(v) for v in self.hist_objects_pool[elem].last_box]
                tmp['last_frame'] = int(self.hist_objects_pool[elem].last_frame)
                tmp['update_status'] = self.hist_objects_pool[elem].update_status
                tmp['color'] = self.hist_objects_pool[elem].color
                saved_info[elem] = tmp

            with open(os.path.join(filepath, filename), 'w') as doc:
                json.dump(saved_info, doc)
        else:
            return print("History record flag is False!")

class IOU_tracker_with_polygon_region(object):
    def __init__(self,
                 frame_space_dist=10,
                 transformer=None,
                 ):
        # objects pool, all tracked objects are saved in this dict
        self.objects_pool = {}
        self.hist_objects_pool = {}
        self.objects_count = 0
        self.hist_objects_record_flag = True     # Record the history information or not
        self.image_record_flag = True           # save image or not

        # threshvalue for tracking
        self.frame_space_dist = frame_space_dist  # ignore the object with long time interval
        self.thresh_iou = 0.2  # ignore the object with low iou

        # image info
        self.img_height = 1080
        self.img_width = 1920
        self.obj_pool_display_height = 100      # objects_pool display height;
        self.obj_pool_display_width = 100       # objects_pool display width;
        self.obj_pool_display_channel = 3

        # display setting
        self.display_monitor_region = True

        # Coodinate transformer
        self.transformer = transformer
        self.polygonpts = self.set_polygon_region(transformer.endpoints)

    def get_available_id(self):
        out_put = self.objects_count
        if self.objects_count < 100000:
            self.objects_count += 1
        else:
            self.reset_available_id()       # reset the count
        return out_put  # The max id is 99999

    def reset_available_id(self):
        self.objects_count = 0

    def get_available_color(self, id):
        i = id % len(colors)
        return colors[i]

    def isTrackFinish(self, frame):
        '''Check that the object is still in monitoring region'''
        delete_obj_list = []
        for elem in self.objects_pool:
            # print(elem,frame,self.objects_pool[elem].last_frame)
            if (frame - self.objects_pool[elem].last_frame) > self.frame_space_dist:
                self.objects_pool[elem].update_status = False
                delete_obj_list.append(elem)
        delete_obj_dict = {}
        for elem in delete_obj_list:
            del_obj = self.objects_pool.pop(elem)
            delete_obj_dict[del_obj.id] = del_obj
        return delete_obj_dict

    def update(self, box, img=None):
        box_info = [box[0], box[1], box[2], box[3]]
        # if self.isBoxInRegion(box_info):
        box_ct = get_box_center(box, mode="ratio", ratio_x=0.5, ratio_y=0.75)

        input_box_ct = np.array([list(box_ct)]).astype(float).tolist()

        # transformed point
        if self.transformer is not None:
            cur_pos = self.transformer.get_pred_transform(input_box_ct)
        else:
            cur_pos = None

        if self.isBoxInPolygonRegion(box_ct) > 0:
            frame_info = box[4]

            matched_obj = self.match(box_info, frame_info)
            if matched_obj:
                self.objects_pool[matched_obj.id].update(box=box_info, frame=frame_info, transformed_box_ct=cur_pos)
            else:
                obj_id = self.get_available_id()
                obj = vehicle_object(obj_id)  # create a new vehicle object
                obj.set_first_frame(box=box_info, frame=frame_info, img=img, transformed_box_ct=cur_pos)
                obj.set_color(self.get_available_color(obj_id))     # set color for displaying
                obj.update(box=box_info, frame=frame_info, transformed_box_ct=cur_pos)
                self.objects_pool[obj_id] = obj
        del_objs = self.isTrackFinish(box[4])
        if self.hist_objects_record_flag and del_objs:
            for elem in del_objs:
                self.hist_objects_pool[elem] = del_objs[elem]

    def match(self, box, frame):
        possible_obj_list = []
        for k, v in self.objects_pool.items():
            cmp_id = k
            cmp_locat = v.last_box
            cmp_frame = v.last_frame
            cmp_iou = IOU(box, cmp_locat)
            cmp_frame_dist = frame - cmp_frame
            if cmp_iou >= self.thresh_iou and cmp_frame_dist <= self.frame_space_dist:
                possible_obj_list.append([v, cmp_iou, cmp_frame_dist * 1. / self.frame_space_dist])
        matched_obj = self.rank(possible_obj_list, 0.6, 0.4)
        return matched_obj

    def rank(self, objs_list, weight_spatial=0.5, weight_temporal=0.5):
        ''' find the nearest object in spatial and temporal space, the default weights of two space are 0.5 and 0.5 '''
        if objs_list == []:
            return None
        else:
            def takeSecond(elem):
                return elem[1]

            dist_list = []
            for elem in objs_list:
                dist = elem[1] * weight_spatial + elem[2] * weight_temporal
                dist_list.append([objs_list[0], dist])

            dist_list.sort(key=takeSecond)
            return dist_list[-1][0][0]

    def isBoxInRegion(self, box):
        '''Check a object in the setting region or not
            Note: In vertical direction, we take the box bottom as a reference to check the present of object.
        '''
        if box[0] > self.region_left and box[2] < self.img_width - self.region_right and box[3] > self.region_top and \
                box[3] < self.img_height - self.region_bottom:
            return True
        else:
            return False

    def isBoxInPolygonRegion(self, box):
        '''Check a object in the setting region or not
            Note: the region is a polygon.
        '''
        pt = box
        return cv2.pointPolygonTest(self.polygonpts, pt, measureDist=False)

    def set_polygon_region(self, pts):
        '''[[a,b],[c,d],[e,f],[g,h]] → [np.array([[a,b]],[[c,d]]....)]'''
        if isinstance(pts[0], list):
            pts_list = []
            pt_list = []
            for elem in pts:
                pt_list = []
                pt_list.append(elem)
                pts_list.append(pt_list)
            self.polygonpts = np.array(pts_list).astype(int)
        else:
            pts_list = pts
            self.polygonpts = pts

        return self.polygonpts



    def draw_trajectory(self, img):
        if self.display_monitor_region:
            # cv2.line(img, (0, self.region_top), (img.shape[1] - 1, self.region_top), (255, 255, 255), 5)
            # cv2.line(img, (0, img.shape[0] - self.region_bottom), (img.shape[1] - 1, img.shape[0] - self.region_bottom),
            #          (255, 255, 255), 5)
            cv2.drawContours(img, [self.polygonpts], -1, (0, 0, 255), 3)
        for k, v in self.objects_pool.items():  # draw all objects in the pool.
            if v.update_status:
                if len(v.list) > 1:
                    for i in range(len(v.list) - 1):
                        # center_1 = get_box_center(v.list[i][0], mode="bottom")
                        # center_2 = get_box_center(v.list[i + 1][0], mode="bottom")
                        center_1 = get_box_center(v.list[i][0], mode="ratio", ratio_x=0.5, ratio_y=0.75)
                        center_2 = get_box_center(v.list[i + 1][0], mode="ratio", ratio_x=0.5, ratio_y=0.75)
                        cv2.line(img, center_1, center_2, v.color, 5)
                    cv2.putText(img, "ID:{}".format(v.id), (v.last_box[2], v.last_box[3])
                                , cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, v.color, 3)
                    print("v.speed", v.speed)
                    cv2.putText(img, "{}km/h".format(round(v.speed), 1), (v.last_box[2], v.last_box[3]+30)
                                , cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, v.color, 3)
        return img

    def draw_objects_pool(self):
        if len(self.objects_pool) > 0:
            img_height = self.obj_pool_display_height
            img_width = self.obj_pool_display_width * len(self.objects_pool)
            disp_objs_pool_img = np.zeros((img_width, img_height, self.obj_pool_display_channel), np.uint8)
            obj_count = 0
            for k, v in self.objects_pool.items():
                chosen_img = cv2.resize(v.first_img, (self.obj_pool_display_width, self.obj_pool_display_height))
                disp_objs_pool_img[
                self.obj_pool_display_width * obj_count:self.obj_pool_display_width * (obj_count + 1),
                0:self.obj_pool_display_height] = chosen_img
                cv2.putText(disp_objs_pool_img, "ID:{}".format(v.id),
                            (0, self.obj_pool_display_height * (obj_count + 1) - 3), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            v.color, 2)
                # cv2.putText(disp_objs_pool_img, "{}km/h".format(round(v.speed, 1)),
                #             (0, self.obj_pool_display_height * (obj_count + 1) - 3 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                #             v.color, 2)
                obj_count += 1
            return disp_objs_pool_img
        else:
            return None

    def save_data(self, filepath):
        if self.hist_objects_record_flag:
            filename = r"tracking_info.json"
            saved_info = {}
            for elem in self.hist_objects_pool:
                tmp = {}
                # tmp['list'] = [[[int(v[0][0]), int(v[0][1]), int(v[0][2]), int(v[0][3])], int(v[1])] for v in
                #                self.hist_objects_pool[elem].list]
                tmp['id'] = self.hist_objects_pool[elem].id
                tmp['last_box'] = [int(v) for v in self.hist_objects_pool[elem].last_box]
                tmp['last_frame'] = int(self.hist_objects_pool[elem].last_frame)
                tmp['update_status'] = self.hist_objects_pool[elem].update_status
                tmp['color'] = self.hist_objects_pool[elem].color
                tmp['speed'] = self.hist_objects_pool[elem].speed
                saved_info[elem] = tmp

            with open(os.path.join(filepath, filename), 'w') as doc:
                json.dump(saved_info, doc)
        else:
            return print("History record flag is False!")


class STP_tracker(object):
    def __init__(   self,  
                    frame_space_dist=50,
                    region_top=450,
                    region_bottom=50,
                    region_left=50,
                    region_right=50,
                    obj_STP=None,
                    match_mode = 'Prob'):   # Mode:Prob/Dist
        # objects pool, all tracked objects are saved in this dict
        self.objects_pool = {}
        self.objects_count = 0
        
        self.last_append_id = None
        self.new_object_append_status = False
        
        # threshvalue for tracking
        self.frame_space_dist = frame_space_dist    # ignore the object with long time interval
        self.region_top = region_top                # ignore the object outside the certain region
        self.region_bottom = region_bottom
        self.region_left = region_left
        self.region_right = region_right
        self.thresh_probability = 0.001      # ignore the object with low probability
        self.thresh_distance = 2            # ignore the object with far distance   
                                            # the value dependent on frame_space_dist
        
        # Single Camera Spatial-temporal prior
        self.obj_STP = obj_STP
        self.match_mode = 'Prob'
        
        # image info
        self.img_height = 1080
        self.img_width = 1920
        self.obj_pool_display_height = 100
        self.obj_pool_display_width = 100
        self.obj_pool_display_channel = 3
        
        # display setting
        self.display_monitor_region = False
        
    def version(self):
        return print("===== Written by sunzhu, 2019-03-19, Version 1.0 =====")
    
    def get_available_id(self):
        out_put = self.objects_count
        if self.objects_count < 100000:
            self.objects_count += 1
        else:
            self.objects_count = 0
        return out_put     # The max id is 99999
        
    def get_available_color(self, id):
        i = id % len(colors)
        return colors[i]
        
    def isTrackFinish(self, frame):
        del_id_list = []
        for elem in self.objects_pool:
            # print(elem,frame,self.objects_pool[elem].last_frame)
            if (frame - self.objects_pool[elem].last_frame) > self.frame_space_dist or self.objects_pool[elem].update_status == False:
                self.objects_pool[elem].update_status = False
                del_id_list.append(elem)
                
        # can not delete dict key-value in iter
        del_id_dict = {}
        for elem in del_id_list:
            pop_obj = self.objects_pool.pop(elem)   # Delete elem in objects_pool
            del_id_dict[pop_obj.id] = pop_obj
        return del_id_dict
        
    def update(self, box, img=None):
        if len(box) == 1:     # Consider the conditions that the box is empty
            return self.isTrackFinish(box[0]), self.objects_pool
        box_info = [box[0], box[1], box[2], box[3]]
        self.new_object_append_status = False
        if self.isBoxInRegion(box_info) == 0:
            frame_info = box[4]
            matched_obj = self.match(box_info, frame_info)
            # print(matched_obj)
            if matched_obj:
                self.objects_pool[matched_obj.id].update(box_info, frame_info)
            else:
                obj_id = self.get_available_id()
                obj = vehicle_object(obj_id)                # create a new vehicle object
                self.last_append_id = obj_id
                self.new_object_append_status = True
                obj.set_first_frame(box=box_info, frame=frame_info, img=img)
                obj.set_color(self.get_available_color(obj_id))  # set color for displaying
                obj.update(box_info, frame_info)
                self.objects_pool[obj_id] = obj
        if self.isBoxInRegion(box_info) == 1:
            frame_info = box[4]
            matched_obj = self.match(box_info, frame_info)
            if matched_obj:
                self.objects_pool[matched_obj.id].update_status = False
        return self.isTrackFinish(box[4]), self.objects_pool
        
    def match(self, box, frame):
        possible_obj_list = []
        
        # # root path
        # # dataset_root = r"E:\DataSet\trajectory\concatVD\wuqi2B"
        # # img_current = cv2.imread(os.path.join(dataset_root,str(frame)+'.jpg'))
        # # if img_current is not None:
            # # # print(img_current.shape)
            # # pass
        # # else:
            # # return
        # obj_img = img_current[box[1]:box[3],box[0]:box[2]].copy()
        # # print(obj_img.shape)
        # # cv2.namedWindow('img_current',cv2.WINDOW_NORMAL)
        # cv2.imshow("obj_img",obj_img)
        for k,v in self.objects_pool.items():
            cmp_id = k
            cmp_locat = v.last_box
            cmp_frame = v.last_frame
            center_x, center_y = get_box_center(box, mode='bottom')
            base_center_x, base_center_y = get_box_center(cmp_locat, mode='bottom')
            # print("---------")
            # print("cmp_id",cmp_id)
            pt_centers = self.obj_STP.perspective_transformer.get_pred_transform(
                    np.array(
                        [[center_x, center_y],
                        [base_center_x, base_center_y]]
                        , np.float)
                    )
            pt_center_x, pt_center_y = pt_centers[0][0]
            pt_base_center_x, pt_base_center_y = pt_centers[0][1]
            
            # # ==== TEST: Display the probability map =====
            # img_3 = self.draw_color_probability_map(img_current,pt_base_center_x,pt_base_center_y)
            # cv2.namedWindow("img_current",cv2.WINDOW_NORMAL)
            # cv2.imshow("img_current",img_3)
            # cv2.waitKey(1)
            
            cmp_frame_dist = frame-cmp_frame
            if self.match_mode == 'Prob':
                cmp_result = self.obj_STP.get_probability(pt_center_x, pt_center_y, pt_base_center_x, pt_base_center_y)[2]
                if cmp_result >= self.thresh_probability and cmp_frame_dist <= self.frame_space_dist:
                    possible_obj_list.append([v, cmp_result, cmp_frame_dist*1./self.frame_space_dist])
            else:   # Dist mode
                cmp_result = self.obj_STP.get_distance(pt_center_x, pt_center_y, pt_base_center_x, pt_base_center_y)
                if cmp_result <= self.thresh_distance and cmp_frame_dist <= self.frame_space_dist:
                    possible_obj_list.append([v, cmp_result, cmp_frame_dist*1./self.frame_space_dist])
        matched_obj = self.rank(possible_obj_list)

        return matched_obj
        
    def rank(self, objs_list):
        '''find the nearest object in spatial and temporal space, the default weights of two space are 0.5 and 0.5'''
        if objs_list == []:
            return None
        else:
            def takeSecond(elem):
                return elem[1]
            dist_list = []
            for elem in objs_list:
                dist = elem[1]
                dist_list.append([objs_list[0], dist])
            
            dist_list.sort(key=takeSecond)
            if self.match_mode == 'Prob':
                return dist_list[-1][0][0]
            else:
                return dist_list[0][0][0]
        
    def isBoxInRegion(self, box):
        ''' Check a object in the setting region or not
            Note: In vertical direction, we take the box bottom as a reference to check the present of object.
        '''
        if box[0]>self.region_left and box[2]<self.img_width-self.region_right and box[3]>self.region_top and box[3]<self.img_height-self.region_bottom:
            return 0
        elif box[3]<self.region_top:
            return 1
        elif box[3]>self.img_height-self.region_bottom:
            return -1
      
    def draw_trajectory(self, img):
        self.draw_objects_pool()
        if self.display_monitor_region:
            cv2.line(img, (0, self.region_top), (img.shape[1]-1, self.region_top), (255, 255, 255), 5)
            cv2.line(img, (0, img.shape[0]-self.region_bottom), (img.shape[1]-1, img.shape[0]-self.region_bottom), (255, 255, 255), 5)
        for k,v in self.objects_pool.items():
            if v.update_status:
                if len(v.list) > 0:
                    for i in range(len(v.list)-1):
                        center_1 = get_box_center(v.list[i][0], mode="bottom")
                        center_2 = get_box_center(v.list[i+1][0], mode="bottom")
                        cv2.line(img, center_1, center_2, v.color, 5)
                    cv2.putText(img, "ID:{}".format(v.id), (v.last_box[2], v.last_box[3])
                    ,cv2.FONT_HERSHEY_COMPLEX, 2, v.color, 5)
        return img
            
    def draw_objects_pool(self, mode='h', set_range=0):
        from Draw_trajectory import draw_objects_pool
        return draw_objects_pool(self.objects_pool, self.obj_pool_display_height, self.obj_pool_display_width, self.obj_pool_display_channel,mode=mode,set_range=set_range)
            # # if len(self.objects_pool)>0:
                # # img_height = self.obj_pool_display_height
                # # if set_range > 0:
                    # # img_width = max(set_range,self.obj_pool_display_width*len(self.objects_pool))
                # # else:
                    # # img_width = self.obj_pool_display_width*len(self.objects_pool)
                # # disp_objs_pool_img = np.zeros((img_height,img_width,self.obj_pool_display_channel),np.uint8)
                # # obj_count = 0
                # # for k,v in self.objects_pool.items():
                    # # chosen_img = cv2.resize(v.first_img,(self.obj_pool_display_width,self.obj_pool_display_height))
                    # # disp_objs_pool_img[ 0:self.obj_pool_display_height,obj_count*self.obj_pool_display_width:(obj_count+1)*self.obj_pool_display_width] = chosen_img
                    # # cv2.putText(disp_objs_pool_img,"ID:{}".format(v.id),(self.obj_pool_display_width*(obj_count),self.obj_pool_display_height-3),cv2.FONT_HERSHEY_SIMPLEX,1,v.color,2)
                    # # obj_count += 1
                # # return disp_objs_pool_img
            # # else:
                # # return None
            
    def draw_color_probability_map(self,img_current,pt_base_center_x,pt_base_center_y,alpha=0.5):
        # probability color map
        p_map = self.obj_STP.get_probability_map(base_x=pt_base_center_x,base_y=pt_base_center_y)
        p_map = cv2.applyColorMap(p_map,cv2.COLORMAP_JET)
        color_p_map = cv2.resize(p_map,(int(self.obj_STP.perspective_transformer.transformed_width_for_disp),int(self.obj_STP.perspective_transformer.transformed_height_for_disp)))
        color_p_map = cv2.flip(color_p_map,0)   # 0:vertical flip
        pt_color_p_map = self.obj_STP.perspective_transformer.get_inverse_disp_transform(color_p_map)

        img_3 = cv2.addWeighted(img_current, alpha, pt_color_p_map, 1-alpha, 0)

        return img_3
            
    def save_data(self,filepath):
        filename = r"tracking_info.json"
        saved_info = {}
        for elem in self.objects_pool:
            tmp = {}
            tmp['list'] = [[[int(v[0][0]),int(v[0][1]),int(v[0][2]),int(v[0][3])],int(v[1])] for v in self.objects_pool[elem].list]
            tmp['id'] = self.objects_pool[elem].id
            tmp['last_box'] = [int(v) for v in self.objects_pool[elem].last_box]
            tmp['last_frame'] = int(self.objects_pool[elem].last_frame)
            tmp['update_status'] = self.objects_pool[elem].update_status
            tmp['color'] = self.objects_pool[elem].color
            saved_info[elem] = tmp
        with open(os.path.join(filepath,filename),'w') as doc:
            json.dump(saved_info, doc)
            
# # ====== TEST FUNCTIONS =====

def IOU_tracker_test():
    # root path
    dataset_root = r"E:\DataSet\trajectory\concatVD"

    # save root
    save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data'

    # images
    cam = ['wuqi1B', 'wuqi2B', 'wuqi3B', 'wuqi4B']

    # bbox information
    box_info = ['wuqiyuce1.csv', 'wuqiyuce2.csv', 'wuqiyuce3.csv', 'wuqiyuce4.csv']

    # cam id
    device_id = '1'

    tracker = IOU_tracker(region_top=450)
    # tracker = STP_tracker()
    tracker.display_monitor_region = True
    # tracker.hist_objects_record_flag = True
    tracker.hist_objects_record_flag = False

    img_filepath = os.path.join(dataset_root, cam[int(device_id) - 1])
    img_savepath = os.path.join(save_root, cam[int(device_id) - 1])
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)

    # data = pd.read_csv(os.path.join(dataset_root,box_info_1))
    data = pd.read_csv(os.path.join(dataset_root, box_info[int(device_id) - 1]))
    prev_fps = -1
    id_in_frame = 0

    for index, row in data[["x1", "y1", "x2", "y2", "fps"]].iterrows():
        # filename
        filename = str(row['fps']) + '.jpg'
        # img_filename = os.path.join(img_filepath_1,filename)
        img_filename = os.path.join(img_filepath, filename)
        img = cv2.imread(img_filename)
        cp_img = img[row['y1']:row['y2'], row['x1']:row['x2']].copy()  # copy
        tracker.update([row['x1'], row['y1'], row['x2'], row['y2'], row['fps']], cp_img)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        traj_img = tracker.draw_trajectory(img)
        obj_pool_img = tracker.draw_objects_pool()
        if traj_img is not None:
            cv2.imshow('img', traj_img)
        if obj_pool_img is not None:
            cv2.imshow('obj_pool', obj_pool_img)
        cv2.waitKey(1)

    tracker.save_data(os.path.join(save_root, cam[int(device_id) - 1]))


def IOU_tracker_test_polygon_region():
    # root path
    dataset_root = r"E:\Project\CV\Data\image"

    # save root
    save_root = r'E:\Project\CV\Data\results\saved_0831\saved_results'

    # images
    # cam = ['wuqi1B', 'wuqi2B', 'wuqi3B', 'wuqi4B']
    cam = ['00M35S_1589990435',
           '00M35S_1589994035',
           '00M35S_1590001235',
           '59M35S_1590004775',
           '59M35S_1590008375',
           '59M35S_1590011975',
           '59M35S_1590015575',
           '59M35S_1590019175',
           '59M35S_1590022775',
           'source_img_0800']

    # bbox information
    # box_info = ['wuqiyuce1.csv', 'wuqiyuce2.csv', 'wuqiyuce3.csv', 'wuqiyuce4.csv']
    box_info = ['00M35S_1589990435.csv',
                '00M35S_1589994035.csv',
                '00M35S_1590001235.csv',
                '59M35S_1590004775.csv',
                '59M35S_1590008375.csv',
                '59M35S_1590011975.csv',
                '59M35S_1590015575.csv',
                '59M35S_1590019175.csv',
                '59M35S_1590022775.csv',
                'boxInfo20200812.csv']

    # cam id
    device_id = '2'

    # MonitoringRegion
    json_path = r"E:\Project\CV\Data\settings\0002_transformer.json"
    # info_dict = {}
    # with open(json_path, 'r') as doc:
    #     info_dict = json.load(doc)

    import Perspective_transform
    Pt_transformer = Perspective_transform.Perspective_transformer(json_path)


    # tracker = IOU_tracker(region_top=450)
    # tracker = IOU_tracker_with_polygon_region(polygon_pts=info_dict["endpoints"])
    tracker = IOU_tracker_with_polygon_region( transformer=Pt_transformer )
    # tracker = STP_tracker()
    tracker.display_monitor_region = True
    tracker.hist_objects_record_flag = True
    tracker.image_record_flag = True

    img_filepath = os.path.join(dataset_root, cam[int(device_id) - 1])
    img_savepath = os.path.join(save_root, cam[int(device_id) - 1])
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)

    # data = pd.read_csv(os.path.join(dataset_root,box_info_1))
    data = pd.read_csv(os.path.join(dataset_root, box_info[int(device_id) - 1]), dtype={'x1': int, 'y1': int, 'x2': int, 'y2': int, 'fps': int})
    prev_fps = -1
    id_in_frame = 0

    for index, row in data[["x1", "y1", "x2", "y2", "fps"]].iterrows():
        # filename
        # check the object is in the region

        filename = str(row['fps']).zfill(4) + '.jpg'
        # img_filename = os.path.join(img_filepath_1,filename)
        img_filename = os.path.join(img_filepath, filename)
        img = cv2.imread(img_filename)
        cp_img = img[row['y1']:row['y2'], row['x1']:row['x2']].copy()  # copy
        tracker.update([row['x1'], row['y1'], row['x2'], row['y2'], row['fps']], cp_img)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        traj_img = tracker.draw_trajectory(img)         # trajectory lines
        obj_pool_img = tracker.draw_objects_pool()      # objects imgs

        if traj_img is not None:
            cv2.imshow('img', traj_img)
            if tracker.image_record_flag:
                cv2.imwrite(os.path.join(img_savepath, filename), traj_img)
        if obj_pool_img is not None:
            cv2.imshow('obj_pool', obj_pool_img)
        cv2.waitKey(1)

    tracker.save_data(os.path.join(save_root, cam[int(device_id) - 1]))

def STP_tracker_test():
    # root path
    dataset_root = r"E:\DataSet\trajectory\concatVD"
    
    # root path for tracking information
    tracking_root = r"D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data"

    # save root
    save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data\resutls\single_stp_tracking'

    # json file for perspective transformation
    pt_trans_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\ROItools\data'
    
    # images
    cam = ['wuqi1B', 'wuqi2B', 'wuqi3B', 'wuqi4B']

    # bbox information
    box_info = ['wuqiyuce1.csv', 'wuqiyuce2.csv', 'wuqiyuce3.csv', 'wuqiyuce4.csv']

    # device_info
    device_info = [
        ['wuqi1B', 'wuqiyuce1.csv', 'ROI_cam_1_transformer.json'],
        ['wuqi2B', 'wuqiyuce2.csv', 'ROI_cam_2_transformer.json'],
        ['wuqi3B', 'wuqiyuce3.csv', 'ROI_cam_3_transformer.json'],
        ['wuqi4B', 'wuqiyuce4.csv', 'ROI_cam_4_transformer.json'],
    ]
    # test cam
    device_id = 1
    
    # file path
    img_filepath = os.path.join(dataset_root, device_info[device_id][0])
    tracking_info_filepath = os.path.join(tracking_root, device_info[device_id][0])
    img_savepath = os.path.join(save_root, device_info[device_id][0])
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)
    pt_savepath = os.path.join(pt_trans_root, device_info[device_id][2])
    
    time_interval = 5
    
    trace_record = []
    
    from data_generator import load_tracking_info
    tracker_record = load_tracking_info(tracking_info_filepath)
    
    from Perspective_transform import Perspective_transformer
    pt_obj = Perspective_transformer(pt_savepath)
    
    from cameras_associate import Single_camera_STP
    STP_Predictor = Single_camera_STP(
                        tracker_record,
                        pt_obj,
                        time_interval = time_interval)
    STP_Predictor.var_beta_x = int(25*25/time_interval)
    STP_Predictor.var_beta_y = int(25/time_interval)+1
    STP_Predictor.update_predictor()
    
    tracker = STP_tracker(frame_space_dist=50, obj_STP=STP_Predictor)
    tracker.match_mode = 'Prob'
    tracker.display_monitor_region = True
    tracker.version()
    
    data = pd.read_csv(os.path.join(dataset_root, device_info[device_id][1]))
    # data = pd.read_csv(os.path.join(dataset_root,box_info_2))
    prev_fps = -1
    id_in_frame = 0
    for index, row in data[["x1", "y1", "x2", "y2", "fps"]].iterrows():
        if row['fps']%time_interval == 0:
            # filename
            filename = str(row['fps'])+'.jpg'
            # img_filename = os.path.join(img_filepath_1,filename)
            img_filename = os.path.join(img_filepath, filename)
            img = cv2.imread(img_filename)
            cp_img = img[row['y1']:row['y2'], row['x1']:row['x2']].copy()
            
            tracker.update([row['x1'], row['y1'], row['x2'], row['y2'], row['fps']], cp_img)
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            tray_img = tracker.draw_trajectory(img)
            obj_pool_img = tracker.draw_objects_pool()
            if tray_img is not None:
                cv2.imshow('img', tray_img)
            if obj_pool_img is not None:
                cv2.imshow('obj_pool_img', obj_pool_img)
            cv2.imwrite(os.path.join(save_root, filename), tray_img)
            cv2.waitKey(1)
    tracker.save_data(os.path.join(save_root, device_info[device_id][0]))


def TempTest():
    # MonitoringRegion
    # json_path = r"E:\Project\CV\Data\settings\0001_transformer.json"
    # with open(json_path, 'r') as doc:
    #     info_dict = json.load(doc)
    # print(info_dict)
    img = cv2.imread(r"E:\Project\CV\Data\timg.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, thresh = cv2.threshold(img, 200, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [np.array([[[0, 0]], [[0, 100]], [[120, 100]], [[120, 0]]])]

    print(cv2.pointPolygonTest(contours[0], (50, 50), measureDist=False))
    #
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    #
    # cv2.namedWindow("thresh", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    #
    # cv2.imshow("thresh", thresh)
    # cv2.imshow("test", img)
    #
    # cv2.waitKey()
    pass

if __name__=="__main__":
    
    # ===== TEST:IOU_tracker test =====
    # IOU_tracker_test()

    # ===== TEST:STP_tracker test =====
    # STP_tracker_test()

    # ===== TEST:TEMP =====
    # TempTest()

    # ==== TEST:IOU_tracker_test_polygon_region test ====
    IOU_tracker_test_polygon_region()
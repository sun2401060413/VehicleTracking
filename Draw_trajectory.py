'''
Draw trajectory
Draw trajectory of multi-objects on multi-cameras
Written by sunzhu, 2019-04-09, version 1.0
'''

import os
import cv2
import numpy as np

# ===== CONFIGURATION =====
from Single_camera_track import colors
from Single_camera_track import get_box_center

vehicle_shape = {
    'car':(2,5),
    'truck':(2.5,10),
    'bus':(2.5,12),
}

device_info = {
    0:"YK88+565",
    1:"YK88+715"
}

# ===== UTILS FUNCITONS =====

def draw_objects_pool(objects_pool,set_height,set_width,set_channel,mode='v',set_range=0):
    if mode == 'v': # mode = v: vertical
        if len(objects_pool)>0:
            img_width = set_width
            if set_range > 0:
                img_height = max(set_range,set_height*len(objects_pool))
            else:
                img_height = set_height*len(objects_pool)
            disp_objs_pool_img = np.zeros((img_height,img_width,set_channel),np.uint8)
            
            obj_count = 0
            for k,v in objects_pool.items():
                chosen_img = cv2.resize(v.first_img,(set_width,set_height))
                disp_objs_pool_img[ set_height*obj_count:set_height*(obj_count+1),0:set_width] = chosen_img
                cv2.putText(disp_objs_pool_img,"ID:{}".format(v.id),(0,set_height*(obj_count+1)-3),cv2.FONT_HERSHEY_SIMPLEX,1,v.color,2)
                obj_count=obj_count+1
            return disp_objs_pool_img
        else:
            return None
    else:   # mode = h
        if len(objects_pool)>0:
            img_height = set_height
            if set_range > 0:
                img_width = max(set_range,set_width*len(objects_pool))
            else:
                img_width = set_width*len(objects_pool)
            disp_objs_pool_img = np.zeros((img_height,img_width,set_channel),np.uint8)
            obj_count = 0
            for k,v in objects_pool.items():
                chosen_img = cv2.resize(v.first_img,(set_width,set_height))
                disp_objs_pool_img[ 0:set_height,obj_count*set_width:(obj_count+1)*set_width] = chosen_img
                cv2.putText(disp_objs_pool_img,"ID:{}".format(v.id),(set_width*(obj_count),set_height-3),cv2.FONT_HERSHEY_SIMPLEX,1,v.color,2)
                obj_count += 1
            return disp_objs_pool_img
        else:
            return None
    
def draw_all_results(img_dict,s_obj_dict,m_obj_dict,disp_img,deveice_id_list,img_height,img_width,obj_range,disp_height):
   
    device_count = len(deveice_id_list)
    # Create canvas
    canvas_img_height = img_height + obj_range + disp_height
    canvas_img_width = img_width*device_count + obj_range*(device_count-1)
    canvas_img = np.zeros((canvas_img_height,canvas_img_width,3),np.uint8)
    
    # Draw images
    img_count = 0
    for elem in deveice_id_list:
        if img_dict.__contains__(elem):
            resize_img = cv2.resize(img_dict[elem],(img_width,img_height))
            canvas_img[0:img_height,img_count*(img_width+obj_range):img_count*(img_width+obj_range)+img_width] = resize_img
        if s_obj_dict.__contains__(elem) and s_obj_dict[elem] is not None:
            resize_s_obj_img = cv2.resize(s_obj_dict[elem],(img_width,obj_range))
            canvas_img[img_height:img_height+obj_range,img_count*(img_width+obj_range):img_count*(img_width+obj_range)+img_width] = resize_s_obj_img
        if m_obj_dict.__contains__(elem) and m_obj_dict[elem] is not None:
            resize_m_obj_img = cv2.resize(m_obj_dict[elem],(obj_range,img_height))
            canvas_img[0:img_height,img_count*(img_width+obj_range)+img_width:img_count*(img_width+obj_range)+img_width+obj_range] = resize_m_obj_img
        img_count+=1
    resize_disp_img = cv2.resize(disp_img,(canvas_img_width,disp_height))
    canvas_img[img_height+obj_range:canvas_img_height] = resize_disp_img
    return canvas_img

def get_img_shape_in_dict(img_dict):
    img_height,img_width = None,None
    for elem in img_dict:
        if img_height is None:
            img_height,img_width,_= img_dict[elem].shape
        else:
            break
    return img_height,img_width
    
def draw_objects_on_canvas(obj_canvas,obj_camera_array):
    canvas_img = obj_canvas.canvas.copy()
    count = 0
    for elem in obj_camera_array.trackers_dict:
        for obj in obj_camera_array.trackers_dict[elem].objects_pool:
            if obj_camera_array.trackers_dict[elem].objects_pool[obj].update_status is True:
                obj_box = obj_camera_array.trackers_dict[elem].objects_pool[obj].last_box
                center_x,center_y = get_box_center(obj_box,mode='bottom')
                pt_center_x,pt_center_y = obj_camera_array.trackers_dict[elem].obj_STP.perspective_transformer.get_pred_transform(np.array([[center_x,center_y]],np.float))[0][0]
                disp_center_x,disp_center_y = int(pt_center_x*obj_canvas.scale_dict[elem][0]),int(pt_center_y*obj_canvas.scale_dict[elem][1])

                obj_vehicle = Vehicle(id=obj_camera_array.get_global_id(elem,obj_camera_array.trackers_dict[elem].objects_pool[obj].id),color=obj_camera_array.trackers_dict[elem].objects_pool[obj].color,scale=(obj_canvas.scale_dict[elem][0],obj_canvas.scale_dict[elem][1]))
                canvas_img = obj_vehicle.draw(canvas_img,count*obj_canvas.cam_region_width+disp_center_y,disp_center_x)
                if not obj_canvas.trajectory_dict.__contains__(obj):
                    new_obj_trajectory = Trajectory(id = obj, color = obj_camera_array.trackers_dict[elem].objects_pool[obj].color)
                    obj_canvas.trajectory_dict[obj] = new_obj_trajectory
                obj_canvas.trajectory_dict[obj].update((count*obj_canvas.cam_region_width+disp_center_y,disp_center_x),-1,0)
        count+=1
    count = 0
    for elem in obj_camera_array.multi_tracker_dict:
        for obj in obj_camera_array.multi_tracker_dict[elem].objects_pool:
            # print("update_status:",obj_camera_array.multi_tracker_dict[elem].objects_pool[obj].update_status)
            if obj_camera_array.multi_tracker_dict[elem].objects_pool[obj].update_status is True:
                last_frame = obj_camera_array.multi_tracker_dict[elem].objects_pool[obj].last_frame
                last_box = obj_camera_array.multi_tracker_dict[elem].objects_pool[obj].last_box
                center_x,center_y = get_box_center(last_box,mode='bottom')
                pt_center_x,pt_center_y = obj_camera_array.trackers_dict[elem].obj_STP.perspective_transformer.get_pred_transform(np.array([[center_x,center_y]],np.float))[0][0]
                delta_frame = obj_camera_array.time_stamp - last_frame
                disp_center_x,disp_center_y = int((pt_center_x+delta_frame*obj_camera_array.trackers_dict[elem].obj_STP.motion_params_4_all['mean_x'])*obj_canvas.scale_dict[elem][0]),int((pt_center_y+delta_frame*obj_camera_array.trackers_dict[elem].obj_STP.motion_params_4_all['mean_y'])*obj_canvas.scale_dict[elem][1])
                # print("disp_center_x,disp_center_y:",disp_center_x,disp_center_y)
                # obj_vehicle = Vehicle(id=obj_camera_array.get_global_id(elem,obj_camera_array.multi_tracker_dict[elem].objects_pool[obj].id),color=obj_camera_array.multi_tracker_dict[elem].objects_pool[obj].color,scale=(obj_canvas.scale_dict[elem][0],obj_canvas.scale_dict[elem][1]))
                # canvas_img = obj_vehicle.draw(canvas_img,count*obj_canvas.cam_region_width+disp_center_y,disp_center_x)
                cv2.circle(canvas_img,(count*obj_canvas.cam_region_width+disp_center_y,disp_center_x),2,obj_camera_array.multi_tracker_dict[elem].objects_pool[obj].color,2)
                canvas_img = cv2.putText(canvas_img,"{}".format(obj_camera_array.get_global_id(elem,obj_camera_array.multi_tracker_dict[elem].objects_pool[obj].id)),(count*obj_canvas.cam_region_width+disp_center_y+3,disp_center_x),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                
                if not obj_canvas.trajectory_dict.__contains__(obj):
                    print("====obj:",obj)
                    new_obj_trajectory = Trajectory(id = obj, color = obj_camera_array.trackers_dict[elem].objects_pool[obj].color)
                    print("================")
                    obj_canvas.trajectory_dict[obj] = new_obj_trajectory
                obj_canvas.trajectory_dict[obj].update((count*obj_canvas.cam_region_width+disp_center_y,disp_center_x),-1,1)
        count+=1
    
    # delete object info from last device
    delete_list = []
    for obj in obj_canvas.trajectory_dict:
        # if obj_canvas.trajectory_dict[obj].last_point[0]>=canvas_img.shape[1]:
        if obj_canvas.trajectory_dict[obj].last_point[0]>=(obj_canvas.device_count-1)*obj_canvas.cam_region_width+obj_canvas.region_dict[obj_camera_array.last_tracker_id]['monitor_region'].shape[1]-20:
            delete_list.append(obj)
    for obj in delete_list:
        obj_canvas.trajectory_dict.pop(obj)
        
    for elem in obj_canvas.trajectory_dict:
        obj = obj_canvas.trajectory_dict[elem]
        chosen_list = obj.get_disp_points()
        for i in range(len(chosen_list)-1):
            canvas_img = cv2.line(canvas_img,chosen_list[i][0],chosen_list[i+1][0],obj.color,2)
    # x,y = 23,4
    # cv2.circle(canvas_img,(int(x*obj_canvas.scale_dict[0][1]),int(y*obj_canvas.scale_dict[0][0])),2,(255,255,255),2)
    return canvas_img
    
# ===== CLASS =====
class Trajectory(object):
    def __init__(self,id=-1,color=(255,255,255)):
        self.id = id
        self.color = color
        self.list = []
        self.first_point = None
        self.first_frame = None
        self.last_point = None
        self.last_frame = None
        self.last_type = None
        self.update_status = True
        self.last_observed_iter = None
        
    def update(self,point,frame=-1,type=0):
        '''type=0,1
        0:monitored
        1:unmonitored
        '''
        self.list.append([point,frame,type])
        self.last_point,self.last_frame,self.last_type = point,frame,type
        if type==0:
            self.last_observed_iter = len(self.list)
    def get_disp_points(self):
        total = len(self.list)
        prev_points = [elem for elem in self.list[:self.last_observed_iter] if elem[2] == 0]
        prev_points.extend(self.list[self.last_observed_iter:])
        return prev_points

class Canvas(object):
    def __init__(self,
                 obj_camera_array=None,
                 img_height=200,
                 img_width=1000):
        self.obj_camera_array = obj_camera_array
        self.img_height = img_height
        self.img_width = img_width
        
        self.device_count = 0
        self.cam_region_width = 0
        
        self.starting_points_dict = {}
        self.scale_dict = {}
        self.region_dict = {}
        self.vehicle_dict = {}
        
        self.trajectory_dict = {}
        
        self.lane_markers_display = True
        
        self.get_scale()
        self.get_regions()
        self.get_canvas()
        
        
    def draw(self):
        img = self.canvas.copy()
        # for elem in self.vehicle_dict:
            # x,y = self.vehicle_dict[elem]['position'][0],self.vehicle_dict[elem]['position'][1]
            # img = self.vehicle_dict[elem]['object'].draw(img=img,x,y)
        return img

    def get_scale(self):
        self.device_count = len(self.obj_camera_array.trackers_dict)     # devices number in camera array
        self.cam_region_width = int(self.img_width/self.device_count)    # each image
        
        for elem in self.obj_camera_array.multi_tracker_dict:
            if self.obj_camera_array.associate_dict.__contains__(elem):
                # print("elem:",elem)
                associte_id = self.obj_camera_array.associate_dict[elem]
                self.starting_points_dict[elem] = self.obj_camera_array.multi_tracker_dict[elem].obj_multi_cameras_STP.get_start_point_transform(start_x_in_cam_2=0,start_y_in_cam_2=0)
            else:
                self.starting_points_dict[elem] = self.starting_points_dict[self.obj_camera_array.get_reverse_associate_dict()[elem]]
        # print("starting_points_dict:",self.starting_points_dict)
        for elem in self.obj_camera_array.trackers_dict:
            if self.obj_camera_array.multi_tracker_dict.__contains__(elem):
                # print("self.starting_points_dict[elem][1]:",self.starting_points_dict[elem])
                scale_y = self.cam_region_width/self.starting_points_dict[elem][1]
            else:
                scale_y = np.mean([self.scale_dict[e][1] for e in self.scale_dict if self.scale_dict[e] is not None])
            scale_x = self.img_height/self.obj_camera_array.trackers_dict[elem].obj_STP.perspective_transformer.transformed_width_for_pred
            self.scale_dict[elem] = (scale_x,scale_y)
        # print("self.scale_dict:",self.scale_dict)
        return self.scale_dict   
  
    def get_canvas(self):
        self.canvas = np.zeros((self.img_height,self.img_width,3),np.uint8)
        count = 0
        for elem in self.region_dict:
            self.canvas = self.region_dict[elem]["monitor_region"].draw(img=self.canvas,x=count*self.cam_region_width)
            self.canvas = self.region_dict[elem]["unmonitor_region"].draw(img=self.canvas,x=count*self.cam_region_width+self.region_dict[elem]["monitor_region"].shape[1])
            count+=1
        if self.lane_markers_display:
            self.canvas = cv2.line(self.canvas,(0,int(self.img_height/2)),(self.img_width-1,int(self.img_height/2)),(255,255,255),2)
        
        return self.canvas
    # inter camera
    
    def get_regions(self):
        self.get_scale()
        for elem in self.obj_camera_array.trackers_dict:
            print(self.scale_dict)
            monitor_region_width = int(self.obj_camera_array.trackers_dict[elem].obj_STP.perspective_transformer.transformed_height_for_pred*self.scale_dict[elem][1])
            
            int(self.obj_camera_array.trackers_dict[elem].obj_STP.perspective_transformer.transformed_height_for_pred*self.scale_dict[elem][1])
            monitor_region_height = self.img_height
            unmonitor_region_width = self.cam_region_width - monitor_region_width
            unmonitor_region_height = self.img_height
            
            obj_monitor_region = Region(id=elem,shape=(monitor_region_height,monitor_region_width),disp_info=device_info[elem])
            obj_unmonitor_region = Region(type="unmonitored",id=elem,shape=(unmonitor_region_height,unmonitor_region_width))
            
            self.region_dict[elem] = {
                    "monitor_region":obj_monitor_region,
                    "unmonitor_region":obj_unmonitor_region,
                    }    
                    
class Vehicle(object):
    def __init__(self,
                type=None,
                color=None,
                id=None,
                scale=(10,10)):
        self.type = type
        self.color = color
        self.id = id
        self.scale = scale
        
        if self.type is not None:
            self.shape = (int(vehicle_shape[self.type][0]*self.scale[0]),
                          int(vehicle_shape[self.type][1]*self.scale[1]))
        else:
            self.shape = (int(vehicle_shape['car'][0]*self.scale[0]),
                          int(vehicle_shape['car'][1]*self.scale[1]))
        
    def draw(self,img,x,y,mode='visible',icon=None):
        if icon is None:
            top_left = (x,y-int(self.shape[0]/2))
            top_right = (x+self.shape[1],y-int(self.shape[0]/2))
            bottom_left = (x,y+int(self.shape[0]/2))
            bottom_right = (x+self.shape[1],y+int(self.shape[0]/2))
            img = cv2.circle(img,(x,y),2,self.color,2)
            img = cv2.rectangle(img,top_left,bottom_right,self.color,2)
            img = cv2.line(img,top_left,bottom_right,self.color,2)
            img = cv2.line(img,top_right,bottom_left,self.color,2)
        else:
            resize_icon = cv2.resize(icon,(self.shape[1],self.shape[0]))
            x_top,x_bottom = x-int(self.shape[0]/2),x+int(self.shape[0]/2)
            y_top,y_bottom = y,y+self.shape[1]
            img[x_top:x_bottom,y_top:y_bottom] = resize_icon
            
        if self.id is not None:
            img = cv2.putText(img,str(self.id),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        return img
        
class Region(object):
    def __init__(self,
                type=None,
                color=(0,0,100),
                id=None,
                shape=None,
                disp_info=None):
        self.type = type
        self.color = color
        self.id = id
        self.shape = shape
        self.disp_info = disp_info
    def draw(self,img,x):
        # Draw color
        if self.type is not None:
            region_img = np.ones((self.shape[0],self.shape[1],3),np.uint8)*self.color
        else:
            region_img = np.ones((self.shape[0],self.shape[1],3),np.uint8)
        # Draw text
        if self.disp_info is not None:
            region_img = cv2.putText(region_img,self.disp_info,(0,self.shape[0]-3),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
        img[0:self.shape[0],x:x+self.shape[1]] = region_img
        return img
        
# ===== TEST FUNCITONS =====
def Canvas_test():
    obj = Canvas()
    img = obj.get_canvas()
    cv2.imshow('img',img)
    cv2.waitKey()

def Vehicle_test():
    img = np.zeros((600,800,3),np.uint8)
    obj = Vehicle(type='car',color=colors[0],id=0)
    img = obj.draw(img=img,x=100,y=100)
    # icon = np.ones((20,100,3),np.uint8)*255
    # img = obj.draw(img=img,x=100,y=100,icon=icon)
    cv2.imshow('img',img)
    cv2.waitKey()
    
def Region_test():
    img = np.zeros((100,800,3),np.uint8)
    obj = Region(type='unmonitored',shape=(100,200))
    img = obj.draw(img=img,x=100)
    cv2.imshow('img',img)
    cv2.waitKey()
    
if __name__=="__main__":
    pass
    # Vehicle_test()
    # Region_test()
    # Canvas_test()
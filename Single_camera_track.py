'''
SCT: Single camera tracking.
Multi-objects tracking with single camera.
written by sunzhu, 2019-03-19, version 1.0
'''

import os,sys
import pandas as pd
import cv2
import json

# color setting for displaying
colors = [[31  , 0   , 255] ,[0   , 159 , 255] ,[255 , 95  , 0] ,[255 , 19  , 0] ,
    [255 , 0   , 0]   ,[255 , 38  , 0]   ,[0   , 255 , 25]  ,[255 , 0   , 133] ,
    [255 , 172 , 0]   ,[108 , 0   , 255] ,[0   , 82  , 255] ,[0   , 255 , 6]   ,
    [255 , 0   , 152] ,[223 , 0   , 255] ,[12  , 0   , 255] ,[0   , 255 , 178] ]

# # ===== UTILS FUNCTIONS =====     
def isContactWithImage(img_height,img_width,box_x1,box_y1,box_x2,box_y2,thresh=5):
    '''check the box have contact with image boundaries or not'''
    if box_x1<=thresh or box_x2>=img_width-thresh or box_y1<=thresh or box_y2>=img_height-thresh:
        return True
    else:
        return False

def IOU(box1,box2):
    '''intersection over union: IOU'''
    x1,y1,x3,y3,width1,height1 = box1[0],box1[1],box1[2],box1[3],box1[2]-box1[0],box1[3]-box1[1]
    x2,y2,x4,y4,width2,height2 = box2[0],box2[1],box2[2],box2[3],box2[2]-box2[0],box2[3]-box2[1]
    # Intersection
    i_width = width1+width2-(max(x3,x4)-min(x1,x2))
    i_height = height1+height2-(max(y3,y4)-min(y1,y2))

    if i_width <=0 or i_height <= 0:
        IOU = 0
    else:
        i_area = i_width*i_height       # intersection area
        area1 = width1*height1
        area2 = width2*height2
        o_area = area1 + area2 - i_area # union area
        IOU = i_area*1./o_area        # intersection over union
    return IOU
    
def get_box_center(box,mode="both"):
    if mode == "both":
        return int((box[0]+box[2])/2),int((box[1]+box[3])/2)
    if mode == "bottom":
        return int((box[0]+box[2])/2),int(box[3])

# # ===== CLASS ======
class vehicle_object(object):
    def __init__(self,id):
        self.list = []
        self.id = id
        self.last_box = None
        self.last_frame = None
        self.update_status = True
        self.color = None
        
    def update(self,box,frame):
        self.list.append([box,frame])
        self.last_box = box
        self.last_frame = frame
        
    def set_color(self,color):
        self.color = color
     
        
class IOU_tracker(object):
    def __init__(self,  
                    frame_space_dist=5,
                    region_top=450,
                    region_bottom=50,
                    region_left=50,
                    region_right=50):
        # objects pool, all tracked objects are saved in this dict
        self.objects_pool = {}
        
        # threshvalue for tracking
        self.frame_space_dist = frame_space_dist    # ignore the object with long time interval
        self.region_top = region_top                # ignore the object not in region
        self.region_bottom = region_bottom
        self.region_left = region_left
        self.region_right = region_right
        self.thresh_iou = 0.3                       # ignore the object with low iou
        
        # image info
        self.img_height = 1080
        self.img_width = 1920
        
        # display setting
        self.display_monitor_region = False
        
    def get_available_id(self):
        exists_id = [id for id in self.objects_pool]
        i = 0
        while i < 100000:
            if i not in exists_id:
                return i
            else:
                i+=1
        return None     # The max id is 99999
        
    def get_available_color(self,id):
        i = id%len(colors)
        return colors[i]
        
    def isTrackFinish(self,frame):
        for elem in self.objects_pool:
            # print(elem,frame,self.objects_pool[elem].last_frame)
            if (frame - self.objects_pool[elem].last_frame) > self.frame_space_dist:
                self.objects_pool[elem].update_status = False
        
    def update(self,box):
        box_info = [box[0],box[1],box[2],box[3]]
        if self.isBoxInRegion(box_info):
            frame_info = box[4]
            matched_obj = self.match(box_info,frame_info)
            if matched_obj:
                self.objects_pool[matched_obj.id].update(box_info,frame_info)
            else:
                obj_id = self.get_available_id()
                obj = vehicle_object(obj_id)                # create a new vehicle object
                obj.set_color(self.get_available_color(obj_id))  # set color for displaying
                obj.update(box_info,frame_info)
                self.objects_pool[obj_id] = obj
        self.isTrackFinish(box[4])
        
    def match(self,box,frame):
        possible_obj_list = []
        for k,v in self.objects_pool.items():
            cmp_id = k
            cmp_locat = v.last_box
            cmp_frame = v.last_frame
            cmp_iou = IOU(box,cmp_locat)
            cmp_frame_dist = frame-cmp_frame
            if cmp_iou>=self.thresh_iou and cmp_frame_dist<=self.frame_space_dist:
                possible_obj_list.append([v,cmp_iou,cmp_frame_dist*1./self.frame_space_dist])
        matched_obj = self.findMostMatchedObject(possible_obj_list,0.6,0.4)
        return matched_obj
        
    def findMostMatchedObject(self,objs_list,weight_spatial=0.5,weight_temporal=0.5):
        '''find the nearest object in spatial and temporal space, the default of two space is 0.5 and 0.5'''
        if objs_list == []:
            return None
        else:
            def takeSecond(elem):
                return elem[1]
            dist_list = []
            for elem in objs_list:
                dist = elem[1]*weight_spatial+elem[2]*weight_temporal
                dist_list.append([objs_list[0],dist])
            
            dist_list.sort(key=takeSecond)
            return dist_list[-1][0][0]
        
    def isBoxInRegion(self,box):
        '''Check a object in the setting region or not
            Note: In vertical direction, we take the box bottom as a reference to check the present of object.
        '''
        if box[0]>self.region_left and box[2]<self.img_width-self.region_right and box[3]>self.region_top and box[3]<self.img_height-self.region_bottom:
            return True
        else:
            return False
      
    def draw_trajectory(self,img):
        if self.display_monitor_region:
            cv2.line(img,(0,self.region_top),(img.shape[1]-1,self.region_top),(255,255,255),5)
            cv2.line(img,(0,img.shape[0]-self.region_bottom),(img.shape[1]-1,img.shape[0]-self.region_bottom),(255,255,255),5)
        for k,v in self.objects_pool.items():
            if v.update_status:
                if len(v.list)>1:
                    for i in range(len(v.list)-1):
                        center_1 = get_box_center(v.list[i][0],mode="bottom")
                        center_2 = get_box_center(v.list[i+1][0],mode="bottom")
                        cv2.line(img,center_1,center_2,v.color,5)
                    cv2.putText(img,"ID:{}".format(v.id),(v.last_box[2],v.last_box[3])
                    ,cv2.FONT_HERSHEY_COMPLEX,2,v.color,5)
            cv2.namedWindow("img",cv2.WINDOW_NORMAL)
            cv2.imshow("img",img)
            cv2.waitKey(1)
            
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
            json.dump(saved_info,doc)
        

class STP_tracker(object):
    def __init__(self,  
                    frame_space_dist=5,
                    region_top=450,
                    region_bottom=50,
                    region_left=50,
                    region_right=50):
        # objects pool, all tracked objects are saved in this dict
        self.objects_pool = {}
        
        # threshvalue for tracking
        self.frame_space_dist = frame_space_dist    # ignore the object with long time interval
        self.region_top = region_top                # ignore the object not in region
        self.region_bottom = region_bottom
        self.region_left = region_left
        self.region_right = region_right
        self.thresh_iou = 0.3                       # ignore the object with low iou
        
        # image info
        self.img_height = 1080
        self.img_width = 1920
        
        # display setting
        self.display_monitor_region = False
        
    def get_available_id(self):
        exists_id = [id for id in self.objects_pool]
        i = 0
        while i < 100000:
            if i not in exists_id:
                return i
            else:
                i+=1
        return None     # The max id is 99999
        
    def get_available_color(self,id):
        i = id%len(colors)
        return colors[i]
        
    def isTrackFinish(self,frame):
        for elem in self.objects_pool:
            # print(elem,frame,self.objects_pool[elem].last_frame)
            if (frame - self.objects_pool[elem].last_frame) > self.frame_space_dist:
                self.objects_pool[elem].update_status = False
        
    def update(self,box):
        box_info = [box[0],box[1],box[2],box[3]]
        if self.isBoxInRegion(box_info):
            frame_info = box[4]
            matched_obj = self.match(box_info,frame_info)
            if matched_obj:
                self.objects_pool[matched_obj.id].update(box_info,frame_info)
            else:
                obj_id = self.get_available_id()
                obj = vehicle_object(obj_id)                # create a new vehicle object
                obj.set_color(self.get_available_color(obj_id))  # set color for displaying
                obj.update(box_info,frame_info)
                self.objects_pool[obj_id] = obj
        self.isTrackFinish(box[4])
        
    def match(self,box,frame):
        possible_obj_list = []
        for k,v in self.objects_pool.items():
            cmp_id = k
            cmp_locat = v.last_box
            cmp_frame = v.last_frame
            cmp_iou = IOU(box,cmp_locat)
            cmp_frame_dist = frame-cmp_frame
            if cmp_iou>=self.thresh_iou and cmp_frame_dist<=self.frame_space_dist:
                possible_obj_list.append([v,cmp_iou,cmp_frame_dist*1./self.frame_space_dist])
        matched_obj = self.findMostMatchedObject(possible_obj_list,0.6,0.4)
        return matched_obj
        
    def findMostMatchedObject(self,objs_list,weight_spatial=0.5,weight_temporal=0.5):
        '''find the nearest object in spatial and temporal space, the default of two space is 0.5 and 0.5'''
        if objs_list == []:
            return None
        else:
            def takeSecond(elem):
                return elem[1]
            dist_list = []
            for elem in objs_list:
                dist = elem[1]*weight_spatial+elem[2]*weight_temporal
                dist_list.append([objs_list[0],dist])
            
            dist_list.sort(key=takeSecond)
            return dist_list[-1][0][0]
        
    def isBoxInRegion(self,box):
        '''Check a object in the setting region or not
            Note: In vertical direction, we take the box bottom as a reference to check the present of object.
        '''
        if box[0]>self.region_left and box[2]<self.img_width-self.region_right and box[3]>self.region_top and box[3]<self.img_height-self.region_bottom:
            return True
        else:
            return False
      
    def draw_trajectory(self,img):
        if self.display_monitor_region:
            cv2.line(img,(0,self.region_top),(img.shape[1]-1,self.region_top),(255,255,255),5)
            cv2.line(img,(0,img.shape[0]-self.region_bottom),(img.shape[1]-1,img.shape[0]-self.region_bottom),(255,255,255),5)
        for k,v in self.objects_pool.items():
            if v.update_status:
                if len(v.list)>1:
                    for i in range(len(v.list)-1):
                        center_1 = get_box_center(v.list[i][0],mode="bottom")
                        center_2 = get_box_center(v.list[i+1][0],mode="bottom")
                        cv2.line(img,center_1,center_2,v.color,5)
                    cv2.putText(img,"ID:{}".format(v.id),(v.last_box[2],v.last_box[3])
                    ,cv2.FONT_HERSHEY_COMPLEX,2,v.color,5)
            cv2.namedWindow("img",cv2.WINDOW_NORMAL)
            cv2.imshow("img",img)
            cv2.waitKey(1)
            
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
            json.dump(saved_info,doc)
# # ====== TEST FUNCTIONS =====
def IOU_tracker_test():
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
    
    tracker = IOU_tracker()  
    tracker.display_monitor_region = True
    
    img_filepath_2 = os.path.join(dataset_root,cam_2) 
    img_savepath_2 = os.path.join(save_root,cam_2)
    if not os.path.exists(img_savepath_2):
        os.mkdir(img_savepath_2)

    # data = pd.read_csv(os.path.join(dataset_root,box_info_1))
    data = pd.read_csv(os.path.join(dataset_root,box_info_2))
    prev_fps = -1
    id_in_frame = 0
    for index, row in data[["x1", "y1", "x2", "y2","fps"]].iterrows():
        # print(row['x1'],row['y1'],row['x2'],row['y2'],row['fps'])
        tracker.update([row['x1'],row['y1'],row['x2'],row['y2'],row['fps']])
        # filename
        filename = str(row['fps'])+'.jpg'
        # img_filename = os.path.join(img_filepath_1,filename)
        img_filename = os.path.join(img_filepath_2,filename)
        img = cv2.imread(img_filename)
        tracker.draw_trajectory(img)
    tracker.save_data(os.path.join(save_root,cam_2))
    
if __name__=="__main__":

    IOU_tracker_test()
    # # root path
    # dataset_root = r"E:\DataSet\trajectory\concatVD"

    # # save root
    # save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data'

    # # images
    # cam_1 = 'wuqi1B'
    # cam_2 = 'wuqi2B'
    # cam_3 = 'wuqi3B'
    # cam_4 = 'wuqi4B'

    # # bbox information
    # box_info_1 = 'wuqiyuce1.csv'
    # box_info_2 = 'wuqiyuce2.csv'
    # box_info_3 = 'wuqiyuce3.csv'
    # box_info_4 = 'wuqiyuce4.csv'
    
    # tracker = IOU_tracker()  
    # tracker.display_monitor_region = True
    
    # img_filepath_2 = os.path.join(dataset_root,cam_2) 
    # img_savepath_2 = os.path.join(save_root,cam_2)
    # if not os.path.exists(img_savepath_2):
        # os.mkdir(img_savepath_2)

    # # data = pd.read_csv(os.path.join(dataset_root,box_info_1))
    # data = pd.read_csv(os.path.join(dataset_root,box_info_2))
    # prev_fps = -1
    # id_in_frame = 0
    # for index, row in data[["x1", "y1", "x2", "y2","fps"]].iterrows():
        # # print(row['x1'],row['y1'],row['x2'],row['y2'],row['fps'])
        # tracker.update([row['x1'],row['y1'],row['x2'],row['y2'],row['fps']])
        # # filename
        # filename = str(row['fps'])+'.jpg'
        # # img_filename = os.path.join(img_filepath_1,filename)
        # img_filename = os.path.join(img_filepath_2,filename)
        # img = cv2.imread(img_filename)
        # tracker.draw_trajectory(img)
    # tracker.save_data(os.path.join(save_root,cam_2))

'''
Camera associate
written by sunzhu, 2019-03-21, version 1.0
'''

import os,sys
import math
import pandas as pd
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt

from Single_camera_track import IOU_tracker
from data_generator import load_tracking_info
from Single_camera_track import get_box_center
import Perspective_transform as pt

from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# style set for figure
sns.set_palette('deep', desat=.6)
sns.set_context(rc={'figure.figsize': (8, 5) } )


# root path
dataset_root = r"E:\DataSet\trajectory\concatVD"

# save root
save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data'

# perspective transform data
pt_trans_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\ROItools\data'

# device_info
device_info = [
    ['wuqi1B','wuqiyuce1.csv','ROI_cam_1_transformer.json'],
    ['wuqi2B','wuqiyuce2.csv','ROI_cam_2_transformer.json'],
    ['wuqi3B','wuqiyuce3.csv','ROI_cam_3_transformer.json'],
    ['wuqi4B','wuqiyuce4.csv','ROI_cam_4_transformer.json'],
]

# camera associate
associate_cam = {
    0:1,
    1:0,
    2:3,
    3:2
}


associate_dict_c1_c2 = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:4
}
# camera associate
associate_dict_c3_c4 = {
    0:0,
    1:1,
    2:2,
    3:3,
    4:4
}


class Gauss_distribution(object):
    def __init__(self,mu,sigma):
        self.mu = mu
        self.sigma = sigma
        
    def get_probability(self,x):
        
        if type(self.mu) is np.float64:
            return self._1D_gaussian(x)
        if type(self.mu) is np.ndarray:
            return self._2D_gaussian(x)
        return None
        
    def _2D_gaussian(self,x):
        '''Get the probability of x with 2D gaussian'''
        n = self.mu.shape[0]
        
        sigma_det = np.linalg.det(self.sigma)
        sigma_inv = np.linalg.inv(self.sigma)
        
        print("sigma_det,sigma_inv:",sigma_det,sigma_inv)
        
        den = np.sqrt((2*np.pi)**n*sigma_det)
        num = np.einsum('...k,kl,...l->...',x-self.mu,sigma_inv,x-self.mu)

        return np.exp(-num/2)/den
        
    def _1D_gaussian(self,x):
        '''Get the probability of x with 2D gaussian'''
        den = np.sqrt(2*np.pi*np.power(self.sigma,2))
        num = np.power(x-self.mu,2)/np.power(self.sigma,2)
        print("den,num:",den,num)
        return np.exp(-num/2)/den

    def get_normal_1D_gaussian(self,x):
        '''Get the probability of x with 2D gaussian'''
        x_norm = (x-self.mu)/self.sigma
        den = np.sqrt(2*np.pi*np.power(1,2))
        num = np.power(x_norm,2)/np.power(1,2)
        print("den,num:",den,num)
        return np.exp(-num/2)/den
        
def gaussian_test():
    '''No use for this task, just for testing the effect of gaussian distribution'''
    # 1D gauss testing
    obj_1D = Gauss_distribution(0,2)

    x = np.linspace(-10,10,100)
    y = obj_1D.get_probability(x)
    
    plt.plot(x,y,'b-',linewidth=3)
    
    # 2D gauss testing
    X = np.linspace(-3,3,60)
    Y = np.linspace(-4,4,60)
    X,Y = np.meshgrid(X,Y)
    mu = np.array([0.,0.])
    Sigma = np.array([[1.,-0.5],[-0.5,1.5]])
    pos = np.empty(X.shape+(2,))
    pos[:,:,0]= X
    pos[:,:,1] = Y

    obj_2D = Gauss_distribution(mu,Sigma)
    Z = obj_2D.get_probability(pos)

    fig =plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,Z,rstride=3,cstride=3,linewidth=1,antialiased =True)
    cset = ax.contour(X,Y,Z,zdir='z',offset=-0.15)

    ax.set_zlim(-0.15,0.2)
    ax.set_zticks(np.linspace(0,0.2,5))
    ax.view_init(27,-21)
    plt.show()

def get_dist_in_deltaT(pt_trace):
    '''Get moving distance of object within delta t'''
    delta_loc = {}
    for k in pt_trace:
        obj_trace = pt_trace[k]
        obj_delta_loc = []
        for i in range(len(obj_trace)-1):
            x1,y1,t1 = obj_trace[i][0][0],obj_trace[i][0][1],obj_trace[i][1]
            x2,y2,t2 = obj_trace[i+1][0][0],obj_trace[i+1][0][1],obj_trace[i+1][1]
            delta_x = x2 - x1
            delta_y = y2 - y1
            delta_t = t2 - t1
            delta_x = delta_x/delta_t
            delta_y = delta_y/delta_t
            obj_delta_loc.append([delta_x,delta_y])
        delta_loc[k] = obj_delta_loc
    return delta_loc
    
def get_statistical_paras_of_dist_in_deltaT(delta_loc):
    output_dict = {}
    for k in delta_loc:
        tmp_dict = {}
        loc_x,loc_y = zip(*delta_loc[k])
        tmp_dict['mean_x'] = np.mean(loc_x)
        tmp_dict['var_x'] = np.var(loc_x)
        tmp_dict['mean_y'] = np.mean(loc_y)
        tmp_dict['var_y'] = np.var(loc_y)
        tmp_dict['mean_xy'] = np.array([np.mean(loc_x),np.mean(loc_y)])
        tmp_dict['cov_xy'] = np.cov(loc_x,loc_y)
        output_dict[k] = tmp_dict
    return output_dict
    
def learn_spatial_temporal_parameters():
    return

def get_STP_in_single_camera(pt_trace):
    '''
    Parameters:
        pt_trace:trace after perspective transformation in a single camera
    '''
    dist_dict_1 = get_dist_in_deltaT(pt_trace)
    result = get_statistical_paras_of_dist_in_deltaT(dist_dict_1)
    return result
    
def get_STP_in_multi_cameras(pt_trace_1,pt_trace_2,associate_dict):
    '''
    Parameters:
        pt_trace_1:trace after perspective transformation in cam_1
        pt_trace_2:trace after perspective transformation in cam_2
        associate_dict:mapping relation between objects in cam_1 and cam_2
    '''
    # chosen_id_cam_1 = 2
    # chosen_id_cam_2 = associate_dict[chosen_id_cam_1]
    # pt_box_1,frame_cam_1 = zip(*pt_trace_1[str(chosen_id_cam_1)])
    # pt_box_2,frame_cam_2 = zip(*pt_trace_2[str(chosen_id_cam_2)])
    
    STP_S_1 = get_STP_in_single_camera(pt_trace_1)
    STP_S_2 = get_STP_in_single_camera(pt_trace_2)

    objs_associate_info = []
    for k in associate_dict:
        objs_associate_info.append([pt_trace_1[str(k)][0],pt_trace_2[str(k)][0]])
    for i,v in enumerate(objs_associate_info):
        x,y = v[0][0][0],v[0][0][1]
        delta_t = v[1][1]-v[0][1]
        pred_x = x + delta_t*STP_S_1[str(i)]['mean_x']
        pred_y = y + delta_t*STP_S_1[str(i)]['mean_y']
        v.append([pred_x,pred_y])
    src_pts = np.array([elem[2] for elem in objs_associate_info])
    dst_pts = np.array([elem[1][0] for elem in objs_associate_info])
    
    print(src_pts)
    print(dst_pts)
    # Get affine transform matrix using least square algorithm
    trans_matrix,_ = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,1.0)
    print(trans_matrix)
    
    input_vector = np.array(src_pts)
    pt = cv2.transform(input_vector[None,:,:],trans_matrix)
    M = np.mat(trans_matrix[:2])
    print(M)
    
    im = np.mat((src_pts[0][0],src_pts[0][1],1)).T
    print(im)
    rim = M*im
    print(rim)
    
    # TEST: affine transformation
    
    
        
        
        
    # delta_t_between_cams = frame_cam_2[0] - frame_cam_1[0]
    # print(delta_t_between_cams)
    
    # pred_x = pt_box_1[0][0]+delta_t_between_cams*STP_S_1[str(chosen_id_cam_1)]['mean_x']
    # pred_y = pt_box_1[0][1]+delta_t_between_cams*STP_S_1[str(chosen_id_cam_1)]['mean_y']
    # print(pred_x,pred_y)
    
    
    return
    
def get_box_center(box_list):
    center_box_list = []
    for elem in box_list:
        center_box_list.append([(elem[0]+elem[2])*1.0/2,elem[3]*1.0])
    return center_box_list
    
def get_pt_box_info(box_info,pt_obj):
    '''Get perspective transformation of box bottom center'''
    if type(box_info) is dict:
        pt_box_info = {}
        for elem in box_info:
            box_list,frame_list = zip(*box_info[elem]['list'])
            center_box_list = get_box_center(box_list)
            pt_box_list = pt_obj.get_pred_transform(center_box_list)
            center_box_l = pt_obj.get_inverse_pred_transform(pt_box_list[0])
            pt_box_info[elem]=list(zip(pt_box_list[0],frame_list))
        return pt_box_info
    if type(box_info) is list:
        pt_box_info = []
        box_list,frame_list = zip(*box_info)
        center_box_list = get_box_center(box_list)
        pt_box_list = pt_obj.get_pred_transform(center_box_list)
        return list(zip(pt_box_list[0],frame_list))
            
def match_based_on_spatial_temperal_prior_test_1(pt_box_info,pt_obj,t_interval=30):
    '''test location predicting of object,
    Predict object location based on the object location in current frame, movement characteristics and time interval setting
    Parameters:
        pt_box_info: Bounding box information after perspective transformation
        pt_obj: perspective transformer object
        t_interval: time interval(frames)
    '''
    # file path
    img_root = r'E:\DataSet\trajectory\concatVD\wuqi1B'
    save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data\resutls\loc_predicate\Frame30'
    test_id = '1'

    dist_dict_1 = get_dist_in_deltaT(pt_box_info)
    d = get_statistical_paras_of_dist_in_deltaT(dist_dict_1)
    obj_nx = Gauss_distribution(    t_interval*d[test_id]['mean_x'],
                                    t_interval**2*d[test_id]['var_x'])
    obj_ny = Gauss_distribution(    t_interval*d[test_id]['mean_y'],
                                    t_interval*d[test_id]['var_y'])
       
    # test on id 1
    for v,elem in enumerate(pt_box_info[test_id]):
        fname_1 = str(elem[1])+'.jpg'
        fname_2 = str(elem[1]+t_interval)+'.jpg'
        print(elem[0])
        x_elem = get_rational_value(elem[0][0]*10,0,80)
        y_elem = get_rational_value(elem[0][1]*10,0,300)

        print(pt_box_info['1'][v+t_interval][0])
        
        p_map = get_probability_map(obj_nx,obj_ny,elem[0][0],elem[0][1])
        color_p_map = np.zeros([p_map.shape[0],p_map.shape[1],3],dtype=np.uint8)
        color_p_map[:,:,1] = p_map
        color_p_map = cv2.resize(color_p_map,(int(pt_obj.transformed_width_for_disp),int(pt_obj.transformed_height_for_disp)))
        color_p_map = cv2.flip(color_p_map,0)   # 0:vertical flip
        pt_color_p_map = pt_obj.get_inverse_disp_transform(color_p_map)
        
        img_1 = cv2.imread(os.path.join(img_root,fname_1))  # current frame
        img_2 = cv2.imread(os.path.join(img_root,fname_2))  # frame after fixed time interval 
        diff_img = cv2.absdiff(img_1,img_2)                 # For displaying same object from different frames in one image.
        alpha = 0.5
        inverse_elem = pt_obj.get_inverse_pred_transform([elem[0]])
        img_3 = cv2.addWeighted(img_1, alpha, diff_img, 1-alpha, 0)
        alpha = 0.7
        img_3 = cv2.addWeighted(img_3, alpha, pt_color_p_map, 1-alpha, 0)

        cv2.namedWindow('img_3',cv2.WINDOW_NORMAL)
        cv2.imshow('img_3',img_3)
        cv2.imwrite(os.path.join(save_root,fname_1),img_3)
        
        cv2.waitKey()
    return

def get_probability_map(obj_nx,obj_ny,x,y,height=300,width=80):
    bg = np.zeros([height,width])
    base_x = np.linspace(0,8,80)
    base_y = np.linspace(0,30,300)

    delta_x = base_x - x
    delta_y = base_y - y

    p_x = obj_nx.get_probability(delta_x)
    p_y = obj_ny.get_probability(delta_y)
    
    # ----- 单变量分布显示测试 -----
    # plt.plot(base_x,p_x,'b-',linewidth=3)
    # plt.show()
    
    # 生成分布概率矩阵
    mat_p_x = np.mat(p_x)
    mat_p_y = np.mat(p_y).T
    mat_p = mat_p_y*mat_p_x
    
    # 返回分布概率显示图像
    return get_normalize_mat(mat_p)

def get_normalize_mat(data):
    '''Convert the probability map into color image'''
    mx = np.max(data)
    mn = np.min(data)
    mx_mat = np.ones_like(data)*mx
    mn_mat = np.ones_like(data)*mn
    n_data = np.array((data-mn_mat)*255/(mx-mn),dtype=np.uint8)
    return n_data
    
def get_rational_value(x,low,high):
    '''Limit input in range'''
    y = np.max([np.min([x,high]),0])
    return y

    
if __name__=="__main__":
    # cam A
    device_id = 0

    img_filepath_1 = os.path.join(dataset_root,device_info[device_id][0]) 
    img_savepath_1 = os.path.join(save_root,device_info[device_id][0])
    pt_savepath_1 = os.path.join(pt_trans_root,device_info[device_id][2])
    
    # cam B
    associate_device_id = associate_cam[device_id]
    
    img_filepath_2 = os.path.join(dataset_root,device_info[associate_device_id][0]) 
    img_savepath_2 = os.path.join(save_root,device_info[associate_device_id][0])
    pt_savepath_2 = os.path.join(pt_trans_root,device_info[associate_device_id][2])

    tracker_record_1 = load_tracking_info(img_savepath_1)
    tracker_record_2 = load_tracking_info(img_savepath_2)
    
    # # ----- 原始IOU_tracker 跟踪轨迹显示 test -----
    # # print(tracker_record_1['2']['list'])
    # # img = cv2.imread(r'E:\DataSet\trajectory\concatVD\wuqi1B\0.jpg')
    # # for elem in tracker_record_1['4']['list']:
        # # print(elem[0][2],elem[0][3])
        # # cv2.circle(img,(elem[0][2],elem[0][3]),10,(0,0,255),10)
    # # cv2.namedWindow('img_1',cv2.WINDOW_NORMAL)
    # # cv2.imshow('img_1',img)
    # # cv2.waitKey()
    
    # # ----- center 轨迹显示 test -----
    # chosen_id = '3'
    # BBox,BFrame = zip(*tracker_record_1[chosen_id]['list'])
    # Center_BBox = get_box_center(BBox)
    # print(Center_BBox)
    # img = cv2.imread(r'E:\DataSet\trajectory\concatVD\wuqi1B\0.jpg')
    # for elem in Center_BBox:
        # print(elem)
        # cv2.circle(img,(int(elem[0]),int(elem[1])),10,(0,0,255),10)
    # cv2.namedWindow('img_1',cv2.WINDOW_NORMAL)
    # cv2.imshow('img_1',img)
    # cv2.waitKey()
    
    
    pt_obj_1 = pt.Perspective_transformer(pt_savepath_1)
    pt_obj_2 = pt.Perspective_transformer(pt_savepath_2)
    
    pt_box_info_1 = get_pt_box_info(tracker_record_1,pt_obj_1)
    pt_box_info_2 = get_pt_box_info(tracker_record_2,pt_obj_2)
    
    d = get_STP_in_multi_cameras(pt_box_info_1,pt_box_info_2,associate_dict_c1_c2)
    
    # ----- TEST: location predicting in single camera -----
    # match_based_on_spatial_temperal_prior_test_1(pt_box_info_1,pt_obj_1)
    
    # ----- TEST: spatial temperal prior in single camera -----
    # d = get_STP_in_single_camera(pt_box_info_1)

    # ----- TEST: 概率计算 1D ----- 
    # n = 10
    # obj_nx = Gauss_distribution(n*d['1']['mean_x'],n**2*d['1']['var_x'])
    # obj_ny = Gauss_distribution(n*d['1']['mean_y'],n**2*d['1']['var_y'])
    
    # plt.show()
    
    # # obj = Gauss_distribution(d['1']['mean_y'],d['1']['var_y'])
    # obj = Gauss_distribution(np.float64(0),np.float64(0.2))
    # # 1D gauss testing
    # # print(type(d['1']['mean_x']),type(d['1']['var_x']))

    # x = np.linspace(-1,1,100)
    # y = obj.get_probability(x)

    # plt.plot(x,y,'b-',linewidth=3)
    # plt.show()
    
    
    # ----- TEST: 2D gaussian distribution -----
    # X = np.linspace(-1,1,60)
    # Y = np.linspace(-1,1,60)
    # X,Y = np.meshgrid(X,Y)

    # pos = np.empty(X.shape+(2,))
    # pos[:,:,0]= X
    # pos[:,:,1] = Y

    # Z = obj.get_probability(pos)
    # print(np.max(Z))

    # fig =plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X,Y,Z,rstride=3,cstride=3,linewidth=1,antialiased =True)
    # cset = ax.contour(X,Y,Z,zdir='z',offset=-0.15)

    # ax.set_zlim(-0.0001,57)
    # ax.set_zticks(np.linspace(0,0.2,5))
    # ax.view_init(27,-21)
    # plt.show()
    
    # ----- TEST: Display gaussian distribution in 3D -----
    # with sns.axes_style("dark"):
        # sns.jointplot(display_x, display_y, kind="kde",space=0)
    # plt.show()
    
    # pt_tracker_record_1 = {}
    # pt_tracker_record_2 = {}
    
    
    
    
        
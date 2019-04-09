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
from sklearn.linear_model import LinearRegression


# from Single_camera_track import get_box_center


from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# ========== CONFIGURATION ==========
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

# ========== BASIC CLASS ==========
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
        
        # print("sigma_det,sigma_inv:",sigma_det,sigma_inv)
        
        den = np.sqrt((2*np.pi)**n*sigma_det)
        num = np.einsum('...k,kl,...l->...',x-self.mu,sigma_inv,x-self.mu)

        return np.exp(-num/2)/den
        
    def _1D_gaussian(self,x):
        '''Get the probability of x with 2D gaussian'''
        den = np.sqrt(2*np.pi*np.power(self.sigma,2))
        num = np.power(x-self.mu,2)/np.power(self.sigma,2)
        # print("den,num:",den,num)
        return np.exp(-num/2)/den

    def get_normal_1D_gaussian(self,x):
        '''Get the probability of x with 2D gaussian'''
        x_norm = (x-self.mu)/self.sigma
        den = np.sqrt(2*np.pi*np.power(1,2))
        num = np.power(x_norm,2)/np.power(1,2)
        # print("den,num:",den,num)
        return np.exp(-num/2)/den

class Single_camera_STP(object):
    def __init__(   self,
                    tracking_record = None,
                    perspective_transformer = None,
                    time_interval = 25):
        self.perspective_transformer = perspective_transformer
        self.tracking_record = tracking_record
        self.time_interval = time_interval
        
        self.perspective_trace = None
        self.distance_dict = None
        self.motion_params_4_all = None
        self.motion_params_4_each = None
        
        self.prob_alpha_x = 1       # x amplitude multiplier
        self.prob_alpha_y = 1       # y amplitude multiplier
        
        self.var_beta_x = 25   # x range multiplier
        self.var_beta_y = 1         # y range multiplier
        
        self.get_motion_params()
        self.nx_predictor = Gauss_distribution(
                                    self.time_interval*self.motion_params_4_all['mean_x'],
                                    self.time_interval*self.var_beta_x*self.motion_params_4_all['var_x'])
        self.ny_predictor = Gauss_distribution(
                                    self.time_interval*self.motion_params_4_all['mean_y'],
                                    self.time_interval*self.var_beta_y*self.motion_params_4_all['var_y'])
    
    def version(self):
        return print("Written by sunzhu, 2019-03-21, version 1.0")
    
    def get_motion_params(self):
        self.perspective_trace = get_pt_box_info(self.tracking_record,self.perspective_transformer)
        self.distance_dict = get_dist_in_deltaT(self.perspective_trace)
        self.motion_params_4_each,self.motion_params_4_all = get_statistical_paras_of_dist_in_deltaT(self.distance_dict)
        return self.motion_params_4_each,self.motion_params_4_all
        
    def update_predictor(self,tracking_record=None,time_interval=None):
        if tracking_record is not None:
            self.tracking_record = tracking_record
            self.get_motion_params()
        if time_interval is not None:
            self.time_interval = time_interval
        self.nx_predictor = Gauss_distribution(
                                    self.time_interval*self.motion_params_4_all['mean_x'],
                                    self.time_interval*self.var_beta_x*self.motion_params_4_all['var_x'])
        self.ny_predictor = Gauss_distribution(
                                    self.time_interval*self.motion_params_4_all['mean_y'],
                                    self.time_interval*self.var_beta_y*self.motion_params_4_all['var_y'])
            
    def get_probability_map(self,base_x,base_y,start_x=0,start_y=0,height=300,width=80):
        return get_probability_map(self.nx_predictor,self.ny_predictor,base_x,base_y,start_x,start_y,height,width)
        
    def get_probability(self,x,y,base_x,base_y):
        delta_x = x - base_x 
        delta_y = y - base_y
        prob_x = self.nx_predictor.get_probability(delta_x)
        prob_y = self.ny_predictor.get_probability(delta_y)
        prob_xy = self.prob_alpha_x*prob_x*self.prob_alpha_y*prob_y
        return prob_x,prob_y,prob_xy
        
    def get_distance(self,x,y,base_x,base_y,alpha=0.5):
        pred_x = base_x+self.time_interval*self.motion_params_4_all['mean_x']
        pred_y = base_y+self.time_interval*self.motion_params_4_all['mean_y']
        comp_distance = abs(x-pred_x)*(1-alpha)+abs(y-pred_y)*alpha
        return comp_distance

class Multi_cameras_STP(object):
    def __init__(   self,
                    obj_single_camera_STP_cam_1 = None,
                    obj_single_camera_STP_cam_2 = None,
                    associate_dict = None):
        self.obj_single_camera_STP_cam_1 = obj_single_camera_STP_cam_1
        self.obj_single_camera_STP_cam_2 = obj_single_camera_STP_cam_2
        self.associate_dict = associate_dict
        
        self.coord_pairs = None
        self.coord_transformer = None
        self.starting_point = None
        
        self.prob_alpha_x = 1       # x multiplier for position probability calculating
        self.prob_alpha_y = 1       # y multiplier for position probability calculating
        
        self.var_beta_x = 5         # x range multiplier
        self.var_beta_y = 1         # y range multiplier
        
        self.get_coord_transformer()
        self.get_start_point_transform()
        
    def version(self):
        return print("Written by sunzhu, 2019-03-21, version 1.0")
    
    def get_coord_transformer(self):
        self.coord_pairs = get_STP_in_multi_cameras(
                                    self.obj_single_camera_STP_cam_1,
                                    self.obj_single_camera_STP_cam_2,
                                    self.associate_dict
                                    )
                                    
        cam_1_x,cam_2_x,cam_1_y,cam_2_y = self.coord_pairs
        
        self.coord_transformer = {}
        
        # Front to back transformer
        F2B_transformer = {}
        
        model_F2B_x = LinearRegression()
        model_F2B_x.fit(cam_1_x, cam_2_x)
        model_F2B_y = LinearRegression()
        model_F2B_y.fit(cam_1_y, cam_2_y)
        
        F2B_transformer['x'] = model_F2B_x
        F2B_transformer['y'] = model_F2B_y
        
        # Back to front transformer
        B2F_transformer = {}
        
        model_B2F_x = LinearRegression()
        model_B2F_x.fit(cam_2_x, cam_1_x)
        model_B2F_y = LinearRegression()
        model_B2F_y.fit(cam_2_y, cam_1_y)
        
        B2F_transformer['x'] = model_B2F_x
        B2F_transformer['y'] = model_B2F_y
        
        # Coordinate transformer
        self.coord_transformer['F2B'] = F2B_transformer
        self.coord_transformer['B2F'] = B2F_transformer
        
        return self.coord_transformer

    def get_start_point_transform(self,start_x_in_cam_2=0,start_y_in_cam_2=0):
        '''Transform the coordinate value of starting point in cam 2 into the coordinate value in cam 1
            The coordinate value refers the position of object after perspective transforming.
        '''
        corresponding_start_x_in_cam_1 = self.coord_transformer['B2F']['x'].predict([[start_x_in_cam_2]])[0][0]
        corresponding_start_y_in_cam_1 = self.coord_transformer['B2F']['y'].predict([[start_y_in_cam_2]])[0][0]
        # print(corresponding_start_x_in_cam_1,corresponding_start_y_in_cam_1)
        self.starting_point = [corresponding_start_x_in_cam_1,corresponding_start_y_in_cam_1]
        return self.starting_point
    
    def get_probability_map(self,base_x,base_y,t_interval=None,height=300,width=80):
        nx_predictor = Gauss_distribution(  t_interval*self.obj_single_camera_STP_cam_1.motion_params_4_all['mean_x'],
                                            t_interval*self.var_beta_x*self.obj_single_camera_STP_cam_1.motion_params_4_all['var_x'])
        ny_predictor = Gauss_distribution(  t_interval*self.obj_single_camera_STP_cam_1.motion_params_4_all['mean_y'],
                                            t_interval*self.var_beta_y*self.obj_single_camera_STP_cam_1.motion_params_4_all['var_y'])
        # print("starting_points:",self.starting_point[0],self.starting_point[1])
        
        return get_probability_map( nx_predictor,
                                    ny_predictor,
                                    base_x,
                                    base_y,
                                    start_x=self.starting_point[0],
                                    start_y=self.starting_point[1],
                                    )
                                    
    def get_probability(self,x,y,base_x,base_y,t_interval=None):
        trans_x = self.coord_transformer['B2F']['x'].predict([[x]])[0][0]
        trans_y = self.coord_transformer['B2F']['y'].predict([[y]])[0][0]
        delta_x = trans_x - base_x
        delta_y = trans_y - base_y

        # print("delta_x:",delta_x,"delta_y:",delta_y)
        nx_predictor = Gauss_distribution(  
                        t_interval*self.obj_single_camera_STP_cam_1.motion_params_4_all['mean_x'],
                        t_interval*self.var_beta_x*self.obj_single_camera_STP_cam_1.motion_params_4_all['var_x'])
        ny_predictor = Gauss_distribution(  
                        t_interval*self.obj_single_camera_STP_cam_1.motion_params_4_all['mean_y'],
                        t_interval*self.var_beta_y*self.obj_single_camera_STP_cam_1.motion_params_4_all['var_y'])
        # print('p_x',nx_predictor.get_probability(delta_x))      
        # print('p_y',ny_predictor.get_probability(delta_y))
        prob_x = nx_predictor.get_probability(delta_x)
        prob_y = ny_predictor.get_probability(delta_y)
        prob_xy = self.prob_alpha_x*prob_x*self.prob_alpha_y*prob_y
        return prob_x,prob_y,prob_xy
        
    def get_distance(self,x,y,base_x,base_y,t_interval=None,alpha=0.5):
        trans_x = self.coord_transformer['B2F']['x'].predict([[x]])[0][0]
        trans_y = self.coord_transformer['B2F']['y'].predict([[y]])[0][0]
        pred_x = base_x+t_interval*self.obj_single_camera_STP_cam_1.motion_params_4_all['mean_x']
        pred_y = base_y+t_interval*self.obj_single_camera_STP_cam_1.motion_params_4_all['mean_y']
        comp_distance = abs(x-pred_x)*(1-alpha)+abs(y-pred_y)*alpha
        return comp_distance
        
        
# =========== UTILS FUNCTIONS ============
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
    res_4_each = {}  # parameters of each object
    res_4_all = {}  # parameters of all objects
    x_list = []
    y_list = []
    # Calculate the mean,var and cov of x,y for each object
    for k in delta_loc:
        tmp_dict = {}
        loc_x,loc_y = zip(*delta_loc[k])
        tmp_dict['mean_x'] = np.mean(loc_x)
        tmp_dict['var_x'] = np.var(loc_x)
        tmp_dict['mean_y'] = np.mean(loc_y)
        tmp_dict['var_y'] = np.var(loc_y)
        tmp_dict['mean_xy'] = np.array([np.mean(loc_x),np.mean(loc_y)])
        tmp_dict['cov_xy'] = np.cov(loc_x,loc_y)
        res_4_each[k] = tmp_dict
        x_list.extend(loc_x)
        y_list.extend(loc_y)
        
    # Calculate the mean, var and cov of x,y for all objects
    res_4_all['mean_x'] = np.mean(x_list)
    res_4_all['var_x'] = np.var(x_list)
    res_4_all['mean_y'] = np.mean(y_list)
    res_4_all['var_y'] = np.var(y_list)
    res_4_all['mean_xy'] = np.array([np.mean(x_list),np.mean(y_list)])
    res_4_all['cov_xy'] = np.cov(x_list,y_list)
    
    return res_4_each,res_4_all
    
def get_STP_in_single_camera(pt_trace):
    '''
    Parameters:
        pt_trace:trace after perspective transformation in a single camera
    '''
    dist_dict_1 = get_dist_in_deltaT(pt_trace)
    res_4_each,res_4_all = get_statistical_paras_of_dist_in_deltaT(dist_dict_1)
    return res_4_each,res_4_all
    
def get_STP_in_multi_cameras(obj_single_camera_STP_cam_1,obj_single_camera_STP_cam_2,associate_dict):
    '''
    Parameters:
        obj_single_camera_STP_cam_1: STP in cam 1
        obj_single_camera_STP_cam_2: STP in cam 2
        associate_dict:mapping relation between objects in cam_1 and cam_2
    '''
    # chosen_id_cam_1 = 2
    # chosen_id_cam_2 = associate_dict[chosen_id_cam_1]
    # pt_box_1,frame_cam_1 = zip(*pt_trace_1[str(chosen_id_cam_1)])
    # pt_box_2,frame_cam_2 = zip(*pt_trace_2[str(chosen_id_cam_2)])
   
    STP_S_4_each_1 = obj_single_camera_STP_cam_1.motion_params_4_each
    STP_S_4_all_1 = obj_single_camera_STP_cam_1.motion_params_4_all
    STP_S_4_each_2 = obj_single_camera_STP_cam_2.motion_params_4_each
    STP_S_4_all_2 = obj_single_camera_STP_cam_2.motion_params_4_all
    
    pt_trace_1 = obj_single_camera_STP_cam_1.perspective_trace
    pt_trace_2 = obj_single_camera_STP_cam_2.perspective_trace
    
    # # ===== Method 1: Predict location using motion parameters for all objects =====
    # objs_associate_info = []
    # for k in associate_dict:
        # for i in range(min(len(pt_trace_1[str(k)]),len(pt_trace_2[str(k)]))):
            # objs_associate_info.append([pt_trace_1[str(k)][i],pt_trace_2[str(k)][i]])


    # for i,v in enumerate(objs_associate_info):
        # x,y = v[0][0][0],v[0][0][1]
        # delta_t = v[1][1]-v[0][1]
        # pred_x = x + delta_t*STP_S_4_all_1['mean_x']
        # pred_y = y + delta_t*STP_S_4_all_1['mean_y']
        # v.append([pred_x,pred_y])
    # src_x,src_y = np.array([elem[2][0] for elem in objs_associate_info]),np.array([elem[2][1] for elem in objs_associate_info])
    # dst_x,dst_y = np.array([elem[1][0][0] for elem in objs_associate_info]),np.array([elem[1][0][1] for elem in objs_associate_info])
    
    # ===== Method 2 : Predict location using each object's parameter
    objs_associate_info = []
    for k in associate_dict:
        for i in range(min(len(pt_trace_1[str(k)]),len(pt_trace_2[str(k)]))):
            objs_associate_info.append([k,pt_trace_1[str(k)][i],pt_trace_2[str(k)][i]])
            
    for i,v in enumerate(objs_associate_info):
        x,y = v[1][0][0],v[1][0][1]
        delta_t = v[2][1]-v[1][1]
        pred_x = x + delta_t*STP_S_4_each_1[str(objs_associate_info[i][0])]['mean_x']
        pred_y = y + delta_t*STP_S_4_each_1[str(objs_associate_info[i][0])]['mean_y']
        v.append([pred_x,pred_y])
        
    cam_1_x,cam_1_y = np.array([elem[3][0] for elem in objs_associate_info]),np.array([elem[3][1] for elem in objs_associate_info])
    cam_2_x,cam_2_y = np.array([elem[2][0][0] for elem in objs_associate_info]),np.array([elem[2][0][1] for elem in objs_associate_info])
    
    # for i in range(len(src_y)):
        # print(cam_1_y[i],cam_2_y[i])
    # plt.scatter(src_y,dst_y)
    # plt.show()
    
    cam_1_x = np.reshape(cam_1_x,(-1,1))
    cam_2_x = np.reshape(cam_2_x,(-1,1))
    cam_1_y = np.reshape(cam_1_y,(-1,1))
    cam_2_y = np.reshape(cam_2_y,(-1,1))
    
    # pred_x = model_x.predict(src_x)
    # pred_y = model_y.predict(src_y)
    
    # plt.plot(src_y,dst_y,'b-',linewidth=3)
    # plt.scatter(pred_x,dst_x)
    # plt.show()
    
    # for i in range(len(pred_y)):
        # print(pred_y[i],dst_y[i])
    
    # err = pred_x-dst_x
    # e_i = np.linspace(1,len(src_x),len(src_x))
    # plt.plot(e_i,err,'r',linewidth=3)
    # plt.show()
    
    
    # # ===== FAIL: 误差较大,原因未查明,暂时放弃 =====
    # # Get affine transform matrix using least square algorithm
    # trans_matrix,_ = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,1.0)
    # input_vector = np.array(src_pts)
    # pt = cv2.transform(input_vector[None,:,:],trans_matrix)
    # M = np.mat(trans_matrix[:2])
    # for elem in src_pts:
        # im = np.mat((elem[0],elem[1],1)).T
        # rim = M*im
        # print(rim)

    return cam_1_x, cam_2_x, cam_1_y, cam_2_y
    
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

def get_probability_map(obj_nx,obj_ny,base_x,base_y,start_x=0,start_y=0,height=300,width=80):

    x = np.linspace(start_x,start_x+width/10,width)
    y = np.linspace(start_y,start_y+height/10,height)

    delta_x = x - base_x 
    delta_y = y - base_y
    
    p_x = obj_nx.get_probability(delta_x)
    p_y = obj_ny.get_probability(delta_y)
    # # ----- TEST:单变量分布显示 -----
    # plt.plot(base_x,p_x,'b-',linewidth=3)
    # plt.plot(base_y,p_y,'r-',linewidth=3)
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
    if mx-mn > 0:
        n_data = np.array((data-mn_mat)*255/(mx-mn),dtype=np.uint8)
    else:
        n_data = np.zeros_like(data)
    
    # # # ----- TEST: n_data disp -----
    # print("max:",np.max(n_data))
    # print("min:",np.min(n_data))
    # print("shape:",n_data.shape) 
    # print("where:",len(np.where(n_data>1)[0]))
    # cv2.imshow('n_data',n_data)
    # cv2.waitKey()

    return n_data
    
def get_rational_value(x,low,high):
    '''Limit input in range'''
    y = np.max([np.min([x,high]),0])
    return y

# ========== TEST FUNCTIONS ===========
def gaussian_test():
    '''Useless for this task, just for testing the effect of gaussian distribution'''
    print("===== Get in the gaussian_test! ===== ")
    
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

def match_based_on_spatial_temperal_prior_test_1(tracker_record,pt_obj,t_interval=30):
    '''test location predicting of object,
    Predict object location in future based on the current object location, movement characteristics and time interval setting
    Parameters:
        tracker_record: tracking information
        pt_obj: perspective transformer object
        t_interval: time interval(frames)
    '''
    print("===== Get in the match_based_on_spatial_temperal_prior_test_1! ===== ")
    
    test_id = '1'
    # file path
    img_root = os.path.join(dataset_root,device_info[int(test_id)-1][0])
    save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data\resutls\loc_predicate\Frame20'

    
    obj_STP = Single_camera_STP(tracker_record,pt_obj,t_interval)
    # obj_STP.update_predictor(tracker_record,30)
       
    # test on id 1
    for v,elem in enumerate(obj_STP.perspective_trace[test_id]):
        if v+t_interval>len(obj_STP.perspective_trace[test_id])-1:
            break
        fname_1 = str(elem[1])+'.jpg'
        fname_2 = str(elem[1]+t_interval)+'.jpg'
        

        test_x,test_y = obj_STP.perspective_trace[test_id][v+t_interval][0][0],obj_STP.perspective_trace[test_id][v+t_interval][0][1]
        print("Probability:",obj_STP.get_probability(test_x,test_y,elem[0][0],elem[0][1]))
        print("Distance:",obj_STP.get_distance(test_x,test_y,elem[0][0],elem[0][1]))
        p_map = obj_STP.get_probability_map(base_x=elem[0][0],base_y=elem[0][1]) 
        p_map = cv2.applyColorMap(p_map,cv2.COLORMAP_JET)

        color_p_map = cv2.resize(p_map,(int(pt_obj.transformed_width_for_disp),int(pt_obj.transformed_height_for_disp)))

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
        # cv2.imwrite(os.path.join(save_root,fname_1),img_3)
        
        cv2.waitKey()
    return
    
def match_based_on_spatial_temperal_prior_test_2(tracker_record_1,tracker_record_2,pt_obj_1,pt_obj_2,associate_dict,t_interval=30):
    '''test location predicting of object,
    Predict object location in future based on current object location, movement characteristics and time interval setting
    Parameters:
        tracker_record_1: Trace information in cam 1(front one);
        tracker_record_2: Trace information in cam 2(back one);
        pt_obj_1: perspective transformer object of cam 1;
        pt_obj_2: perspective transformer object of cam 2;
        t_interval: time interval(frames)
    '''
    print("===== Get in the match_based_on_spatial_temperal_prior_test_2! ===== ")
    
    # file path
    img_root_1 = r'E:\DataSet\trajectory\concatVD\wuqi1B'
    img_root_2 = r'E:\DataSet\trajectory\concatVD\wuqi2B'
    save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\gen_data\resutls\loc_predicate_multi'
 
    obj_single_camera_STP_cam_1 = Single_camera_STP(tracker_record_1,pt_obj_1)
    obj_single_camera_STP_cam_2 = Single_camera_STP(tracker_record_2,pt_obj_2)
    
    # print(obj_single_camera_STP_cam_1.perspective_trace)
    # print(obj_single_camera_STP_cam_1.motion_params_4_each)
    obj_multi_cameras_STP_c1c2 = Multi_cameras_STP( 
                                    obj_single_camera_STP_cam_1,
                                    obj_single_camera_STP_cam_2,
                                    associate_dict)

    # # ===== TEST:coord_transformer_test =====
    # coord_transformer_test(obj_multi_cameras_STP_c1c2)
    # obj_multi_cameras_STP_c1c2.get_start_point_transform()
    
    pt_box_info_1 = obj_multi_cameras_STP_c1c2.obj_single_camera_STP_cam_1.perspective_trace
    pt_box_info_2 = obj_multi_cameras_STP_c1c2.obj_single_camera_STP_cam_2.perspective_trace
    
    # Test on object id '1'
    object_id = '0'
    
    for i in range(np.min([len(pt_box_info_1[object_id]),len(pt_box_info_2[object_id])])):
        f1 = i
        f2 = i
        fname_1 = str(pt_box_info_1[object_id][f1][1])+'.jpg'
        fname_2 = str(pt_box_info_2[object_id][f2][1])+'.jpg'
        
        img_1 = cv2.imread(os.path.join(img_root_1,fname_1))
        img_2 = cv2.imread(os.path.join(img_root_2,fname_2))
        
        cam_1_x = pt_box_info_1[object_id][f1][0][0]
        cam_1_y = pt_box_info_1[object_id][f1][0][1]
        
        cam_2_x = pt_box_info_2[object_id][f2][0][0]
        cam_2_y = pt_box_info_2[object_id][f2][0][1]
        
        t_interval = pt_box_info_2[object_id][f2][1]-pt_box_info_1[object_id][f1][1]
        
        print(cam_1_x,cam_1_y)
        print(cam_2_x,cam_2_y)
        print(t_interval)
        # print(obj_multi_cameras_STP_c1c2.starting_point)
        
        p_map = obj_multi_cameras_STP_c1c2.get_probability_map(cam_1_x,cam_1_y,t_interval,height=210,width=80)
        p_map = cv2.applyColorMap(p_map,cv2.COLORMAP_JET)
        p=obj_multi_cameras_STP_c1c2.get_probability(cam_2_x,cam_2_y,cam_1_x,cam_1_y,t_interval)
        print(p)
        # dist = obj_multi_cameras_STP_c1c2.get_distance(cam_2_x,cam_2_y,cam_1_x,cam_1_y,t_interval)
        p_map = cv2.resize(p_map,(int(pt_obj_2.transformed_width_for_disp),int(pt_obj_2.transformed_height_for_disp)))
        p_map = cv2.flip(p_map,0)   # 0:vertical flip
        pt_color_p_map = pt_obj_2.get_inverse_disp_transform(p_map)
        
        alpha = 0.5
        img_3 = cv2.addWeighted(img_2, alpha, pt_color_p_map, 1-alpha, 0)
        
        img_4 = np.zeros((int(img_2.shape[0]),int(img_2.shape[1]*2),3),np.uint8)
        img_4[:,:img_1.shape[1],:] = img_1
        img_4[:,img_1.shape[1]:,:] = img_3
        
        # cv2.namedWindow('img_1',cv2.WINDOW_NORMAL)
        # cv2.namedWindow('img_2',cv2.WINDOW_NORMAL)
        cv2.namedWindow('img_4',cv2.WINDOW_NORMAL)
        
        # cv2.imshow('img_1',img_1)
        # cv2.imshow('img_2',img_2)
        cv2.imshow('img_4',img_4)
        
        cv2.imwrite(os.path.join(save_root,fname_1),img_4)
        
        cv2.waitKey()
    return 

def trace_display_test(tracker_record,obj_id='1'):
    BBox,BFrame = zip(*tracker_record[obj_id]['list'])
    Center_BBox = get_box_center(BBox)
    print(Center_BBox)
    img = cv2.imread(r'E:\DataSet\trajectory\concatVD\wuqi1B\0.jpg')
    for elem in Center_BBox:
        print(elem)
        cv2.circle(img,(int(elem[0]),int(elem[1])),10,(0,0,255),10)
    cv2.namedWindow('img_1',cv2.WINDOW_NORMAL)
    cv2.imshow('img_1',img)
    cv2.waitKey()
   
def coord_transformer_test(obj_multi_cameras_STP):
    transformers = obj_multi_cameras_STP.get_coord_transformer()
    print("===== Get in coord_transformer_test！=====")
    print(transformers['B2F']['y'].predict(obj_multi_cameras_STP.coord_pairs[3]))
   
def useless():
    # ===== TEST: 1D gassian display =====
    # x = np.linspace(-1,1,100)
    # y = obj.get_probability(x)

    # plt.plot(x,y,'b-',linewidth=3)
    # plt.show()

    # ===== TEST: 2D gaussian display =====
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
    
    # ===== TEST: 2D gaussian display in 3D effect =====
    # with sns.axes_style("dark"):
        # sns.jointplot(display_x, display_y, kind="kde",space=0)
    # plt.show()
    
    # pt_tracker_record_1 = {}
    # pt_tracker_record_2 = {}
    pass
    
# ========== MAIN FUNCTIONS ===========
if __name__=="__main__":
    print('==== Cameras_associate ====')
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

    from data_generator import load_tracking_info
    tracker_record_1 = load_tracking_info(img_savepath_1)
    tracker_record_2 = load_tracking_info(img_savepath_2)
    
    import Perspective_transform as pt
    pt_obj_1 = pt.Perspective_transformer(pt_savepath_1)
    pt_obj_2 = pt.Perspective_transformer(pt_savepath_2)
    
    pt_box_info_1 = get_pt_box_info(tracker_record_1,pt_obj_1)
    pt_box_info_2 = get_pt_box_info(tracker_record_2,pt_obj_2)
    
    # # ======= TEST: trace display ======
    # trace_display_test(tracker_record_1,obj_id='4')
    
    # ====== TEST: location predicting ======
    # match_based_on_spatial_temperal_prior_test_1( tracker_record_1, 
                                                    # pt_obj_1,
                                                    # 20)
    match_based_on_spatial_temperal_prior_test_2(   tracker_record_1,
                                                    tracker_record_2,
                                                    pt_obj_1,
                                                    pt_obj_2,
                                                    associate_dict_c1_c2,
                                                    )

    # ====== TEST: Spatial-Temperal Prior (STP) in single camera ======
    # d = get_STP_in_single_camera(pt_box_info_1)
    # d = get_STP_in_multi_cameras(pt_box_info_1,pt_box_info_2,associate_dict_c1_c2)
 
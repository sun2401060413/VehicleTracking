'''
SCT: Single camera tracking.
Multi-objects tracking with single camera.
written by sunzhu, 2019-03-19, version 1.0
'''

import os,sys
import cv2
import json
import numpy as np


        
class Perspective_transformer(object):
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.get_perspective_info()
        
    def get_perspective_info(self):
        with open(self.json_file_path, 'r') as doc:
            data = json.load(doc)
            self.transform_matrix_for_disp = data['transform_matrix_for_disp']
            self.transform_matrix_for_pred = data['transform_matrix_for_pred']
            self.original_img_width = data['original_img_width']
            self.original_img_height = data['original_img_height']
            self.transformed_height_for_disp = data['transformed_height_for_disp']
            self.transformed_width_for_disp = data['transformed_width_for_disp']
            self.transformed_height_for_pred = data['transformed_height_for_pred']
            self.transformed_width_for_pred = data['transformed_width_for_pred']
            self.endpoints = data['endpoints']
            
    def get_disp_transform(self, img):
        '''img to img'''
        prespective = cv2.warpPerspective(
                        img,
                        np.array(self.transform_matrix_for_disp),
                        (int(self.transformed_width_for_disp), int(self.transformed_height_for_disp)),
                        cv2.INTER_LINEAR)
        return prespective
        
    def get_pred_transform(self, pt_list):
        ''' point to point '''
        input_vector = np.array(pt_list)
        return  cv2.perspectiveTransform(input_vector[None, :, :], np.array(self.transform_matrix_for_pred))
        
    def get_inverse_disp_transform(self, img):
        '''inverse perspective transform for displaying'''
        inverse_matrix = np.linalg.inv(np.mat(self.transform_matrix_for_disp))
        inverse_prespective = cv2.warpPerspective(
                        img,
                        np.array(inverse_matrix),
                        (int(self.original_img_width), int(self.original_img_height)),
                        cv2.INTER_LINEAR)
        return inverse_prespective
        
    def get_inverse_pred_transform(self, pt_list):
        '''Inverse perspective transform for prediction'''
        inverse_matrix = np.linalg.inv(np.mat(self.transform_matrix_for_pred))
        input_vector = np.array(pt_list)
        return cv2.perspectiveTransform(input_vector[None, :, :], inverse_matrix)
        
def draw_init_ROI(img_path, savepath=None, top=450, bottom=-50, left=50, right=-50):
    '''用于匹配轨迹显示与透视变换区域设置
        call instance:    draw_init_ROI(r"D:\Project\tensorflow_model\VehicleTracking\data_generator\ROItools\cam_4.jpg")
    '''
    fileroot, filename = os.path.split(img_path)
    img = cv2.imread(img_path)
    height, width = img.shape[0], img.shape[1]
    cv2.line(img, (0, top), (width-1, top), (255, 255, 255), 5)
    cv2.line(img, (0, height+bottom), (width-1, height+bottom), (255, 255, 255), 5)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)
    if savepath is None:
        savepath = fileroot
    savefilename = 'ROI_'+filename
    cv2.imwrite(os.path.join(savepath, savefilename), img)
    cv2.waitKey()
    return
    
    
    

if __name__=="__main__":
    # # root path
    # dataset_root = r"E:\DataSet\trajectory\concatVD"
    #
    # # save root
    # save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\my_data'
    #
    # # perspective transform data
    # persp_trans_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\ROItools\data'
    #
    #
    # # images
    # cam_1 = 'wuqi1B'
    # cam_2 = 'wuqi2B'
    # cam_3 = 'wuqi3B'
    # cam_4 = 'wuqi4B'
    #
    # # bbox information
    # box_info_1 = 'wuqiyuce1.csv'
    # box_info_2 = 'wuqiyuce2.csv'
    # box_info_3 = 'wuqiyuce3.csv'
    # box_info_4 = 'wuqiyuce4.csv'
    #
    # # prespective information
    # perspective_info_1 = 'ROI_cam_1_transformer.json'
    # perspective_info_2 = 'ROI_cam_2_transformer.json'
    # perspective_info_3 = 'ROI_cam_3_transformer.json'
    # perspective_info_4 = 'ROI_cam_4_transformer.json'

    # ----- inverse perspective transform for disp test 1 -----
    # test_img = cv2.imread(r'D:\Project\tensorflow_model\VehicleTracking\data_generator\ROItools\data\inverse_img_1.jpg')
    # obj = Perspective_transformer(os.path.join(persp_trans_root,perspective_info_1))
    # persp_img = obj.get_inverse_disp_transform(test_img)
    # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    # cv2.imshow('img',persp_img)
    # cv2.waitKey()

    # ----- inverse perspective transform for pred test 1 -----
    # test_img = cv2.imread(r'E:\DataSet\trajectory\concatVD\wuqi1B\0.jpg')
    # obj = Perspective_transformer(os.path.join(persp_trans_root,perspective_info_1))
    # persp_pred = obj.get_inverse_pred_transform([[0.0,0.0],[0.0,30.0],[8.0,0.0],[8.0,30.0]])
    # for elem in persp_pred[0]:
        # print(elem)
        # cv2.circle(test_img,(int(elem[0]),int(elem[1])),10,(0,0,255),10)
    # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    # cv2.imshow('img',test_img)
    # cv2.waitKey()
    
    # ----- perspective transform test 1 -----
    # test_img = cv2.imread(r'E:\DataSet\trajectory\concatVD\wuqi4B\0.jpg')
    # obj = Perspective_transformer(os.path.join(persp_trans_root,perspective_info_1))
    # persp_img = obj.get_disp_transform(test_img)
    # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    # cv2.imshow('img',persp_img)
    # cv2.waitKey()
    
    # ----- perspective transform test 2 -----

    # obj = Perspective_transformer(os.path.join(persp_trans_root,perspective_info_4))
    obj = Perspective_transformer(r"E:\Project\CV\Data\settings\0001_transformer.json")
    persp_list = obj.get_pred_transform(obj.endpoints)
    print("obj.endpoints:", obj.endpoints)
    print("persp_list", persp_list)

    #### Results:#####
    # obj.endpoints: [[412.0, 672.0], [566.0, 622.0], [1504.0, 750.0], [1398.0, 880.0]]
    # persp_list[[[-9.04537628e-15 - 0.00000000e+00]
    #             [9.00000000e+00 - 2.00559198e-14]
    #             [9.00000000e+00  3.00000000e+01]
    # [-4.44645781e-15
    # 3.00000000e+01]]]


    # ----- endpoints display -----
    # for elem in obj.endpoints:
        # cv2.circle(test_img,(int(elem[0]),int(elem[1])),10,(0,0,255),10)
    # cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    # cv2.imshow("img",test_img)
    # cv2.waitKey()
    
    # ----- disp transform test -----
    # trans_img = obj.get_disp_transform(test_img)
    # cv2.imshow('trans_img',trans_img)
    # cv2.waitKey()
    
    # ----- pred transform test -----
    # # print('obj.endpoints:',obj.endpoints)
    # # trans_point = obj.get_pred_transform(obj.endpoints)
    # # print('trans_point:',trans_point)

    # ----- info read test -----
    # # print(obj.transform_matrix_for_disp)
    # # print(obj.transform_matrix_for_pred)
    # # print(obj.original_img_width)
    # # print(obj.original_img_height)
    # # print(obj.transformed_height_for_disp)
    # # print(obj.transformed_width_for_disp)
    # # print(obj.transformed_height_for_pred)
    # # print(obj.transformed_width_for_pred)
    # # print(obj.endpoints)
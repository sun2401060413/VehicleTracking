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
    def __init__(self,json_file_path):
        self.json_file_path = json_file_path
        self.get_perspective_info()
        
    def get_perspective_info(self):
        with open(self.json_file_path,'r') as doc:
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
            
    def get_disp_transform(self,img):
        prespective = cv2.warpPerspective(
                        img,
                        np.array(self.transform_matrix_for_disp),
                        (int(self.transformed_width_for_disp),int(self.transformed_height_for_disp)),
                        cv2.INTER_LINEAR)
        return prespective
        
    def get_pred_transform(self,pt_list):
        input_vector = np.array(pt_list)
        return  cv2.perspectiveTransform(input_vector[None,:,:], np.array(self.transform_matrix_for_pred))
        

if __name__=="__main__":
    # root path
    dataset_root = r"E:\DataSet\trajectory\concatVD"

    # save root
    save_root = r'D:\Project\tensorflow_model\VehicleTracking\data_generator\my_data'

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
    
    # prespective information
    perspective_info_1 = 'wuqiyuce1.json'
    perspective_info_2 = 'wuqiyuce2.json'
    perspective_info_3 = 'wuqiyuce3.json'
    perspective_info_4 = 'wuqiyuce4.json'
    
    
    # img
    test_img = cv2.imread(r'E:\DataSet\trajectory\concatVD\wuqi1B\0.jpg')
    obj = Perspective_transformer(os.path.join(dataset_root,perspective_info_1))
    
    # ----- endpoints display -----
    # # for elem in obj.endpoints:
        # # cv2.circle(test_img,(int(elem[0]),int(elem[1])),10,(0,0,255),10)
    # # cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    # # cv2.imshow("img",test_img)
    # # cv2.waitKey()
    
    # ----- disp transform test -----
    # # trans_img = obj.get_disp_transform(test_img)
    # # cv2.imshow('trans_img',trans_img)
    # # cv2.waitKey()
    
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
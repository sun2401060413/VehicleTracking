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
        
    def get_pred_transform(self, pt_list, w_scale=1, h_scale=1):
        ''' point to point '''
        input_vector = np.array(pt_list)
        trans_vector = cv2.perspectiveTransform(input_vector[None, :, :], np.array(self.transform_matrix_for_pred))
        if w_scale != 1:
            trans_vector[:, :, 0] = trans_vector[:, :, 0]*-1 + self.transformed_width_for_pred
        if h_scale != 1:
            trans_vector[:, :, 1] = trans_vector[:, :, 1]*-1 + self.transformed_height_for_pred
        return trans_vector
        
    def get_inverse_disp_transform(self, img):
        '''inverse perspective transform for displaying'''
        inverse_matrix = np.linalg.inv(np.mat(self.transform_matrix_for_disp))
        inverse_prespective = cv2.warpPerspective(
                        img,
                        np.array(inverse_matrix),
                        (int(self.original_img_width), int(self.original_img_height)),
                        cv2.INTER_LINEAR)
        return inverse_prespective
        
    def get_inverse_pred_transform(self, pt_list, w_scale=1, h_scale=1):
        '''Inverse perspective transform for prediction'''
        inverse_matrix = np.linalg.inv(np.mat(self.transform_matrix_for_pred))
        input_vector = np.array(pt_list)
        if w_scale != 1:
            input_vector[:, 0] = input_vector[:, 0]*-1 + self.transformed_width_for_pred
        if h_scale != 1:
            input_vector[:, 1] = input_vector[:, 1]*-1 + self.transformed_height_for_pred
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
    from Common import ROI_RESULT
    from Common import cam_names, data_path, box_info, roi_info, track_info, save_path

    # ===== inverse perspective transform for disp test 1 =====
    test_img = cv2.imread(os.path.join(ROI_RESULT, "001\\reverse_pt_img.png"))
    obj = Perspective_transformer(roi_info[0])
    persp_img = obj.get_inverse_disp_transform(test_img)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', persp_img)
    cv2.waitKey()

    # ===== inverse perspective transform for pred test 1 =====
    # test_img = cv2.imread(os.path.join(data_path[0], "0001.jpg"))
    # obj = Perspective_transformer(roi_info[0])
    # print(obj.endpoints)
    # z_img = np.zeros((obj.original_img_height, obj.original_img_width, 3), np.uint8)
    # for v, elem in enumerate(obj.endpoints):
    #     cv2.circle(z_img, (int(elem[0]), int(elem[1])), 2, (255, 255, 255), 2)
    #     cv2.putText(z_img, "{}".format(v), (int(elem[0])+5, int(elem[1])+5)
    #                 , cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 255, 255), 3)
    # persp_pred = obj.get_inverse_pred_transform([[0.0, 0.0], [0.0, 30.0], [8.0, 0.0], [8.0, 30.0]])
    # for elem in persp_pred[0]:
    #     print(elem)
    #     cv2.circle(test_img, (int(elem[0]), int(elem[1])), 10, (0, 0, 255), 10)
    # cv2.namedWindow('z_img', cv2.WINDOW_NORMAL)
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('z_img', z_img)
    # cv2.imshow('img', test_img)
    # cv2.waitKey()

    
    # ===== perspective transform test 1 =====
    # test_img = cv2.imread(os.path.join(data_path[0], "0001.jpg"))
    # obj = Perspective_transformer(roi_info[0])
    # persp_img = obj.get_disp_transform(test_img)
    # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # cv2.imshow('img', persp_img)
    # cv2.waitKey()
    
    # ===== perspective transform test 2 =====
    # obj = Perspective_transformer(roi_info[0])
    # persp_list = obj.get_pred_transform(obj.endpoints, h_scale=-1)
    # print("obj.endpoints:", obj.endpoints)
    # print("persp_list", persp_list)

    #### Results:#####
    # obj.endpoints: [[412.0, 672.0], [566.0, 622.0], [1504.0, 750.0], [1398.0, 880.0]]
    # persp_list[[[-9.04537628e-15 - 0.00000000e+00]
    #             [9.00000000e+00 - 2.00559198e-14]
    #             [9.00000000e+00  3.00000000e+01]
    #             [-4.44645781e-15 3.00000000e+01]]]


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
"""
SCT: Single camera tracking.
Multi-objects tracking in single camera.
written by sunzhu on 2019-03-19, version 1.0
Updated by sunzhu on 2020-08-18, version 1.1
Updated by sunzhu on 2020-09-29, version 1.2
"""

import os
import sys
import pandas as pd
import cv2
import json
import numpy as np
import operator
from scipy.optimize import linear_sum_assignment

from Common import COLORS as colors
from Common import ROOT, SRC_IMAGES, DET_RESULT, ROI_RESULT, SCT_RESULT, FRAME_RATE
from data_generator import DataGenerator
# ==== Files path setting, you could comment the code line below and set your own path ===-
from Common import cam_names, data_path, box_info, roi_info, save_path, track_info


# # ===== UTILS FUNCTIONS =====
def is_contact_with_image(img_height, img_width, box_x1, box_y1, box_x2, box_y2, thresh=5):
    ''' checking the box contact with image boundaries or not '''
    if box_x1 <= thresh or box_x2 >= img_width - thresh or box_y1 <= thresh or box_y2 >= img_height - thresh:
        return True
    else:
        return False


def iou(box1, box2):
    """Intersection over union"""
    x1, y1, x3, y3, width1, height1 = box1[0], box1[1], box1[2], box1[3], box1[2] - box1[0], box1[3] - box1[1]
    x2, y2, x4, y4, width2, height2 = box2[0], box2[1], box2[2], box2[3], box2[2] - box2[0], box2[3] - box2[1]
    # Intersection
    i_width = width1 + width2 - (max(x3, x4) - min(x1, x2))
    i_height = height1 + height2 - (max(y3, y4) - min(y1, y2))

    if i_width <= 0 or i_height <= 0:
        ret = 0
    else:
        i_area = i_width * i_height  # intersection area
        area1 = width1 * height1
        area2 = width2 * height2
        o_area = area1 + area2 - i_area  # union area
        ret = i_area * 1. / o_area  # intersection over union
    return ret


def get_box_center(box, mode="both", ratio_x=0.5, ratio_y=0.5):
    '''Updated on 2020-08-19'''
    if mode == "both":
        return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
    if mode == "bottom":
        return int((box[0] + box[2]) / 2), int(box[3])
    if mode == "ratio":
        return int(box[0] + ratio_x * abs(box[2] - box[0])), int(box[1] + ratio_y * abs(box[3] - box[1]))


def get_offset_box(box, offset=0):
    '''Resize bounding box, offset refers the distance'''
    return [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset]


# # ===== CLASSES ======
class VehicleObject(object):
    def __init__(self, id):
        self.list = []
        self.image = []
        self.id = id
        self.filter = None

        self.first_box = None  # image space position
        self.first_box_ct_in_world = None  # world space position
        self.first_frame = None  # frame number
        self.first_img = None  # object img

        self.large_img = None   # max img of same obj in record

        self.last_box = None  # image space position
        self.last_box_ct_in_world = None  # world space position
        self.last_frame = None  # frame number

        # optional, used in filter
        self.prediction = None
        self.trace = []

        self.update_status = True
        self.speed = 0  # speed in world space
        self.color = None

    def set_first_frame(self, box=None, frame=None, img=None, transformed_box_ct=None):
        """ create a new object record
        :param box: bounding box info
        :param frame: frame number
        :param img: cropped object img
        :param transformed_box_ct: world pos
        :return: None """
        self.first_box = box
        self.first_frame = frame
        self.first_img = img
        self.large_img = img
        if transformed_box_ct is not None:
            self.first_box_ct_in_world = transformed_box_ct

    def update(self, box=None, frame=None, img=None, transformed_box_ct=None):
        """ Update record
        :param box: bounding box info
        :param frame: frame number
        :param img: cropped object img
        :param transformed_box_ct:
        :return: None
        """

        self.list.append([[int(elem) for elem in box], int(frame)])  # int32 无法在json中序列化
        if img is not None:
            # self.image.append(img)
            self.large_img = self.get_larger_img(im_1=self.first_img, im_2=img)
        self.last_box = box
        self.last_frame = frame
        if transformed_box_ct is not None:
            self.last_box_ct_in_world = transformed_box_ct

        if not operator.eq(self.first_box_ct_in_world, self.last_box_ct_in_world).all():
            self.speed = self.get_avgspeed()

    def set_color(self, color):
        self.color = color

    def get_avgspeed(self):
        """Calculate the average speed
        :return: object speed in world space
        """
        if self.last_frame != self.first_frame:
            # position instance: v1: [[[x1,y1]]], v2: [[[x2,y2]]]
            s_distance = np.linalg.norm(self.last_box_ct_in_world.flatten() - self.first_box_ct_in_world.flatten())
            t_interval = self.last_frame - self.first_frame
            self.speed = (s_distance / t_interval) * FRAME_RATE * 3.6  # km/h
            return self.speed
        else:
            return 0

    @staticmethod
    def get_larger_img(im_1, im_2):
        if im_1.shape[0] + im_1.shape[1] > im_2.shape[0] + im_2.shape[1]:
            return im_1
        else:
            return im_2


def rank(objs_list, weight_spatial=0.5, weight_temporal=0.5):
    """ find the nearest object in spatial and temporal space, the default weights of two space are 0.5 and 0.5 """
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

class Tracker(object):
    """
    base class of tracker
    """
    def __init__(self,
                 frame_space_dist=10,
                 transformer=None,
                 ):
        self.current_frame_img = None

        # objects pool, all tracked objects are saved in this dict
        self.objects_pool = {}
        self.hist_objects_pool = {}
        self.objects_count = 0

        # threshvalue for tracking
        self.frame_space_dist = frame_space_dist  # ignore the object with long time interval

        self.hist_objects_record_flag = True  # Record the history information or not
        self.hist_objects_img_record_flag = False  # Record the hist objects image or not
        self.image_record_flag = True  # save image or not

        # image info
        self.img_height = 1080
        self.img_width = 1920
        self.obj_pool_display_height = 100  # objects_pool display height;
        self.obj_pool_display_width = 100  # objects_pool display width;
        self.obj_pool_display_channel = 3

        # display setting
        self.display_monitor_region = True

        # Coodinate transformer
        self.transformer = transformer
        self.polygonpts = self.set_polygon_region(transformer.endpoints)

    def get_available_id(self):
        """Give each object an id"""
        out_put = self.objects_count
        if self.objects_count < 100000:
            self.objects_count += 1
        else:
            self.reset_available_id()  # reset the count
        return out_put  # The max id is 99999

    def reset_available_id(self):
        """Re-number the id"""
        self.objects_count = 0

    @staticmethod
    def get_available_color(obj_id):
        """Give each object an available color"""
        i = obj_id % len(colors)
        return colors[i]

    def is_track_finish(self, frame):
        """Check that the object is still in monitoring region"""
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
        """ Update the tracking info
            ==== This is a basic implementation for updating, you can re-write it in subclasses =====.
        """
        box_info = [box[0], box[1], box[2], box[3]]
        # if self.isBoxInRegion(box_info):
        # box_ct = get_box_center(box, mode="ratio", ratio_x=0.5, ratio_y=0.75)

        box_ct = get_box_center(box, mode="bottom")

        input_box_ct = np.array([list(box_ct)]).astype(float).tolist()

        # transformed point
        if self.transformer is not None:
            # cur_pos = self.transformer.get_pred_transform(input_box_ct, h_scale=-1)
            cur_pos = self.transformer.get_pred_transform(input_box_ct)
        else:
            cur_pos = None

        if self.isBoxInPolygonRegion(box_ct) > 0:
            frame_info = box[4]

            matched_obj = self.match(box_info, frame_info)
            if matched_obj:
                self.objects_pool[matched_obj.id].update(box=box_info, frame=frame_info, transformed_box_ct=cur_pos,
                                                         img=img)
            else:
                obj_id = self.get_available_id()
                obj = VehicleObject(obj_id)  # create a new vehicle object
                obj.set_first_frame(box=box_info, frame=frame_info, img=img, transformed_box_ct=cur_pos)
                obj.set_color(self.get_available_color(obj_id=obj_id))  # set color for displaying
                obj.update(box=box_info, frame=frame_info, transformed_box_ct=cur_pos)
                self.objects_pool[obj_id] = obj
        del_objs = self.is_track_finish(box[4])
        if self.hist_objects_record_flag and del_objs:
            for elem in del_objs:
                self.hist_objects_pool[elem] = del_objs[elem]

    def match(self, box, frame):
        """This is an abstrat function, implement details in subclasses"""
        return None

    def isBoxInRegion(self, box):
        """Check a object in the setting region or not
            Note: In vertical direction, we take the box bottom as a reference to check the present of object.
        """
        if box[0] > self.region_left and box[2] < self.img_width - self.region_right and box[3] > self.region_top and \
                box[3] < self.img_height - self.region_bottom:
            return True
        else:
            return False

    def isBoxInPolygonRegion(self, box):
        """Check a object in the setting region or not
            Note: the region is a polygon.
        """
        pt = box
        return cv2.pointPolygonTest(self.polygonpts, pt, measureDist=False)

    def set_polygon_region(self, pts):
        """[[a,b],[c,d],[e,f],[g,h]] → [np.array([[a,b]],[[c,d]]....)]"""
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
        """Draw the tracking results"""
        if self.display_monitor_region:
            cv2.drawContours(img, [self.polygonpts], -1, (0, 0, 255), 3)
        for k, v in self.objects_pool.items():  # draw all objects in the pool.
            if v.update_status:
                if len(v.list) > 1:
                    for i in range(len(v.list) - 1):
                        # center_1 = get_box_center(v.list[i][0], mode="ratio", ratio_x=0.5, ratio_y=0.75)
                        center_1 = get_box_center(v.list[i][0], mode="bottom")
                        # center_2 = get_box_center(v.list[i + 1][0], mode="ratio", ratio_x=0.5, ratio_y=0.75)
                        center_2 = get_box_center(v.list[i + 1][0], mode="bottom")
                        cv2.line(img, center_1, center_2, v.color, 5)
                    cv2.rectangle(img, (int(v.last_box[0]), int(v.last_box[1])), (int(v.last_box[2]), int(v.last_box[3])), v.color, 3)
                    cv2.putText(img, "ID:{}".format(v.id), (int(v.last_box[2]), int(v.last_box[3]))
                                , cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, v.color, 3)
                    print("v.speed", v.speed)
                    cv2.putText(img, "{}km/h".format(round(v.speed), 1), (int(v.last_box[2]), int(v.last_box[3] + 30))
                                , cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, v.color, 3)
        return img

    def draw_objects_pool(self):
        """Draw the objects pool"""
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
                obj_count += 1
            return disp_objs_pool_img
        else:
            return None

    def save_data(self, filepath):
        """Save tracking info"""
        if self.hist_objects_record_flag:
            filename = r"tracking_info.json"
            saved_info = {}
            for elem in self.hist_objects_pool:             # hist_objects
                tmp = {}
                tmp['id'] = self.hist_objects_pool[elem].id
                tmp['first_box'] = [int(v) for v in self.hist_objects_pool[elem].first_box]
                tmp['first_frame'] = int(self.hist_objects_pool[elem].first_frame)
                tmp['last_box'] = [int(v) for v in self.hist_objects_pool[elem].last_box]
                tmp['last_frame'] = int(self.hist_objects_pool[elem].last_frame)
                tmp['update_status'] = self.hist_objects_pool[elem].update_status
                tmp['color'] = self.hist_objects_pool[elem].color
                tmp['speed'] = self.hist_objects_pool[elem].speed
                tmp['list'] = self.hist_objects_pool[elem].list

                if self.hist_objects_img_record_flag:
                    img_path = os.path.join(filepath, "images\\id_{:0>4d}.jpg".format(int(self.hist_objects_pool[elem].id)))
                    cv2.imwrite(img_path, self.hist_objects_pool[elem].large_img)
                saved_info[elem] = tmp

            with open(os.path.join(filepath, filename), 'w') as doc:
                json.dump(saved_info, doc)
        else:
            return print("History record flag is False!")


class IOUTracker(Tracker):
    def __init__(self,
                 frame_space_dist=10,
                 transformer=None,
                 thresh_iou=0.2
                 ):
        super(IOUTracker, self).__init__(
            frame_space_dist=frame_space_dist,
            transformer=transformer
        )
        self.thresh_iou = thresh_iou

    @staticmethod
    def iou_tracker_version(self):
        print("Class IOUTracker, Version 1.2.0, Updated by SunZhu on Sep 29, 2020")

    def match(self, box, frame):
        """ Match objects in current frame """
        possible_obj_list = []
        for k, v in self.objects_pool.items():
            cmp_id = k
            cmp_locat = v.last_box
            cmp_frame = v.last_frame
            cmp_iou = iou(box, cmp_locat)
            cmp_frame_dist = frame - cmp_frame
            if cmp_iou >= self.thresh_iou and cmp_frame_dist <= self.frame_space_dist:
                possible_obj_list.append([v, cmp_iou, cmp_frame_dist * 1. / self.frame_space_dist])
        matched_obj = rank(possible_obj_list, 0.6, 0.4)
        return matched_obj


class STPTracker(Tracker):
    """STPTracker: Tracker based on Spatial-Temporal prior """
    def __init__(self,
                 frame_space_dist=50,
                 transformer=None,
                 match_mode='Prob',     # Mode:Prob/Dist
                 stp_prior=None):       # object of stp_prior
        super(STPTracker, self).__init__(
            frame_space_dist=frame_space_dist,
            transformer=transformer
        )

        self.last_append_id = None
        self.new_object_append_status = False

        # threshvalue for tracking
        self.thresh_probability = 0.001  # ignore the object with low probability
        self.thresh_distance = 2  # ignore the object with far distance
        # the value dependent on frame_space_dist

        # Single Camera Spatial-temporal prior
        self.stp_prior = stp_prior
        self.match_mode = match_mode

    @staticmethod
    def version(self):
        return print("===== Written by sunzhu on Sep 29, 2020, Version 1.2 =====")

    def match(self, box, frame):
        possible_obj_list = []
        for k, v in self.objects_pool.items():
            cmp_id = k
            cmp_locat = v.last_box
            cmp_frame = v.last_frame
            center_x, center_y = get_box_center(box, mode='bottom')
            base_center_x, base_center_y = get_box_center(cmp_locat, mode='bottom')

            pt_centers = self.stp_prior.perspective_transformer.get_pred_transform(
                np.array(
                    [[center_x, center_y],
                     [base_center_x, base_center_y]]
                    , np.float)
            )
            pt_center_x, pt_center_y = pt_centers[0][0]
            pt_base_center_x, pt_base_center_y = pt_centers[0][1]

            # # ==== TEST: Display the probability map =====
            # img_3 = self.draw_color_probability_map(img_current, pt_base_center_x, pt_base_center_y)
            # cv2.namedWindow("img_current", cv2.WINDOW_NORMAL)
            # cv2.imshow("img_current", img_3)
            # cv2.waitKey(1)

            cmp_frame_dist = frame - cmp_frame
            if self.match_mode == 'Prob':
                # cmp_result = self.stp_prior.get_probability(pt_center_x, pt_center_y, pt_base_center_x, pt_base_center_y)[2]
                # test: Visualization of prediction
                # self.display_probability_map(base_x=pt_base_center_x, base_y=pt_base_center_y)
                cmp_result = \
                self.stp_prior.get_probability(pt_center_x, pt_center_y, pt_base_center_x, pt_base_center_y)[2]
                if cmp_result >= self.thresh_probability and cmp_frame_dist <= self.frame_space_dist:
                    possible_obj_list.append([v, cmp_result, cmp_frame_dist * 1. / self.frame_space_dist])
            else:  # Dist mode
                cmp_result = self.stp_prior.get_distance(pt_center_x, pt_center_y, pt_base_center_x, pt_base_center_y)
                if cmp_result <= self.thresh_distance and cmp_frame_dist <= self.frame_space_dist:
                    possible_obj_list.append([v, cmp_result, cmp_frame_dist * 1. / self.frame_space_dist])
        matched_obj = self.rank(possible_obj_list)

        return matched_obj

    def rank(self, objs_list):
        """find the nearest object in spatial and temporal space, the default weights of two space are 0.5 and 0.5"""
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

    def display_probability_map(self,  base_x=0, base_y=0):
        """Display the probabiltiy map of prediction (For testing)"""
        p_map = self.stp_prior.get_probability_map(base_x=base_x, base_y=base_y, start_x=0, start_y=0, length_x=15, length_y=110, height=110, width=15)
        p_map = cv2.applyColorMap(p_map, cv2.COLORMAP_JET)
        color_p_map = cv2.resize(p_map, (int(self.transformer.transformed_width_for_disp), int(self.transformer.transformed_height_for_disp)))
        color_p_map = cv2.flip(color_p_map, 0)   # 0:vertical flip
        pt_color_p_map = self.transformer.get_inverse_disp_transform(color_p_map)
        alpha = 0.5

        dsp_pb_map = cv2.addWeighted(pt_color_p_map, alpha, self.current_frame_img, 1-alpha, 0)
        cv2.namedWindow("p_map", cv2.WINDOW_NORMAL)
        cv2.imshow("p_map", dsp_pb_map)
        cv2.waitKey()


class KalmanFilter(object):
    """Kalman Filter class
    Attributes: None
    """

    def __init__(self):
        """Initialize paras
        """
        self.dt = 0.005  # delta time

        self.A = np.array([[1, 0], [0, 1]])  # matrix in observation equations
        self.u = np.zeros((2, 1))  # previous state vector

        # (x,y) tracking object center
        self.b = np.array([[0], [255]])  # vector of observations

        self.P = np.diag((3.0, 3.0))  # covariance matrix
        self.F = np.array([[1.0, self.dt], [0.0, 1.0]])  # state transition mat

        self.Q = np.eye(self.u.shape[0])  # process noise matrix
        self.R = np.eye(self.b.shape[0])  # observation noise matrix
        self.lastResult = np.array([[0], [255]])

    def predict(self):
        """Predict state vector u and variance of uncertainty P (covariance).
            where,
            u: previous state vector
            P: previous covariance matrix
            F: state transition matrix
            Q: process noise matrix
        Equations:
            u'_{k|k-1} = Fu'_{k-1|k-1}
            P_{k|k-1} = FP_{k-1|k-1} F.T + Q
            where,
                F.T is F transpose
        Args:
            None
        Return:
            vector of predicted state estimate
        """
        # Predicted state estimate
        self.u = np.round(np.dot(self.F, self.u))
        # Predicted estimate covariance
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        self.lastResult = self.u  # same last predicted result
        return self.u

    def correct(self, b, flag):
        """Correct or update state vector u and variance of uncertainty P (covariance).
        where,
        u: predicted state vector u
        A: matrix in observation equations
        b: vector of observations
        P: predicted covariance matrix
        Q: process noise matrix
        R: observation noise matrix
        Equations:
            C = AP_{k|k-1} A.T + R
            K_{k} = P_{k|k-1} A.T(C.Inv)
            u'_{k|k} = u'_{k|k-1} + K_{k}(b_{k} - Au'_{k|k-1})
            P_{k|k} = P_{k|k-1} - K_{k}(CK.T)
            where,
                A.T is A transpose
                C.Inv is C inverse
        Args:
            b: vector of observations
            flag: if "true" prediction result will be updated else detection
        Return:
            predicted state vector u
        """

        if not flag:  # update using prediction
            self.b = self.lastResult
        else:  # update using detection
            self.b = b
        C = np.dot(self.A, np.dot(self.P, self.A.T)) + self.R
        K = np.dot(self.P, np.dot(self.A.T, np.linalg.inv(C)))

        self.u = np.round(self.u + np.dot(K, (self.b - np.dot(self.A,
                                                              self.u))))
        self.P = self.P - np.dot(K, np.dot(C, K.T))
        self.lastResult = self.u
        return self.u


class KLFTracker(Tracker):
    def __init__(self,
                 frame_space_dist=10,
                 transformer=None,
                 thresh_dist=100,
                 ):
        super(KLFTracker, self).__init__(
            frame_space_dist=frame_space_dist,
            transformer=transformer
        )
        # Do something
        # https://github.com/srianant/kalman_filter_multi_object_tracking/blob/master/tracker.py

        self.thresh_dist = thresh_dist
        self.objects_pool_elem_list = []
        self.assignment = []


    def update(self, boxes, img=None):
        """ Update the tracking info
            solve the assignment problem by using Hungarian Algorithm
        """

        boxes_in_region = self.get_boxes_in_region(boxes)
        if len(boxes_in_region) > 0:
            frame = boxes_in_region[0][0][4]

            trace_count = len(self.objects_pool)        # current trace count
            det_count = len(boxes_in_region)            # current detected object count

            self.objects_pool_elem_list = [elem for elem in self.objects_pool]

            # cost matrix
            print("==========")
            cost = np.zeros(shape=(trace_count, det_count))
            for i in range(len(self.objects_pool_elem_list)):
                for j in range(det_count):
                    try:
                        diff = np.array(self.objects_pool[self.objects_pool_elem_list[i]].prediction) - np.array(boxes_in_region[j][1])  # 误差
                        cost[i][j] = np.linalg.norm(diff)
                    except:
                        pass
            # Average the squared ERROR
            cost = (0.5) * cost

            print(cost)

            # init assignmetn list
            self.assignment = [-1]*trace_count
            row_ind, col_ind = linear_sum_assignment(cost)
            for i in range(len(row_ind)):
                self.assignment[row_ind[i]] = col_ind[i]

            # identify tracks with no assignment, if any
            un_assigned_tracks = []
            for i in range(len(self.assignment)):
                if self.assignment[i] != -1:
                    # check for cost distance threshold.
                    # If cost is very high then un_assign (delete) the track
                    if cost[i][self.assignment[i]] > self.thresh_dist:
                        self.assignment[i] = -1
                        un_assigned_tracks.append(i)
                    pass

            # if tracks are not detected for long time, remove them
            del_objs = self.is_track_finish(frame)
            if self.hist_objects_record_flag and del_objs:
                for elem in del_objs:
                    self.hist_objects_pool[elem] = del_objs[elem]

            # process un_assigned detects
            un_assigned_detects = []
            for i in range(len(boxes_in_region)):
                if i not in self.assignment:
                    un_assigned_detects.append(i)

            # Start new tracks
            if len(un_assigned_detects) != 0:
                for i in range(len(un_assigned_detects)):
                    obj_id = self.get_available_id()
                    obj = VehicleObject(obj_id)  # create a new vehicle object
                    obj.filter = KalmanFilter()
                    box = boxes_in_region[un_assigned_detects[i]][0]
                    cp_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy()
                    obj.set_first_frame(box=box[:4], frame=box[4], img=cp_img, transformed_box_ct=boxes_in_region[un_assigned_detects[i]][2])
                    obj.set_color(self.get_available_color(obj_id=obj_id))  # set color for displaying
                    obj.update(box=box[:4], frame=box[4], transformed_box_ct=boxes_in_region[un_assigned_detects[i]][2])
                    obj.prediction = boxes_in_region[un_assigned_detects[i]][1]
                    self.objects_pool[obj_id] = obj

            # Update KalmanFilter state, lastResults and tracks trace
            # print("======================")
            for i in range(len(self.assignment)):
                # print(i, ":", self.assignment)
                self.objects_pool[self.objects_pool_elem_list[i]].filter.predict()

                if self.assignment[i] != -1:
                    self.objects_pool[self.objects_pool_elem_list[i]].prediction = self.objects_pool[self.objects_pool_elem_list[i]].filter.correct(
                                                boxes_in_region[self.assignment[i]][1], 1)
                else:
                    self.objects_pool[self.objects_pool_elem_list[i]].prediction = self.objects_pool[self.objects_pool_elem_list[i]].filter.correct(
                                                np.array([[0], [0]]), 0)

                self.objects_pool[self.objects_pool_elem_list[i]].trace.append(self.objects_pool[self.objects_pool_elem_list[i]].prediction)
                self.objects_pool[self.objects_pool_elem_list[i]].filter.lastResult = self.objects_pool[self.objects_pool_elem_list[i]].prediction


    def is_track_finish(self, frame):
        """Check that the object is still in monitoring region"""
        delete_obj_list = []
        for v, elem in enumerate(self.objects_pool):
            # print(elem,frame,self.objects_pool[elem].last_frame)
            if (frame - self.objects_pool[elem].last_frame) > self.frame_space_dist:
                self.objects_pool[elem].update_status = False
                delete_obj_list.append(elem)
        delete_obj_dict = {}
        for elem in delete_obj_list:
            del_obj = self.objects_pool.pop(elem)
            idx = self.objects_pool_elem_list.index(elem)
            self.objects_pool_elem_list.pop(idx)
            self.assignment.pop(idx)
            delete_obj_dict[del_obj.id] = del_obj

        return delete_obj_dict


    def get_boxes_in_region(self, boxes):
        boxes_in_region = []
        for box in boxes:
            box_ct = get_box_center(box, mode='bottom')
            if self.isBoxInPolygonRegion(box_ct) > 0:
                try:
                    input_box_ct = np.array([list(box_ct)]).astype(float).tolist()
                    boxes_in_region.append([box, input_box_ct[0], self.transformer.get_pred_transform(input_box_ct)])
                except:
                    pass
        return boxes_in_region


# # ====== TEST FUNCTIONS =====
def iou_tracker_test(cam_id=0):    # c_tracker='iou'/c_tracker='stp'
    # Default files path setting is in Common.py

    # cam id
    device_id = cam_id

    # Create a perspective transformer
    import Perspective_transform
    Pt_transformer = Perspective_transform.Perspective_transformer(roi_info[device_id])

    # Create an IOU_tracker
    tracker = IOUTracker(transformer=Pt_transformer)

    # Tracker settings
    tracker.display_monitor_region = True
    tracker.hist_objects_record_flag = False
    tracker.hist_objects_img_record_flag = False
    tracker.image_record_flag = False

    img_filepath = data_path[device_id]
    img_savepath = save_path[device_id]
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)

    data_obj = DataGenerator(csv_filename=box_info[device_id],
                          image_fileroot=data_path[device_id])
    data_gen = data_obj.data_gen()
    try:
        while True:
            img, boxes = data_gen.__next__()
            filename = str(boxes[0][4]).zfill(4) + '.jpg'
            for box in boxes:
                cp_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy()
                tracker.update(box, cp_img)
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            obj_pool_img = tracker.draw_objects_pool()
            traj_img = tracker.draw_trajectory(img)
            if traj_img is not None:
                cv2.imshow('img', traj_img)
                if tracker.image_record_flag:
                    cv2.imwrite(os.path.join(img_savepath, filename), traj_img)
            if obj_pool_img is not None:
                cv2.imshow('obj_pool', obj_pool_img)
            cv2.waitKey(1)

    except StopIteration:
        pass

    tracker.save_data(save_path[device_id])

    return




def stp_tracker_test(cam_id=0):
    # test cam
    device_id = cam_id

    # file path
    img_filepath = data_path[device_id]
    tracking_info_filepath = track_info[device_id]
    img_savepath = save_path[device_id]
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)
    pt_savepath = roi_info[device_id]

    time_interval = 1

    trace_record = []

    from data_generator import load_tracking_info
    tracker_record = load_tracking_info(tracking_info_filepath)

    from Perspective_transform import Perspective_transformer
    pt_obj = Perspective_transformer(pt_savepath)

    from cameras_associate import SingleCameraSTP
    STP_Predictor = SingleCameraSTP(
        tracker_record,
        pt_obj,
        time_interval=time_interval,
        var_beta_x=20,
        var_beta_y=3
    )

    # ==== Reset the predictor paras if you need ====
    # STP_Predictor.update_predictor(var_bata_x=20, var_bate_y=3)

    tracker = STPTracker(frame_space_dist=5, transformer=pt_obj, stp_prior=STP_Predictor)
    tracker.match_mode = 'Prob'
    tracker.display_monitor_region = True
    tracker.hist_objects_record_flag = False
    tracker.hist_objects_img_record_flag = False
    tracker.image_record_flag = True

    from Draw_trajectory import draw_objects_pool

    data_obj = DataGenerator(csv_filename=box_info[device_id],
                          image_fileroot=data_path[device_id])
    data_gen = data_obj.data_gen()

    # ===========TEMP============
    img_savepath = r"E:\Project\CV\trajectory\VehicleTracking\results\sct\002\new"
    try:
        while True:
            img, boxes = data_gen.__next__()
            filename = str(boxes[0][4]).zfill(4) + '.jpg'
            tracker.current_frame_img = img
            for box in boxes:
                cp_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy()
                tracker.update(box, cp_img)
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            # obj_pool_img = tracker.draw_objects_pool()
            obj_pool_img = draw_objects_pool(tracker.objects_pool, 100, 100, 3, mode="h", set_range=700)

            traj_img = tracker.draw_trajectory(img)
            if traj_img is not None:
                cv2.imshow('img', traj_img)
                if tracker.image_record_flag:
                    cv2.imwrite(os.path.join(img_savepath, filename), traj_img)
            if obj_pool_img is not None:
                cv2.imshow('obj_pool', obj_pool_img)
                cv2.imwrite(os.path.join(img_savepath, "objects_pool\\"+filename), obj_pool_img)
            cv2.waitKey(1)

    except StopIteration:
        pass
    tracker.save_data(save_path[device_id])
    return




def klf_tracker_test(cam_id=0):
    # Default files path setting is in Common.py

    # cam id
    device_id = cam_id

    # Create a perspective transformer
    import Perspective_transform
    Pt_transformer = Perspective_transform.Perspective_transformer(roi_info[device_id])

    # Create an IOU_tracker
    tracker = KLFTracker(transformer=Pt_transformer)

    # Tracker settings
    tracker.display_monitor_region = True
    tracker.hist_objects_record_flag = True
    tracker.hist_objects_img_record_flag = True
    tracker.image_record_flag = True

    img_filepath = data_path[device_id]
    img_savepath = save_path[device_id]
    if not os.path.exists(img_savepath):
        os.mkdir(img_savepath)

    data_obj = DataGenerator(csv_filename=box_info[device_id],
                          image_fileroot=data_path[device_id])
    data_gen = data_obj.data_gen()
    try:
        while True:
            img, boxes = data_gen.__next__()
            filename = str(boxes[0][4]).zfill(4) + '.jpg'
            tracker.update(boxes, img)
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            obj_pool_img = tracker.draw_objects_pool()
            traj_img = tracker.draw_trajectory(img)
            if traj_img is not None:
                cv2.imshow('img', traj_img)
                if tracker.image_record_flag:
                    cv2.imwrite(os.path.join(img_savepath, filename), traj_img)
            if obj_pool_img is not None:
                cv2.imshow('obj_pool', obj_pool_img)
            cv2.waitKey(1)
    except StopIteration:
        pass
    return

    tracker.save_data(save_path[device_id])


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


if __name__ == "__main__":
    # ===== TEST:iou_tracker test : PASS =====
    # iou_tracker_test(cam_id=1)

    # ===== TEST:STP_tracker test : PASS =====
    stp_tracker_test(cam_id=2)

    # ===== TEST:KLF_tracker test : NOT PASS =====
    # klf_tracker_test(cam_id=0)

    # ===== TEST:TEMP =====
    # TempTest()

    # ===== TEST: KalmanFilter =====
    # rlist = [[1, 1], [3, 2], [4, 4], [6, 4]]
    # obj = KalmanFilter()
    # obj.lastResult = np.array([0, 0])
    # for elem in rlist:
    #     pred = obj.predict()
    #     print("pred:", pred)
    #     print("corr", obj.correct(elem, 1))
    #     obj.lastResult = pred
    #     print("obj.lastResult", obj.lastResult)

    print("=== Mission accomplished! ===")

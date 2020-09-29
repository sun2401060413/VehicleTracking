'''
    透视投影变换ROI区域设置工具

    功能：读入图片或视频，设置ROI，根据ROI与设置宽度与长度,计算透视投影变换矩阵。
    
    ====== TO DO LIST ======
        1. 根据车道数生成默认车道线;
        2. 在ROI区域设置时可调整车道线位置;
        ...
        
    版本：Version1.0 SunZhu 2019-02-12
'''
import os, sys, cv2, time
import numpy as np
from Window import Ui_TabWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog,QTabWidget,QLabel,QWidget 
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QPoint, QRect, QPointF
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QPolygon, QPainterPath, QPolygonF,QBrush
import json
# import glog2 as glob
import glog

class mywindow(QTabWidget,Ui_TabWidget): 

    def __init__(self):
        super(mywindow,self).__init__()
        self.setupUi(self)
        
        self.filename = None
        self.filefolder = None
        self.trans_img = None
        self.th = None
        
        self.imgisOpened    = False
        self.videoisOpened  = False
        self.ROI_points = []
        
        self.img_width  = None
        self.img_height = None
        
        self.transformed_width_for_pred = None
        self.transformed_height_for_pred = None
        
        self.transformed_width_for_disp = None
        self.transformed_height_for_disp = None
        
        self.disp_img_points = []
        self.ratio_roi_points = []
        
        self.offset_x   = 0
        self.offset_y   = 0
        
        self.img_center_x = 0
        self.img_center_y = 0
        
        self.lb_center_x = 0
        self.lb_center_y = 0
        
        self.transform_matrix_for_pred = None
        self.transform_matrix_for_disp = None
        
        MousePressStatus = False
        self.setMouseTracking(False)
        
        self.lb = MyLabel(self.tab)
        self.lb.setGeometry(self.label.geometry())
 
        self.lb.raise_()
        self.lb.init_display_status()
      
    def videoprocessing(self):
        global videoName #在这里设置全局变量以便在线s程中使用

        filepath = get_default_filepath()
        videoName,videoType= QFileDialog.getOpenFileName(self,
                                    "打开视频",
                                    filepath,
                                    #" *.jpg;;*.png;;*.jpeg;;*.bmp")
                                    " *.mp4;;*.avi;;All Files (*)")
        if videoName:
            self.videoisOpened = True
            self.imgisOpened = False
            self.lb.display_status  =   False
            
            self.pushButton_3.setEnabled(True)
            self.pushButton_4.setEnabled(False)
            self.pushButton_5.setEnabled(False)
            
            cap = cv2.VideoCapture(str(videoName))
            self.img_width      = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.img_height     = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.fps            = cap.get(cv2.CAP_PROP_FPS)
            self.frame_count    = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            open_filepath_root,filename = os.path.split(videoName)
            self.filefolder = open_filepath_root
            self.filename = filename
            
            set_default_filepath(open_filepath_root)
            self.th = Thread(self)
            self.th.init_()
            self.th.img_width    =   int(self.img_width/2)
            self.th.img_height   =   int(self.img_height/2)
            
            self.lb_center_x = self.lb.geometry().width()/2
            self.lb_center_y = self.lb.geometry().height()/2
            
            self.img_center_x = int(self.th.img_width/2)
            self.img_center_y = int(self.th.img_height/2)
            
            self.offset_x = abs(self.img_center_x - self.lb_center_x)
            self.offset_y = abs(self.img_center_y - self.lb_center_y)
            
            self.lb.tmp_center_pt = QPoint(self.lb_center_x,self.lb_center_y)
            
            self.th.changePixmap.connect(self.setImage)
            self.th.start()

    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
        
    def imageprocessing(self):
        filepath = get_default_filepath()
        imgName,imgType= QFileDialog.getOpenFileName(self,
                                    "打开图片",
                                    filepath,
                                    #" *.jpg;;*.png;;*.jpeg;;*.bmp")
                                    " *.png;;*.jpg;;*.jpeg;;*.bmp;;All Files (*)")
        if imgName:
            
            self.imgisOpened = True
            self.videoisOpened = False
            self.lb.display_status  =   False
            
            self.pushButton_3.setEnabled(True)
            self.pushButton_4.setEnabled(False)
            self.pushButton_5.setEnabled(False)
            
            open_filepath_root,filename = os.path.split(imgName)
            set_default_filepath(open_filepath_root)
            
            self.filefolder = open_filepath_root
            self.filename = filename

            self.trans_img = cv_imread(imgName)
            img_shape = self.trans_img.shape
            self.img_height, self.img_width = img_shape[0], img_shape[1]

            self.lb_center_x = self.lb.geometry().width()/2
            self.lb_center_y = self.lb.geometry().height()/2
            
            self.img_center_x = int(self.img_width/4)
            self.img_center_y = int(self.img_height/4)
            
            self.offset_x = abs(self.img_center_x - self.lb_center_x)
            self.offset_y = abs(self.img_center_y - self.lb_center_y)
            
            self.lb.tmp_center_pt = QPoint(self.lb_center_x,self.lb_center_y)

            png = QtGui.QPixmap(imgName).scaled(self.img_width/2, self.img_height/2)
            self.label.setPixmap(png)
        
    def roi_setting(self):
        self.lb.display_status  =   True
        self.lb.default_rect    =   True
        self.pushButton_4.setEnabled(True)
        self.pushButton_5.setEnabled(True)
        
        init_roi_rect = self.lb.init_roi_rect(self.img_height/2, self.img_width/2, self.offset_x, self.offset_y)
        
        self.disp_img_points = []
        self.disp_img_points.append(QPoint(int(self.offset_x), int(self.offset_y)))
        self.disp_img_points.append(QPoint(int(self.img_width/2+self.offset_x), int(self.offset_y)))
        self.disp_img_points.append(QPoint(int(self.img_width/2+self.offset_x), int(self.img_height/2+self.offset_y)))
        self.disp_img_points.append(QPoint(int(self.offset_x), int(self.img_height/2+self.offset_y)))
        
        self.lb.img_rect = self.disp_img_points
        
        self.lb.update()
        
    
    def get_transform_matrix(self):
        base_x      =   self.disp_img_points[0].x()
        base_y      =   self.disp_img_points[0].y()
        base_width  =   self.disp_img_points[1].x()-base_x
        base_height =   self.disp_img_points[3].y()-base_y
        
        self.ratio_roi_points = []
        for elem in self.lb.roi_rect:
            # print(elem)
            self.ratio_roi_points.append([(elem.x()-base_x)*self.img_width/base_width,(elem.y()-base_y)*self.img_height/base_height])
            
        self.transformed_width_for_pred = float(self.textedit_1.text())
        self.transformed_height_for_pred = float(self.textedit_2.text())
        
        self.transformed_width_for_disp = float(self.textedit_4.text())
        self.transformed_height_for_disp = float(self.textedit_5.text())
        
        dist_transformed_points = []
        dist_transformed_points.append([0, 0])
        dist_transformed_points.append([self.transformed_width_for_pred, 0])
        dist_transformed_points.append([self.transformed_width_for_pred, self.transformed_height_for_pred])
        dist_transformed_points.append([0, self.transformed_height_for_pred])
        
        self.disp_ratio = 50
        disp_transformed_points = []
        disp_transformed_points.append([0, 0])
        disp_transformed_points.append([self.transformed_width_for_disp, 0])
        disp_transformed_points.append([self.transformed_width_for_disp, self.transformed_height_for_disp])
        disp_transformed_points.append([0, self.transformed_height_for_disp])
        
        self.transform_matrix_for_pred = cv2.getPerspectiveTransform(np.array(self.ratio_roi_points, dtype="float32"), np.array(dist_transformed_points, dtype="float32"))
        self.transform_matrix_for_disp = cv2.getPerspectiveTransform(np.array(self.ratio_roi_points, dtype="float32"), np.array(disp_transformed_points, dtype="float32"))
        print("self.transform_matrix_for_pred:\n", self.transform_matrix_for_pred)
        print("self.transform_matrix_for_disp:\n", self.transform_matrix_for_disp)
    
    def transformation_display(self):
    
        self.get_transform_matrix()
        src = np.array(self.ratio_roi_points)
        pred_vector = cv2.perspectiveTransform(src[None, :, :], self.transform_matrix_for_pred)
        # print("pred_vector",pred_vector)
        
        if self.videoisOpened:
            img = self.th.img
        else:
            img = self.trans_img

        perspective = cv2.warpPerspective(img, self.transform_matrix_for_disp, (int(self.transformed_width_for_disp), int(self.transformed_height_for_disp)), cv2.INTER_LINEAR)
        
        cv2.imshow("img",perspective)
        cv2.waitKey()
        
        return
        
    def transformer_saving(self):
        f_name,ext_name = os.path.splitext(self.filename)
        export_info = {}
        if self.transform_matrix_for_pred is None:
            self.get_transform_matrix()

        export_info["transform_matrix_for_pred"] = self.transform_matrix_for_pred.tolist()
        export_info["transform_matrix_for_disp"] = self.transform_matrix_for_disp.tolist()
        export_info["original_img_width"] = self.img_width
        export_info["original_img_height"] = self.img_height
        

        export_info["transformed_width_for_pred"] = self.transformed_width_for_pred 
        export_info["transformed_height_for_pred"] = self.transformed_height_for_pred
        export_info["transformed_width_for_disp"] = self.transformed_width_for_disp
        export_info["transformed_height_for_disp"] = self.transformed_height_for_disp
        
        export_info["endpoints"] = np.array(self.ratio_roi_points).tolist()
        
        with open(os.path.join(self.filefolder,f_name+"_transformer.json"), "w") as doc:
            json.dump(export_info,doc)
        glog.info("文件保存于:{}！".format(self.filefolder))
        return
        
    def closeEvent(self, event):
        '''
        重写closeEvent方法，实现dialog窗体关闭时执行一些代码
        :param event: close()触发的事件
        :return: None
        '''
        reply = QtWidgets.QMessageBox.question(self,
                                               '本程序',
                                               "是否要退出程序？",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            # self.close()      # 线程无法关闭....
            glog.info('ROI设置程序关闭...')
            sys.exit(app.exec_())
        else:
            event.ignore()
        
        
class MyLabel(QtWidgets.QLabel):
    x0, y0, x1, y1 = 0, 0, 0, 0
    
    def init_display_status(self):
        self.display_status     =   False
        self.default_rect       =   True
        self.roi_rect           =   []
        self.img_rect           =   []
        self.chosen_point       =   None
        self.point_index        =   None
        self.MousePressStatus   =   False
        self.img_width          =   None
        self.img_height         =   None
        
        self.tmp_center_pt      =   None
        
    def init_roi_rect(self,height,width,offset_x,offset_y):
        roi_points = []
        roi_points.append(QPoint(int(width/4+offset_x), int(height/4+offset_y)))
        roi_points.append(QPoint(int(width*3/4+offset_x), int(height/4+offset_y)))
        roi_points.append(QPoint(int(width*3/4+offset_x), int(height*3/4+offset_y)))
        roi_points.append(QPoint(int(width/4+offset_x), int(height*3/4+offset_y)))
        self.roi_rect = roi_points
        return roi_points
        
    def paintEvent(self,event):
        if self.display_status:
            super().paintEvent(event)
            painter = QPainter(self)
            painter_path = QPainterPath()
            
            qt_pen_1 = QPen(Qt.red, 2, Qt.SolidLine)
            qt_pen_2 = QPen(Qt.green, 10, Qt.SolidLine)
            qt_pen_3 = QPen(Qt.red, 2, Qt.DotLine)
            
            painter.setPen(qt_pen_1)
            painter.drawPolygon(QPolygon(self.roi_rect), Qt.WindingFill)
            
            painter_path.addPolygon(QPolygonF(self.roi_rect))
            painter_path.closeSubpath()
            
            # qt_brush = QBrush(Qt.green)
            qt_brush = QBrush(QColor(0, 255, 0, 64))
            
            painter.setBrush(qt_brush)
            painter.drawPath(painter_path)
            
            # painter.drawPoint(self.tmp_center_pt)
            painter.setPen(qt_pen_3)
            painter.drawLine(self.roi_rect[0], self.roi_rect[2])
            painter.drawLine(self.roi_rect[1], self.roi_rect[3])
 
            painter.setPen(qt_pen_2)
            for elem in self.roi_rect:
                painter.drawPoint(elem)
            # for elem in self.img_rect:
                # painter.drawPoint(elem) 
            if self.default_rect:
                self.update()   
        
    def mousePressEvent(self, event):
        if self.display_status:
            self.x = event.x()
            self.y = event.y()
            tmp = get_distance_between_current_pt_and_nearest_pt(self.roi_rect, self.x, self.y)
            if tmp[0] < 10:
                self.chosen_point = tmp[1]
                self.point_index = get_point_index(self.roi_rect, self.chosen_point)
                # print(self.point_index)
                self.MousePressStatus = True
                self.default_rect = False
                # print("chosen_point:",self.chosen_point)
            else:
                self.chosen_point = None
                self.point_index = None
            # print("mousePressEvent",event.x(),event.y())
        
    def mouseMoveEvent(self, event):
        if self.display_status:
            if self.MousePressStatus:
                if self.chosen_point:
                   # print("chosen_index:",self.point_index)
                   self.x = event.x()
                   self.y = event.y()
                   self.roi_rect[self.point_index] = QPoint(event.x(), event.y())
                   self.update()
                   # print("mouseMoveEvent",event.x(),event.y())

    def mouseReleaseEvent(self, event):
        if self.display_status:
            if self.MousePressStatus:
                self.MousePressStatus = False
                self.edit_mode = False
                # print("self.point_index",self.point_index)
                # print("roi_rect",self.roi_rect[self.point_index])
                # print("mouseReleaseEvent",event.x(),event.y())

class Thread(QThread):#采用线程来播放视频
    changePixmap = pyqtSignal(QtGui.QImage)
    
    def init_(self):
        self.img_width = 100
        self.img_height = 100
        self.img = None
    
    def run(self):
        cap = cv2.VideoCapture(videoName)
        self.frame_width    = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height   = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.fps            = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count    = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        while (cap.isOpened()==True):
            ret, frame = cap.read()
            if ret:
                self.img = frame
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)#在这里可以对每帧图像进行处理，
                # p = convertToQtFormat.scaled(self.frame_width, self.frame_height, Qt.KeepAspectRatio)
                # p = convertToQtFormat.scaled(self.frame_width, self.frame_height, Qt.IgnoreAspectRatio)
                p = convertToQtFormat.scaled(self.img_width, self.img_height, Qt.IgnoreAspectRatio)
                self.changePixmap.emit(p)
                time.sleep(1.0/self.fps) #控制视频播放的速度
            else:
                break

                
def get_point_index(points_list,point):
    for i, elem in enumerate(points_list):
        if elem == point:
            return i
    return -1
                
def get_distance_between_current_pt_and_nearest_pt(points,x,y):
    dist = []
    for elem in points:
        dist.append([get_manhattanLength(elem.x(),elem.y(),x,y),elem])  
    def takefirstelem(elem):
        return elem[0]
    dist.sort(key=takefirstelem)
    return dist[0]

def get_manhattanLength(x1,y1,x2,y2):
    return abs(x2-x1)+abs(y2-y1)
                    
def get_default_filepath():
    current_file_path = os.path.abspath(__file__)
    current_root, filename = os.path.split(current_file_path)
    if os.path.exists(os.path.join(current_root,"config.json")):
        with open(os.path.join(current_root,"config.json"),"r") as doc:
            config_data = json.load(doc)
    else:
        config_data = {}
        config_data["default_path"] = os.path.abspath(__file__)
        with open(os.path.join(current_root,"config.json"),"w") as doc:
            json.dump(config_data,doc)
    return config_data["default_path"]
    
def set_default_filepath(filepath=None):
    current_file_path = os.path.abspath(__file__)
    current_root, filename = os.path.split(current_file_path)
    with open(os.path.join(current_root,"config.json"),"r") as doc:
        config_data = json.load(doc)
    config_data["default_path"] = filepath
    with open(os.path.join(current_root,"config.json"),"w") as doc:
        json.dump(config_data,doc)

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    # cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
    return cv_img
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    window = mywindow()
    glog.info('ROI设置程序启动...')
    window.show()
    sys.exit(app.exec_())
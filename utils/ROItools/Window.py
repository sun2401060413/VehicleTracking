from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_TabWidget(object):

    def setupUi(self, TabWidget):
        self.desktop = QtWidgets.QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.height = self.screenRect.height()
        self.width = self.screenRect.width()
        
        TabWidget.setObjectName("TabWidget") 
        TabWidget.resize(self.width/1.5, self.height/1.5)

        self.tab = QtWidgets.QWidget() 
        self.tab.setObjectName("tab") 
        
        mainSplitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        mainSplitter.setOpaqueResize(True)
                  
        frame = QtWidgets.QFrame(mainSplitter)
        mainLayout = QtWidgets.QGridLayout(frame)
        mainLayout.setSpacing(6)
        
        
        self.pushButton_1 = QtWidgets.QPushButton(self.tab)
        self.pushButton_1.setGeometry(QtCore.QRect(10, 10, 100, 30))
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab)
        self.pushButton_2.setGeometry(QtCore.QRect(120, 10, 100, 30))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.tab)
        self.pushButton_3.setGeometry(QtCore.QRect(230, 10, 100, 30))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.setEnabled(False)
        self.pushButton_4 = QtWidgets.QPushButton(self.tab)
        self.pushButton_4.setGeometry(QtCore.QRect(340, 10, 100, 30))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.setEnabled(False)
        self.pushButton_5 = QtWidgets.QPushButton(self.tab)
        self.pushButton_5.setGeometry(QtCore.QRect(450, 10, 100, 30))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.setEnabled(False)
        
        self.textlabel_1 = QtWidgets.QLabel('区域宽度',self.tab)
        self.textlabel_1.setGeometry(QtCore.QRect(560, 10, 80, 30))
        self.textedit_1 = QtWidgets.QLineEdit(self.tab)
        self.textedit_1.setGeometry(QtCore.QRect(640, 10, 50, 30))
        self.textedit_1.setAlignment(QtCore.Qt.AlignRight)
        #self.textedit_1.setPlaceholderText("9")
        self.textedit_1.setText("9.0")
        
        self.textlabel_2 = QtWidgets.QLabel('区域长度',self.tab)
        self.textlabel_2.setGeometry(QtCore.QRect(700, 10, 80, 30))
        self.textedit_2 = QtWidgets.QLineEdit(self.tab)
        self.textedit_2.setGeometry(QtCore.QRect(780, 10, 50, 30))
        self.textedit_2.setAlignment(QtCore.Qt.AlignRight)
        # self.textedit_2.setPlaceholderText("30")
        self.textedit_2.setText("30.0")
        
        self.textlabel_4 = QtWidgets.QLabel('输出宽度',self.tab)
        self.textlabel_4.setGeometry(QtCore.QRect(840, 10, 80, 30))
        self.textedit_4 = QtWidgets.QLineEdit(self.tab)
        self.textedit_4.setGeometry(QtCore.QRect(920, 10, 50, 30))
        self.textedit_4.setAlignment(QtCore.Qt.AlignRight)
        #self.textedit_1.setPlaceholderText("300")
        self.textedit_4.setText("800")
        
        self.textlabel_5 = QtWidgets.QLabel('输出高度',self.tab)
        self.textlabel_5.setGeometry(QtCore.QRect(980, 10, 80, 30))
        self.textedit_5 = QtWidgets.QLineEdit(self.tab)
        self.textedit_5.setGeometry(QtCore.QRect(1060, 10, 50, 30))
        self.textedit_5.setAlignment(QtCore.Qt.AlignRight)
        # self.textedit_2.setPlaceholderText("1000")
        self.textedit_5.setText("800")
        
        
        self.textlabel_3 = QtWidgets.QLabel('车道数量',self.tab)
        self.textlabel_3.setGeometry(QtCore.QRect(1120, 10, 80, 30))
        self.textedit_3 = QtWidgets.QLineEdit(self.tab)
        self.textedit_3.setGeometry(QtCore.QRect(1200, 10, 50, 30))
        self.textedit_3.setAlignment(QtCore.Qt.AlignRight)
        # self.textedit_3.setPlaceholderText("2")
        self.textedit_3.setText("2")
        
        self.label = QtWidgets.QLabel(self.tab)             # 在label上显示图片
        self.label.setAlignment(QtCore.Qt.AlignCenter)      # 在label上显示图片
        self.client_rect = QtCore.QRect( self.geometry().x()+10,
                                    self.geometry().y()+60,
                                    self.geometry().width()-10,
                                    self.geometry().height()-120
                                    )
       
        self.label.setGeometry(self.client_rect)
        self.label.setText("")
        self.label.setObjectName("label")

        
        TabWidget.addTab(self.tab, "")
        self.tab1 = QtWidgets.QWidget() # "第二个子窗口"
        self.tab1.setObjectName("监控区域设置")
        TabWidget.addTab(self.tab1, "")

        self.retranslateUi(TabWidget)
        TabWidget.setCurrentIndex(0)
        self.pushButton_1.clicked.connect(TabWidget.imageprocessing) 
        self.pushButton_2.clicked.connect(TabWidget.videoprocessing)
        self.pushButton_3.clicked.connect(TabWidget.roi_setting)
        self.pushButton_4.clicked.connect(TabWidget.transformation_display)
        self.pushButton_5.clicked.connect(TabWidget.transformer_saving)
        QtCore.QMetaObject.connectSlotsByName(TabWidget)

    def retranslateUi(self, TabWidget):
        _translate = QtCore.QCoreApplication.translate
        TabWidget.setWindowTitle(_translate("TabWidget", "ROI设置工具"))
        self.pushButton_1.setText(_translate("TabWidget", "打开图片"))
        self.pushButton_2.setText(_translate("TabWidget", "打开视频"))
        self.pushButton_3.setText(_translate("TabWidget", "设置ROI"))
        self.pushButton_4.setText(_translate("TabWidget", "效果测试"))
        self.pushButton_5.setText(_translate("TabWidget", "保存ROI"))
        TabWidget.setTabText(TabWidget.indexOf(self.tab), _translate("TabWidget", "监控区域设置"))
        TabWidget.setTabText(TabWidget.indexOf(self.tab1), _translate("TabWidget", "参数设置"))
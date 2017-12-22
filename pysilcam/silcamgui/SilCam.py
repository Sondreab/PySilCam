# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SilCam.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SilCam(object):
    def setupUi(self, SilCam):
        SilCam.setObjectName("SilCam")
        SilCam.resize(814, 703)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SilCam.sizePolicy().hasHeightForWidth())
        SilCam.setSizePolicy(sizePolicy)
        SilCam.setWindowTitle("SilCam")
        SilCam.setStatusTip("")
        self.centralwidget = QtWidgets.QWidget(SilCam)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.statusBar = QtWidgets.QLabel(self.centralwidget)
        self.statusBar.setObjectName("statusBar")
        self.gridLayout.addWidget(self.statusBar, 0, 1, 1, 1, QtCore.Qt.AlignTop)
        self.fig_widget = QtWidgets.QWidget(self.centralwidget)
        self.fig_widget.setMinimumSize(QtCore.QSize(790, 400))
        self.fig_widget.setObjectName("fig_widget")
        self.gridLayout.addWidget(self.fig_widget, 1, 0, 1, 2)
        SilCam.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SilCam)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 814, 27))
        self.menubar.setObjectName("menubar")
        self.menuProcessing = QtWidgets.QMenu(self.menubar)
        self.menuProcessing.setEnabled(True)
        self.menuProcessing.setObjectName("menuProcessing")
        SilCam.setMenuBar(self.menubar)
        self.actionOpen = QtWidgets.QAction(SilCam)
        self.actionOpen.setObjectName("actionOpen")
        self.actionExit = QtWidgets.QAction(SilCam)
        self.actionExit.setObjectName("actionExit")
        self.actionRaw = QtWidgets.QAction(SilCam)
        self.actionRaw.setObjectName("actionRaw")
        self.actionConfig = QtWidgets.QAction(SilCam)
        self.actionConfig.setObjectName("actionConfig")
        self.actionController = QtWidgets.QAction(SilCam)
        self.actionController.setObjectName("actionController")
        self.actionSave_Figure = QtWidgets.QAction(SilCam)
        self.actionSave_Figure.setObjectName("actionSave_Figure")
        self.actionLoadProcessed = QtWidgets.QAction(SilCam)
        self.actionLoadProcessed.setObjectName("actionLoadProcessed")
        self.actionVD_Time_series = QtWidgets.QAction(SilCam)
        self.actionVD_Time_series.setObjectName("actionVD_Time_series")
        self.actionExport_Time_series = QtWidgets.QAction(SilCam)
        self.actionExport_Time_series.setObjectName("actionExport_Time_series")
        self.actionSilc_viewer = QtWidgets.QAction(SilCam)
        self.actionSilc_viewer.setObjectName("actionSilc_viewer")
        self.actionServer = QtWidgets.QAction(SilCam)
        self.actionServer.setObjectName("actionServer")
        self.actionConvert_silc_to_bmp = QtWidgets.QAction(SilCam)
        self.actionConvert_silc_to_bmp.setObjectName("actionConvert_silc_to_bmp")
        self.actionExport_summary_data = QtWidgets.QAction(SilCam)
        self.actionExport_summary_data.setObjectName("actionExport_summary_data")
        self.menuProcessing.addAction(self.actionConvert_silc_to_bmp)
        self.menuProcessing.addAction(self.actionExport_summary_data)
        self.menuProcessing.addAction(self.actionServer)
        self.menubar.addAction(self.menuProcessing.menuAction())

        self.retranslateUi(SilCam)
        QtCore.QMetaObject.connectSlotsByName(SilCam)

    def retranslateUi(self, SilCam):
        _translate = QtCore.QCoreApplication.translate
        self.statusBar.setText(_translate("SilCam", "STATUS"))
        self.menuProcessing.setTitle(_translate("SilCam", "Processing"))
        self.actionOpen.setText(_translate("SilCam", "Open"))
        self.actionExit.setText(_translate("SilCam", "Exit"))
        self.actionRaw.setText(_translate("SilCam", "Raw"))
        self.actionConfig.setText(_translate("SilCam", "Config"))
        self.actionController.setText(_translate("SilCam", "Controller"))
        self.actionSave_Figure.setText(_translate("SilCam", "Save Figure"))
        self.actionLoadProcessed.setText(_translate("SilCam", "Load"))
        self.actionVD_Time_series.setText(_translate("SilCam", "VD Time-series"))
        self.actionExport_Time_series.setText(_translate("SilCam", "Export VD Time-series"))
        self.actionSilc_viewer.setText(_translate("SilCam", "silc viewer"))
        self.actionServer.setText(_translate("SilCam", "Realtime server"))
        self.actionConvert_silc_to_bmp.setText(_translate("SilCam", "Convert silc to bmp"))
        self.actionExport_summary_data.setText(_translate("SilCam", "Export summary data"))


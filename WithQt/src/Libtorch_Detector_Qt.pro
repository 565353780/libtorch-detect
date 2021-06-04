QT       += core gui

win32{
DESTDIR = ../bin_win
}
unix{
DESTDIR = ../bin_linux
}

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++14

QMAKE_CXXFLAGS += -D_GLIBCXX_USE_CXX11_ABI=0

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    main.cpp \
    MainWindow.cpp \
    LibTorch_Detector/LibTorch_Detector.cpp \
    LapNet_Detector.cpp \
    DataProcesser.cpp

HEADERS += \
    MainWindow.h \
    LibTorch_Detector/LibTorch_Detector.h \
    LapNet_Detector.h \
    DataProcesser.h

# OpenCV
unix{
INCLUDEPATH += \
    /home/chli/OpenCV/opencv-3.4.0/build/installed/include \
    /home/chli/OpenCV/opencv-3.4.0/build/installed/include/opencv \
    /home/chli/OpenCV/opencv-3.4.0/build/installed/include/opencv2

LIBS += \
    -L/home/chli/OpenCV/opencv-3.4.0/build/installed/lib \
    -lopencv_highgui \
    -lopencv_highgui \
    -lopencv_imgcodecs \
    -lopencv_imgproc \
    -lopencv_core \
    -lopencv_videoio
}
win32{
INCLUDEPATH += \
    C:/Program\ Files/OpenCV/opencv/build/include \
    C:/Program\ Files/OpenCV/opencv/build/include/opencv \
    C:/Program\ Files/OpenCV/opencv/build/include/opencv2

LIBS += \
    -LC:/Program\ Files/OpenCV/opencv/build/x64/vc15/lib \
    -lopencv_world340 \
    -LC:/Program\ Files/OpenCV/opencv/build/bin \
    -lopencv_ffmpeg340
}

# LibTorch
unix{
INCLUDEPATH += \
    /home/chli/anaconda3/envs/py39/lib/python3.9/site-packages/torch/include \
    /home/chli/anaconda3/envs/py39/lib/python3.9/site-packages/torch/include/torch/csrc/api/include

LIBS += \
    -L/home/chli/anaconda3/envs/py39/lib/python3.9/site-packages/torch/lib \
    -ltorch \
    -ltorch_cpu \
    -ltorch_cuda \
    -lc10 \
    -lc10_cuda \
    -lcaffe2_nvrtc \
    -lcaffe2_module_test_dynamic \
    -lcaffe2_detectron_ops_gpu
}
win32{
INCLUDEPATH += \
    $$PWD/LibTorch_Detector/LibTorch/libtorch-win-shared-with-deps-1.8.1+cu111/libtorch/include \
    $$PWD/LibTorch_Detector/LibTorch/libtorch-win-shared-with-deps-1.8.1+cu111/libtorch/include/torch/csrc/api/include

LIBS += \
    -L$$PWD/LibTorch_Detector/LibTorch/libtorch-win-shared-with-deps-1.8.1+cu111/libtorch/lib \
    -ltorch \
    -ltorch_cpu \
    -ltorch_cuda \
    -lc10 \
    -lc10_cuda \
    -lcaffe2_nvrtc \
    -lcaffe2_module_test_dynamic \
    -lcaffe2_detectron_ops_gpu
}

FORMS += \
    MainWindow.ui

TRANSLATIONS += \
    Libtorch_Detector_Qt_zh_CN.ts

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

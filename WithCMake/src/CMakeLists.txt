cmake_minimum_required(VERSION 3.5)

project(LibTorch_Detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Torch_DIR "/home/abaci/LibTorch/libtorch-shared-with-deps-1.7.0+cu110/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

set(OpenCV_DIR "/home/abaci/OpenCV/OpenCV-3.4.0/opencv-3.4.0/build/installed/share/OpenCV")
find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
SET(OpenCV_LIBRARIES ${OpenCV_LIBS})

FILE(GLOB SC_FILES "*.cpp" "*.h")
add_executable(${PROJECT_NAME} ${SC_FILES})

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

cmake_minimum_required(VERSION 3.0.0)
project(task3)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(OpenCV REQUIRED)

aux_source_directory(./src ALL_SRCS)
include_directories(include
${OpenCV_INCLUDE_DIRS})

set(SRC ./cc)
add_executable(task3 ./main.cpp ${ALL_SRCS})
#
target_link_libraries(task3 ${OpenCV_LIBS})


# 寻找Ceres库
set(Ceres_DIR "/usr/local/include/ceres")
find_package(Ceres REQUIRED)

# 包含Ceres头文件目录
include_directories(${CERES_INCLUDE_DIRS})

# 添加你的目标

# 链接Ceres库到你的目标
target_link_libraries(task3 ${CERES_LIBRARIES})

# 设置环境变量
set(ENV{LD_LIBRARY_PATH} "/usr/local/ceres-solver/lib:$ENV{LD_LIBRARY_PATH}")

# 确保这一行不在执行任何命令的地方

cmake_minimum_required(VERSION 3.5)

project(bg_sub_sample)

# Find dependencies
find_package(vpi REQUIRED)
find_package(OpenCV REQUIRED)

# Create executable
add_executable(bg_sub_sample bg_sub_sample.cpp)

# Link VPI and OpenCV
target_link_libraries(bg_sub_sample
    vpi::vpi
    ${OpenCV_LIBS}
)

# OpenCV < 3 uses a different library for image i/o
if(OpenCV_VERSION VERSION_LESS 3)
    target_link_libraries(bg_sub_sample opencv_highgui)
else()
    target_link_libraries(bg_sub_sample opencv_imgcodecs)
endif()

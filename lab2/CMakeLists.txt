cmake_minimum_required(VERSION 3.6)
project(lab2)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

find_package(TBB REQUIRED)
if(TBB_FOUND)
    include_directories(${TBB_INCLUDE_DIRS})
endif()

set(SOURCE_FILES main.cpp)
add_executable(lab2 ${SOURCE_FILES})

target_link_libraries(lab2 ${TBB_LIBRARIES})
target_compile_definitions(lab2 PRIVATE ${TBB_DEFINITIONS})
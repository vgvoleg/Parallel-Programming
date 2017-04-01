cmake_minimum_required(VERSION 3.6)
project(lab3)

set(CMAKE_CXX_STANDARD 11)

# OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}    ${OpenMP_C_FLAGS}")
endif()

# MPI
find_package( MPI REQUIRED QUIET)
include_directories( ${MPI_INCLUDE_PATH} )

set(SOURCE_FILES main.cpp)
add_executable(lab3 ${SOURCE_FILES})

target_link_libraries( ${PROJECT_NAME} ${MPI_LIBRARIES} )
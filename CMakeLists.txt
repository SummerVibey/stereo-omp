# project
cmake_minimum_required (VERSION 2.8)
project (StereoMatcher)

set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-msse3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

# # directories
# set (LIBStereoMatcher_SRC_DIR src)

# Eigen
include_directories( "/usr/local/include/eigen3" )

# OpenCV
find_package(OpenCV 3.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Boost
find_package(Boost 1.55.0 REQUIRED COMPONENTS system filesystem timer)
include_directories(${Boost_INCLUDE_DIRS})

set( THIRD_PARTY_LIBS
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)

# include directory
include_directories("${PROJECT_SOURCE_DIR}/src")

# use sse3 instruction set
# SET(CMAKE_CXX_FLAGS "-msse3")

# # sources
# FILE(GLOB LIBStereoMatcher_SRC_FILES "src/*.cpp")


# build demo program
add_executable(StereoMatcher 
        test/elas_test.cpp
        src/SobelDescriptor.cpp
        src/StereoMatcher.cpp
        src/Filter.cpp
        src/Matrix.cpp
        src/Delaunay.cpp
        )


target_link_libraries(StereoMatcher 
              PUBLIC
              ${THIRD_PARTY_LIBS})


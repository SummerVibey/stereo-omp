# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.9

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dst03/Desktop/stereo_omp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dst03/Desktop/stereo_omp/build

# Include any dependencies generated for this target.
include CMakeFiles/StereoMatcher.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/StereoMatcher.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/StereoMatcher.dir/flags.make

CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o: CMakeFiles/StereoMatcher.dir/flags.make
CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o: ../test/elas_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dst03/Desktop/stereo_omp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o -c /home/dst03/Desktop/stereo_omp/test/elas_test.cpp

CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dst03/Desktop/stereo_omp/test/elas_test.cpp > CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.i

CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dst03/Desktop/stereo_omp/test/elas_test.cpp -o CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.s

CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o.requires:

.PHONY : CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o.requires

CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o.provides: CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o.requires
	$(MAKE) -f CMakeFiles/StereoMatcher.dir/build.make CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o.provides.build
.PHONY : CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o.provides

CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o.provides.build: CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o


CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o: CMakeFiles/StereoMatcher.dir/flags.make
CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o: ../src/SobelDescriptor.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dst03/Desktop/stereo_omp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o -c /home/dst03/Desktop/stereo_omp/src/SobelDescriptor.cpp

CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dst03/Desktop/stereo_omp/src/SobelDescriptor.cpp > CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.i

CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dst03/Desktop/stereo_omp/src/SobelDescriptor.cpp -o CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.s

CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o.requires:

.PHONY : CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o.requires

CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o.provides: CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o.requires
	$(MAKE) -f CMakeFiles/StereoMatcher.dir/build.make CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o.provides.build
.PHONY : CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o.provides

CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o.provides.build: CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o


CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o: CMakeFiles/StereoMatcher.dir/flags.make
CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o: ../src/StereoMatcher.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dst03/Desktop/stereo_omp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o -c /home/dst03/Desktop/stereo_omp/src/StereoMatcher.cpp

CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dst03/Desktop/stereo_omp/src/StereoMatcher.cpp > CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.i

CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dst03/Desktop/stereo_omp/src/StereoMatcher.cpp -o CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.s

CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o.requires:

.PHONY : CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o.requires

CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o.provides: CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o.requires
	$(MAKE) -f CMakeFiles/StereoMatcher.dir/build.make CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o.provides.build
.PHONY : CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o.provides

CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o.provides.build: CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o


CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o: CMakeFiles/StereoMatcher.dir/flags.make
CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o: ../src/Filter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dst03/Desktop/stereo_omp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o -c /home/dst03/Desktop/stereo_omp/src/Filter.cpp

CMakeFiles/StereoMatcher.dir/src/Filter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/StereoMatcher.dir/src/Filter.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dst03/Desktop/stereo_omp/src/Filter.cpp > CMakeFiles/StereoMatcher.dir/src/Filter.cpp.i

CMakeFiles/StereoMatcher.dir/src/Filter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/StereoMatcher.dir/src/Filter.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dst03/Desktop/stereo_omp/src/Filter.cpp -o CMakeFiles/StereoMatcher.dir/src/Filter.cpp.s

CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o.requires:

.PHONY : CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o.requires

CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o.provides: CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o.requires
	$(MAKE) -f CMakeFiles/StereoMatcher.dir/build.make CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o.provides.build
.PHONY : CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o.provides

CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o.provides.build: CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o


CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o: CMakeFiles/StereoMatcher.dir/flags.make
CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o: ../src/Matrix.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dst03/Desktop/stereo_omp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o -c /home/dst03/Desktop/stereo_omp/src/Matrix.cpp

CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dst03/Desktop/stereo_omp/src/Matrix.cpp > CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.i

CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dst03/Desktop/stereo_omp/src/Matrix.cpp -o CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.s

CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o.requires:

.PHONY : CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o.requires

CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o.provides: CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o.requires
	$(MAKE) -f CMakeFiles/StereoMatcher.dir/build.make CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o.provides.build
.PHONY : CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o.provides

CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o.provides.build: CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o


CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o: CMakeFiles/StereoMatcher.dir/flags.make
CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o: ../src/Delaunay.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dst03/Desktop/stereo_omp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o -c /home/dst03/Desktop/stereo_omp/src/Delaunay.cpp

CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dst03/Desktop/stereo_omp/src/Delaunay.cpp > CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.i

CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dst03/Desktop/stereo_omp/src/Delaunay.cpp -o CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.s

CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o.requires:

.PHONY : CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o.requires

CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o.provides: CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o.requires
	$(MAKE) -f CMakeFiles/StereoMatcher.dir/build.make CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o.provides.build
.PHONY : CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o.provides

CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o.provides.build: CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o


# Object files for target StereoMatcher
StereoMatcher_OBJECTS = \
"CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o" \
"CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o" \
"CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o" \
"CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o" \
"CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o" \
"CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o"

# External object files for target StereoMatcher
StereoMatcher_EXTERNAL_OBJECTS =

StereoMatcher: CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o
StereoMatcher: CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o
StereoMatcher: CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o
StereoMatcher: CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o
StereoMatcher: CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o
StereoMatcher: CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o
StereoMatcher: CMakeFiles/StereoMatcher.dir/build.make
StereoMatcher: /usr/local/lib/libopencv_dnn.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_ml.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_objdetect.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_shape.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_stitching.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_superres.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_videostab.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_viz.so.3.4.0
StereoMatcher: /usr/lib/x86_64-linux-gnu/libboost_system.so
StereoMatcher: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
StereoMatcher: /usr/lib/x86_64-linux-gnu/libboost_timer.so
StereoMatcher: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
StereoMatcher: /usr/local/lib/libopencv_calib3d.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_features2d.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_flann.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_highgui.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_photo.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_video.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_videoio.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_imgcodecs.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_imgproc.so.3.4.0
StereoMatcher: /usr/local/lib/libopencv_core.so.3.4.0
StereoMatcher: CMakeFiles/StereoMatcher.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dst03/Desktop/stereo_omp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable StereoMatcher"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/StereoMatcher.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/StereoMatcher.dir/build: StereoMatcher

.PHONY : CMakeFiles/StereoMatcher.dir/build

CMakeFiles/StereoMatcher.dir/requires: CMakeFiles/StereoMatcher.dir/test/elas_test.cpp.o.requires
CMakeFiles/StereoMatcher.dir/requires: CMakeFiles/StereoMatcher.dir/src/SobelDescriptor.cpp.o.requires
CMakeFiles/StereoMatcher.dir/requires: CMakeFiles/StereoMatcher.dir/src/StereoMatcher.cpp.o.requires
CMakeFiles/StereoMatcher.dir/requires: CMakeFiles/StereoMatcher.dir/src/Filter.cpp.o.requires
CMakeFiles/StereoMatcher.dir/requires: CMakeFiles/StereoMatcher.dir/src/Matrix.cpp.o.requires
CMakeFiles/StereoMatcher.dir/requires: CMakeFiles/StereoMatcher.dir/src/Delaunay.cpp.o.requires

.PHONY : CMakeFiles/StereoMatcher.dir/requires

CMakeFiles/StereoMatcher.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/StereoMatcher.dir/cmake_clean.cmake
.PHONY : CMakeFiles/StereoMatcher.dir/clean

CMakeFiles/StereoMatcher.dir/depend:
	cd /home/dst03/Desktop/stereo_omp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dst03/Desktop/stereo_omp /home/dst03/Desktop/stereo_omp /home/dst03/Desktop/stereo_omp/build /home/dst03/Desktop/stereo_omp/build /home/dst03/Desktop/stereo_omp/build/CMakeFiles/StereoMatcher.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/StereoMatcher.dir/depend


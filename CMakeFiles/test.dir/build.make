# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wiman/CppSaving/Project4

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wiman/CppSaving/Project4

# Include any dependencies generated for this target.
include CMakeFiles/test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test.dir/flags.make

CMakeFiles/test.dir/test.c.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/test.c.o: test.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wiman/CppSaving/Project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/test.dir/test.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/test.dir/test.c.o   -c /home/wiman/CppSaving/Project4/test.c

CMakeFiles/test.dir/test.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/test.dir/test.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/wiman/CppSaving/Project4/test.c > CMakeFiles/test.dir/test.c.i

CMakeFiles/test.dir/test.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/test.dir/test.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/wiman/CppSaving/Project4/test.c -o CMakeFiles/test.dir/test.c.s

CMakeFiles/test.dir/innerTools.c.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/innerTools.c.o: innerTools.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wiman/CppSaving/Project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/test.dir/innerTools.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/test.dir/innerTools.c.o   -c /home/wiman/CppSaving/Project4/innerTools.c

CMakeFiles/test.dir/innerTools.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/test.dir/innerTools.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/wiman/CppSaving/Project4/innerTools.c > CMakeFiles/test.dir/innerTools.c.i

CMakeFiles/test.dir/innerTools.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/test.dir/innerTools.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/wiman/CppSaving/Project4/innerTools.c -o CMakeFiles/test.dir/innerTools.c.s

CMakeFiles/test.dir/Matrix.c.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/Matrix.c.o: Matrix.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wiman/CppSaving/Project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/test.dir/Matrix.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/test.dir/Matrix.c.o   -c /home/wiman/CppSaving/Project4/Matrix.c

CMakeFiles/test.dir/Matrix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/test.dir/Matrix.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/wiman/CppSaving/Project4/Matrix.c > CMakeFiles/test.dir/Matrix.c.i

CMakeFiles/test.dir/Matrix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/test.dir/Matrix.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/wiman/CppSaving/Project4/Matrix.c -o CMakeFiles/test.dir/Matrix.c.s

CMakeFiles/test.dir/calculate.c.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/calculate.c.o: calculate.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wiman/CppSaving/Project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building C object CMakeFiles/test.dir/calculate.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/test.dir/calculate.c.o   -c /home/wiman/CppSaving/Project4/calculate.c

CMakeFiles/test.dir/calculate.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/test.dir/calculate.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/wiman/CppSaving/Project4/calculate.c > CMakeFiles/test.dir/calculate.c.i

CMakeFiles/test.dir/calculate.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/test.dir/calculate.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/wiman/CppSaving/Project4/calculate.c -o CMakeFiles/test.dir/calculate.c.s

CMakeFiles/test.dir/proMul.c.o: CMakeFiles/test.dir/flags.make
CMakeFiles/test.dir/proMul.c.o: proMul.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wiman/CppSaving/Project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building C object CMakeFiles/test.dir/proMul.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/test.dir/proMul.c.o   -c /home/wiman/CppSaving/Project4/proMul.c

CMakeFiles/test.dir/proMul.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/test.dir/proMul.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/wiman/CppSaving/Project4/proMul.c > CMakeFiles/test.dir/proMul.c.i

CMakeFiles/test.dir/proMul.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/test.dir/proMul.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/wiman/CppSaving/Project4/proMul.c -o CMakeFiles/test.dir/proMul.c.s

# Object files for target test
test_OBJECTS = \
"CMakeFiles/test.dir/test.c.o" \
"CMakeFiles/test.dir/innerTools.c.o" \
"CMakeFiles/test.dir/Matrix.c.o" \
"CMakeFiles/test.dir/calculate.c.o" \
"CMakeFiles/test.dir/proMul.c.o"

# External object files for target test
test_EXTERNAL_OBJECTS =

test: CMakeFiles/test.dir/test.c.o
test: CMakeFiles/test.dir/innerTools.c.o
test: CMakeFiles/test.dir/Matrix.c.o
test: CMakeFiles/test.dir/calculate.c.o
test: CMakeFiles/test.dir/proMul.c.o
test: CMakeFiles/test.dir/build.make
test: /usr/lib/gcc/x86_64-linux-gnu/7/libgomp.so
test: /usr/lib/x86_64-linux-gnu/libpthread.so
test: CMakeFiles/test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wiman/CppSaving/Project4/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking C executable test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test.dir/build: test

.PHONY : CMakeFiles/test.dir/build

CMakeFiles/test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test.dir/clean

CMakeFiles/test.dir/depend:
	cd /home/wiman/CppSaving/Project4 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wiman/CppSaving/Project4 /home/wiman/CppSaving/Project4 /home/wiman/CppSaving/Project4 /home/wiman/CppSaving/Project4 /home/wiman/CppSaving/Project4/CMakeFiles/test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test.dir/depend


# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

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
CMAKE_COMMAND = /opt/local/bin/cmake

# The command to remove a file.
RM = /opt/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui

# Utility rule file for libigl-imgui-download.

# Include the progress variables for this target.
include CMakeFiles/libigl-imgui-download.dir/progress.make

CMakeFiles/libigl-imgui-download: CMakeFiles/libigl-imgui-download-complete


CMakeFiles/libigl-imgui-download-complete: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-install
CMakeFiles/libigl-imgui-download-complete: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-mkdir
CMakeFiles/libigl-imgui-download-complete: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-download
CMakeFiles/libigl-imgui-download-complete: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-update
CMakeFiles/libigl-imgui-download-complete: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-patch
CMakeFiles/libigl-imgui-download-complete: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-configure
CMakeFiles/libigl-imgui-download-complete: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-build
CMakeFiles/libigl-imgui-download-complete: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-install
CMakeFiles/libigl-imgui-download-complete: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-test
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'libigl-imgui-download'"
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles/libigl-imgui-download-complete
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-done

libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-install: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No install step for 'libigl-imgui-download'"
	cd /Users/Roll/desktop/GeodesyPlus/build/libigl-imgui-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/GeodesyPlus/build/libigl-imgui-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-install

libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'libigl-imgui-download'"
	/opt/local/bin/cmake -E make_directory /Users/Roll/desktop/GeodesyPlus/lib/libigl/cmake/../external/libigl-imgui
	/opt/local/bin/cmake -E make_directory /Users/Roll/desktop/GeodesyPlus/build/libigl-imgui-build
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/tmp
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/src/libigl-imgui-download-stamp
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/src
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-mkdir

libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-download: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-gitinfo.txt
libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-download: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'libigl-imgui-download'"
	cd /Users/Roll/desktop/GeodesyPlus/lib/libigl/external && /opt/local/bin/cmake -P /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/tmp/libigl-imgui-download-gitclone.cmake
	cd /Users/Roll/desktop/GeodesyPlus/lib/libigl/external && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-download

libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-update: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing update step for 'libigl-imgui-download'"
	cd /Users/Roll/desktop/GeodesyPlus/lib/libigl/external/libigl-imgui && /opt/local/bin/cmake -P /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/tmp/libigl-imgui-download-gitupdate.cmake

libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-patch: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No patch step for 'libigl-imgui-download'"
	/opt/local/bin/cmake -E echo_append
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-patch

libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-configure: libigl-imgui-download-prefix/tmp/libigl-imgui-download-cfgcmd.txt
libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-configure: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-update
libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-configure: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No configure step for 'libigl-imgui-download'"
	cd /Users/Roll/desktop/GeodesyPlus/build/libigl-imgui-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/GeodesyPlus/build/libigl-imgui-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-configure

libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-build: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No build step for 'libigl-imgui-download'"
	cd /Users/Roll/desktop/GeodesyPlus/build/libigl-imgui-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/GeodesyPlus/build/libigl-imgui-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-build

libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-test: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "No test step for 'libigl-imgui-download'"
	cd /Users/Roll/desktop/GeodesyPlus/build/libigl-imgui-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/GeodesyPlus/build/libigl-imgui-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-test

libigl-imgui-download: CMakeFiles/libigl-imgui-download
libigl-imgui-download: CMakeFiles/libigl-imgui-download-complete
libigl-imgui-download: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-install
libigl-imgui-download: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-mkdir
libigl-imgui-download: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-download
libigl-imgui-download: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-update
libigl-imgui-download: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-patch
libigl-imgui-download: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-configure
libigl-imgui-download: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-build
libigl-imgui-download: libigl-imgui-download-prefix/src/libigl-imgui-download-stamp/libigl-imgui-download-test
libigl-imgui-download: CMakeFiles/libigl-imgui-download.dir/build.make

.PHONY : libigl-imgui-download

# Rule to build all files generated by this target.
CMakeFiles/libigl-imgui-download.dir/build: libigl-imgui-download

.PHONY : CMakeFiles/libigl-imgui-download.dir/build

CMakeFiles/libigl-imgui-download.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libigl-imgui-download.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libigl-imgui-download.dir/clean

CMakeFiles/libigl-imgui-download.dir/depend:
	cd /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui /Users/Roll/Desktop/GeodesyPlus/lib/libigl/external/.cache/libigl-imgui/CMakeFiles/libigl-imgui-download.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libigl-imgui-download.dir/depend

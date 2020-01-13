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
CMAKE_SOURCE_DIR = /Users/Roll/Desktop/libigl/external/.cache/embree

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Roll/Desktop/libigl/external/.cache/embree

# Utility rule file for embree-download.

# Include the progress variables for this target.
include CMakeFiles/embree-download.dir/progress.make

CMakeFiles/embree-download: CMakeFiles/embree-download-complete


CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-install
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-mkdir
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-download
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-update
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-patch
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-configure
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-build
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-install
CMakeFiles/embree-download-complete: embree-download-prefix/src/embree-download-stamp/embree-download-test
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'embree-download'"
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles/embree-download-complete
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-done

embree-download-prefix/src/embree-download-stamp/embree-download-install: embree-download-prefix/src/embree-download-stamp/embree-download-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No install step for 'embree-download'"
	cd /Users/Roll/desktop/libigl/build/embree-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/embree-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-install

embree-download-prefix/src/embree-download-stamp/embree-download-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'embree-download'"
	/opt/local/bin/cmake -E make_directory /Users/Roll/desktop/libigl/cmake/../external/embree
	/opt/local/bin/cmake -E make_directory /Users/Roll/desktop/libigl/build/embree-build
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/tmp
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/src
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-mkdir

embree-download-prefix/src/embree-download-stamp/embree-download-download: embree-download-prefix/src/embree-download-stamp/embree-download-gitinfo.txt
embree-download-prefix/src/embree-download-stamp/embree-download-download: embree-download-prefix/src/embree-download-stamp/embree-download-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'embree-download'"
	cd /Users/Roll/desktop/libigl/external && /opt/local/bin/cmake -P /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/tmp/embree-download-gitclone.cmake
	cd /Users/Roll/desktop/libigl/external && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-download

embree-download-prefix/src/embree-download-stamp/embree-download-update: embree-download-prefix/src/embree-download-stamp/embree-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing update step for 'embree-download'"
	cd /Users/Roll/desktop/libigl/external/embree && /opt/local/bin/cmake -P /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/tmp/embree-download-gitupdate.cmake

embree-download-prefix/src/embree-download-stamp/embree-download-patch: embree-download-prefix/src/embree-download-stamp/embree-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No patch step for 'embree-download'"
	/opt/local/bin/cmake -E echo_append
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-patch

embree-download-prefix/src/embree-download-stamp/embree-download-configure: embree-download-prefix/tmp/embree-download-cfgcmd.txt
embree-download-prefix/src/embree-download-stamp/embree-download-configure: embree-download-prefix/src/embree-download-stamp/embree-download-update
embree-download-prefix/src/embree-download-stamp/embree-download-configure: embree-download-prefix/src/embree-download-stamp/embree-download-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No configure step for 'embree-download'"
	cd /Users/Roll/desktop/libigl/build/embree-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/embree-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-configure

embree-download-prefix/src/embree-download-stamp/embree-download-build: embree-download-prefix/src/embree-download-stamp/embree-download-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No build step for 'embree-download'"
	cd /Users/Roll/desktop/libigl/build/embree-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/embree-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-build

embree-download-prefix/src/embree-download-stamp/embree-download-test: embree-download-prefix/src/embree-download-stamp/embree-download-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "No test step for 'embree-download'"
	cd /Users/Roll/desktop/libigl/build/embree-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/embree-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/embree/embree-download-prefix/src/embree-download-stamp/embree-download-test

embree-download: CMakeFiles/embree-download
embree-download: CMakeFiles/embree-download-complete
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-install
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-mkdir
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-download
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-update
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-patch
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-configure
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-build
embree-download: embree-download-prefix/src/embree-download-stamp/embree-download-test
embree-download: CMakeFiles/embree-download.dir/build.make

.PHONY : embree-download

# Rule to build all files generated by this target.
CMakeFiles/embree-download.dir/build: embree-download

.PHONY : CMakeFiles/embree-download.dir/build

CMakeFiles/embree-download.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/embree-download.dir/cmake_clean.cmake
.PHONY : CMakeFiles/embree-download.dir/clean

CMakeFiles/embree-download.dir/depend:
	cd /Users/Roll/Desktop/libigl/external/.cache/embree && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/Desktop/libigl/external/.cache/embree /Users/Roll/Desktop/libigl/external/.cache/embree /Users/Roll/Desktop/libigl/external/.cache/embree /Users/Roll/Desktop/libigl/external/.cache/embree /Users/Roll/Desktop/libigl/external/.cache/embree/CMakeFiles/embree-download.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/embree-download.dir/depend


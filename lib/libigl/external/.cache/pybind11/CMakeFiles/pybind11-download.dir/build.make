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
CMAKE_SOURCE_DIR = /Users/Roll/Desktop/libigl/external/.cache/pybind11

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Roll/Desktop/libigl/external/.cache/pybind11

# Utility rule file for pybind11-download.

# Include the progress variables for this target.
include CMakeFiles/pybind11-download.dir/progress.make

CMakeFiles/pybind11-download: CMakeFiles/pybind11-download-complete


CMakeFiles/pybind11-download-complete: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-install
CMakeFiles/pybind11-download-complete: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-mkdir
CMakeFiles/pybind11-download-complete: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-download
CMakeFiles/pybind11-download-complete: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-update
CMakeFiles/pybind11-download-complete: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-patch
CMakeFiles/pybind11-download-complete: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-configure
CMakeFiles/pybind11-download-complete: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-build
CMakeFiles/pybind11-download-complete: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-install
CMakeFiles/pybind11-download-complete: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-test
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'pybind11-download'"
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles/pybind11-download-complete
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-done

pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-install: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No install step for 'pybind11-download'"
	cd /Users/Roll/desktop/libigl/build/pybind11-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/pybind11-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-install

pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'pybind11-download'"
	/opt/local/bin/cmake -E make_directory /Users/Roll/desktop/libigl/cmake/../external/pybind11
	/opt/local/bin/cmake -E make_directory /Users/Roll/desktop/libigl/build/pybind11-build
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/tmp
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/src
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-mkdir

pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-download: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-gitinfo.txt
pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-download: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'pybind11-download'"
	cd /Users/Roll/desktop/libigl/external && /opt/local/bin/cmake -P /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/tmp/pybind11-download-gitclone.cmake
	cd /Users/Roll/desktop/libigl/external && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-download

pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-update: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing update step for 'pybind11-download'"
	cd /Users/Roll/desktop/libigl/external/pybind11 && /opt/local/bin/cmake -P /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/tmp/pybind11-download-gitupdate.cmake

pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-patch: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No patch step for 'pybind11-download'"
	/opt/local/bin/cmake -E echo_append
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-patch

pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-configure: pybind11-download-prefix/tmp/pybind11-download-cfgcmd.txt
pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-configure: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-update
pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-configure: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No configure step for 'pybind11-download'"
	cd /Users/Roll/desktop/libigl/build/pybind11-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/pybind11-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-configure

pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-build: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No build step for 'pybind11-download'"
	cd /Users/Roll/desktop/libigl/build/pybind11-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/pybind11-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-build

pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-test: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "No test step for 'pybind11-download'"
	cd /Users/Roll/desktop/libigl/build/pybind11-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/pybind11-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/pybind11/pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-test

pybind11-download: CMakeFiles/pybind11-download
pybind11-download: CMakeFiles/pybind11-download-complete
pybind11-download: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-install
pybind11-download: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-mkdir
pybind11-download: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-download
pybind11-download: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-update
pybind11-download: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-patch
pybind11-download: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-configure
pybind11-download: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-build
pybind11-download: pybind11-download-prefix/src/pybind11-download-stamp/pybind11-download-test
pybind11-download: CMakeFiles/pybind11-download.dir/build.make

.PHONY : pybind11-download

# Rule to build all files generated by this target.
CMakeFiles/pybind11-download.dir/build: pybind11-download

.PHONY : CMakeFiles/pybind11-download.dir/build

CMakeFiles/pybind11-download.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pybind11-download.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pybind11-download.dir/clean

CMakeFiles/pybind11-download.dir/depend:
	cd /Users/Roll/Desktop/libigl/external/.cache/pybind11 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/Desktop/libigl/external/.cache/pybind11 /Users/Roll/Desktop/libigl/external/.cache/pybind11 /Users/Roll/Desktop/libigl/external/.cache/pybind11 /Users/Roll/Desktop/libigl/external/.cache/pybind11 /Users/Roll/Desktop/libigl/external/.cache/pybind11/CMakeFiles/pybind11-download.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pybind11-download.dir/depend


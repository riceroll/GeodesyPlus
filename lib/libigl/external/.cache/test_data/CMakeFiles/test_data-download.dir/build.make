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
CMAKE_SOURCE_DIR = /Users/Roll/Desktop/libigl/external/.cache/test_data

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Roll/Desktop/libigl/external/.cache/test_data

# Utility rule file for test_data-download.

# Include the progress variables for this target.
include CMakeFiles/test_data-download.dir/progress.make

CMakeFiles/test_data-download: CMakeFiles/test_data-download-complete


CMakeFiles/test_data-download-complete: test_data-download-prefix/src/test_data-download-stamp/test_data-download-install
CMakeFiles/test_data-download-complete: test_data-download-prefix/src/test_data-download-stamp/test_data-download-mkdir
CMakeFiles/test_data-download-complete: test_data-download-prefix/src/test_data-download-stamp/test_data-download-download
CMakeFiles/test_data-download-complete: test_data-download-prefix/src/test_data-download-stamp/test_data-download-update
CMakeFiles/test_data-download-complete: test_data-download-prefix/src/test_data-download-stamp/test_data-download-patch
CMakeFiles/test_data-download-complete: test_data-download-prefix/src/test_data-download-stamp/test_data-download-configure
CMakeFiles/test_data-download-complete: test_data-download-prefix/src/test_data-download-stamp/test_data-download-build
CMakeFiles/test_data-download-complete: test_data-download-prefix/src/test_data-download-stamp/test_data-download-install
CMakeFiles/test_data-download-complete: test_data-download-prefix/src/test_data-download-stamp/test_data-download-test
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'test_data-download'"
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles/test_data-download-complete
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/src/test_data-download-stamp/test_data-download-done

test_data-download-prefix/src/test_data-download-stamp/test_data-download-install: test_data-download-prefix/src/test_data-download-stamp/test_data-download-build
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No install step for 'test_data-download'"
	cd /Users/Roll/desktop/libigl/build/test_data-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/test_data-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/src/test_data-download-stamp/test_data-download-install

test_data-download-prefix/src/test_data-download-stamp/test_data-download-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Creating directories for 'test_data-download'"
	/opt/local/bin/cmake -E make_directory /Users/Roll/desktop/libigl/cmake/../external/../tests/data
	/opt/local/bin/cmake -E make_directory /Users/Roll/desktop/libigl/build/test_data-build
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/tmp
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/src/test_data-download-stamp
	/opt/local/bin/cmake -E make_directory /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/src
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/src/test_data-download-stamp/test_data-download-mkdir

test_data-download-prefix/src/test_data-download-stamp/test_data-download-download: test_data-download-prefix/src/test_data-download-stamp/test_data-download-gitinfo.txt
test_data-download-prefix/src/test_data-download-stamp/test_data-download-download: test_data-download-prefix/src/test_data-download-stamp/test_data-download-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Performing download step (git clone) for 'test_data-download'"
	cd /Users/Roll/desktop/libigl/tests && /opt/local/bin/cmake -P /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/tmp/test_data-download-gitclone.cmake
	cd /Users/Roll/desktop/libigl/tests && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/src/test_data-download-stamp/test_data-download-download

test_data-download-prefix/src/test_data-download-stamp/test_data-download-update: test_data-download-prefix/src/test_data-download-stamp/test_data-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing update step for 'test_data-download'"
	cd /Users/Roll/desktop/libigl/tests/data && /opt/local/bin/cmake -P /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/tmp/test_data-download-gitupdate.cmake

test_data-download-prefix/src/test_data-download-stamp/test_data-download-patch: test_data-download-prefix/src/test_data-download-stamp/test_data-download-download
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No patch step for 'test_data-download'"
	/opt/local/bin/cmake -E echo_append
	/opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/src/test_data-download-stamp/test_data-download-patch

test_data-download-prefix/src/test_data-download-stamp/test_data-download-configure: test_data-download-prefix/tmp/test_data-download-cfgcmd.txt
test_data-download-prefix/src/test_data-download-stamp/test_data-download-configure: test_data-download-prefix/src/test_data-download-stamp/test_data-download-update
test_data-download-prefix/src/test_data-download-stamp/test_data-download-configure: test_data-download-prefix/src/test_data-download-stamp/test_data-download-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "No configure step for 'test_data-download'"
	cd /Users/Roll/desktop/libigl/build/test_data-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/test_data-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/src/test_data-download-stamp/test_data-download-configure

test_data-download-prefix/src/test_data-download-stamp/test_data-download-build: test_data-download-prefix/src/test_data-download-stamp/test_data-download-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No build step for 'test_data-download'"
	cd /Users/Roll/desktop/libigl/build/test_data-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/test_data-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/src/test_data-download-stamp/test_data-download-build

test_data-download-prefix/src/test_data-download-stamp/test_data-download-test: test_data-download-prefix/src/test_data-download-stamp/test_data-download-install
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "No test step for 'test_data-download'"
	cd /Users/Roll/desktop/libigl/build/test_data-build && /opt/local/bin/cmake -E echo_append
	cd /Users/Roll/desktop/libigl/build/test_data-build && /opt/local/bin/cmake -E touch /Users/Roll/Desktop/libigl/external/.cache/test_data/test_data-download-prefix/src/test_data-download-stamp/test_data-download-test

test_data-download: CMakeFiles/test_data-download
test_data-download: CMakeFiles/test_data-download-complete
test_data-download: test_data-download-prefix/src/test_data-download-stamp/test_data-download-install
test_data-download: test_data-download-prefix/src/test_data-download-stamp/test_data-download-mkdir
test_data-download: test_data-download-prefix/src/test_data-download-stamp/test_data-download-download
test_data-download: test_data-download-prefix/src/test_data-download-stamp/test_data-download-update
test_data-download: test_data-download-prefix/src/test_data-download-stamp/test_data-download-patch
test_data-download: test_data-download-prefix/src/test_data-download-stamp/test_data-download-configure
test_data-download: test_data-download-prefix/src/test_data-download-stamp/test_data-download-build
test_data-download: test_data-download-prefix/src/test_data-download-stamp/test_data-download-test
test_data-download: CMakeFiles/test_data-download.dir/build.make

.PHONY : test_data-download

# Rule to build all files generated by this target.
CMakeFiles/test_data-download.dir/build: test_data-download

.PHONY : CMakeFiles/test_data-download.dir/build

CMakeFiles/test_data-download.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_data-download.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_data-download.dir/clean

CMakeFiles/test_data-download.dir/depend:
	cd /Users/Roll/Desktop/libigl/external/.cache/test_data && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Roll/Desktop/libigl/external/.cache/test_data /Users/Roll/Desktop/libigl/external/.cache/test_data /Users/Roll/Desktop/libigl/external/.cache/test_data /Users/Roll/Desktop/libigl/external/.cache/test_data /Users/Roll/Desktop/libigl/external/.cache/test_data/CMakeFiles/test_data-download.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test_data-download.dir/depend

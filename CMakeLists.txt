cmake_minimum_required(VERSION 3.1)
project(geodesy)

#Versioning
set(SHAPEOP_VERSION_MAJOR 0)
set(SHAPEOP_VERSION_MINOR 1)
set(SHAPEOP_VERSION_PATCH 0)
set(SHAPEOP_VERSION_STRING ${SHAPEOP_VERSION_MAJOR}.${SHAPEOP_VERSION_MINOR}.${SHAPEOP_VERSION_PATCH})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


# libigl
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           OFF)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         OFF)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"          ON)
option(LIBIGL_WITH_PNG               "Use PNG"            OFF)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_XML               "Use XML"            OFF)

find_package(LIBIGL REQUIRED QUIET)


#Tests for Clang and AppleClang which is defined in CMake 3.0 and higher
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message("Clang.")
  set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++ -Wall -g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  #    set(CMAKE_CXX_FLAGS_DEBUG "-g")
  add_definitions(-DSHAPEOP_CLANG)
  add_definitions(-DSHAPEOP_EXPORT)
  set(OPENMP OFF CACHE BOOL "OpenMP")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  message("GCC.")
  set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -g")
  set(CMAKE_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG")
  #    set(CMAKE_CXX_FLAGS_DEBUG "-g")
  add_definitions(-DSHAPEOP_GCC)
  add_definitions(-DSHAPEOP_EXPORT)
  set(OPENMP ON CACHE BOOL "OpenMP")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  message("ICC.")
  if(WIN32)
    set(CMAKE_CXX_FLAGS "/Qstd=c++11")
    set(CMAKE_CXX_FLAGS_RELEASE "/O3 /DNDEBUG")
    add_definitions(/DSHAPEOP_ICC)
    add_definitions(/DSHAPEOP_EXPORT)
  else()
    set(CMAKE_CXX_FLAGS "-std=c++11")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    add_definitions(-DSHAPEOP_ICC)
    add_definitions(-DSHAPEOP_EXPORT)
  endif()
  set(OPENMP OFF CACHE BOOL "OpenMP")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  message("MSVC.")
  set(CMAKE_CXX_FLAGS_DEBUG "/MTd")
  set(CMAKE_CXX_FLAGS_RELEASE "/MT /Ox /Ot /fp:fast /GS- /GL /DNDEBUG")
  set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG /INCREMENTAL:NO /OPT:REF")
  add_definitions(/DSHAPEOP_MSVC)
  add_definitions(/DSHAPEOP_EXPORT)
  set(OPENMP ON CACHE BOOL "OpenMP")
else()
  message( FATAL_ERROR "CXX compiler not recognized. CMake will quit." )
endif()


include_directories("./include/")
include_directories("./lib/libShapeOp/src")

add_subdirectory(lib/libShapeOp)

# Add your project files
#file(GLOB SRCFILES *.cpp)

# ============================= linking ===============================
option(MAIN             "compile geodesy executive"    ON)


add_library(tracer SHARED tracer.cpp)   # compile tracer.cpp to a shared library
target_link_libraries(tracer  igl::core igl::opengl_glfw_imgui)   # link igl and imgui to tracer

add_library(geometry SHARED geometry.cpp)   # compile flattener.cpp to a shared library
target_link_libraries(geometry  igl::core igl::opengl_glfw_imgui)    # link igl and imgui to flattener

add_library(model SHARED model.cpp)   # compile flattener.cpp to a shared library
target_link_libraries(model  geometry)    # link igl and imgui to flattener
target_link_libraries(model  ShapeOp)
target_link_libraries(model  igl::core igl::opengl_glfw_imgui)    # link igl and imgui to flattener


if (MAIN)
  message("Compile main.")
  add_executable(${PROJECT_NAME} main.cpp)    # compile geodesy.cpp to executable
  target_link_libraries(${PROJECT_NAME}  igl::core igl::opengl_glfw_imgui ShapeOp)    # link igl
  target_link_libraries(${PROJECT_NAME}  tracer)    # link tracer
  target_link_libraries(${PROJECT_NAME}  geometry) # link flattener
  target_link_libraries(${PROJECT_NAME}  model) # link flattener
else()
  message("Does not compile main.")
endif()






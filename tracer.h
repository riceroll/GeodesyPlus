#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <string>

// igl
#include <igl/opengl/glfw/Viewer.h>

// Eigen
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Dense>

// geodesy
#include "flattener.h"

using namespace std;

#pragma once

struct HalfedgeMeshTr;
struct NodeTr;
struct EdgeTr;
struct FaceTr;
struct HalfedgeTr;
struct NodeTr_trace;
struct NodeTr_cycle;


struct HalfedgeMeshTr {
  // new trace data
  vector<NodeTr *> nodes;
  vector<EdgeTr *> edges;
  vector<FaceTr *> faces;
  vector<HalfedgeTr *> halfedges;
  vector<Halfedge *> boundary; // boundary of the flattened mesh
  vector<HalfedgeTr *> boundary_tr; // boundary of the rebuilt mesh
  vector<vector<NodeTr_cycle *>> cycles_next;
  FaceTr *face_exterior;

  vector<NodeTr_cycle *> nodes_next;

  vector<vector<NodeTr_trace *>> nodes_tr_traces;
  set<NodeTr *> nodes_accepted;    // source of heat method

  void trim_mesh(igl::opengl::glfw::Viewer& viewer, int test2);
};

struct NodeTr {
  int id = -1;

  HalfedgeTr* halfedge = nullptr;

  Eigen::RowVector3d pos;

  double geodesic;
  double geodesy;
};

struct HalfedgeTr {
  int id = -1;

  NodeTr* node = nullptr;
  EdgeTr* edge = nullptr;
  FaceTr* face = nullptr;
  HalfedgeTr* prev = nullptr;
  HalfedgeTr* next = nullptr;
  HalfedgeTr* twin = nullptr;

  Eigen::RowVector3d vector();

  void draw(igl::opengl::glfw::Viewer &viewer, bool label, int id);
};

struct EdgeTr {
  int id = -1;

  HalfedgeTr* halfedge = nullptr;

  NodeTr_trace* node_trace = nullptr;
  NodeTr_cycle* node_cycle = nullptr;

  Eigen::RowVector3d centroid();
};

struct FaceTr {
  int id = -1;

  HalfedgeTr* halfedge = nullptr;

  bool is_exterior = false;

  Eigen::RowVector3d centroid();
};

struct NodeTr_trace {
  EdgeTr* edge = nullptr;

  NodeTr_trace* left = nullptr;
  NodeTr_trace* right = nullptr;
  NodeTr_trace* up = nullptr;
  NodeTr_trace* down = nullptr;

  Eigen::RowVector3d pos;
};

struct NodeTr_cycle {
  HalfedgeTr* halfedge = nullptr;   // the halfedge this sits on

  NodeTr_cycle* left = nullptr;
  NodeTr_cycle* right = nullptr;
  NodeTr_cycle* up = nullptr;
  NodeTr_cycle* down = nullptr;

  int id_halfedge_right = -1;
  HalfedgeTr* halfedge_right = nullptr;   // the halfedge pointing from this to right node

  double t;

  Eigen::RowVector3d pos();
};

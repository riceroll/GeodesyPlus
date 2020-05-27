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

using namespace std;


#pragma once

struct HalfedgeMesh;
struct Node;
struct Edge;
struct Face;
struct Halfedge;

struct HalfedgeMesh {
  Eigen::RowVector3d center;

  vector<Node*> nodes;
  vector<Edge*> edges;
  vector<Face*> faces;
  vector<Halfedge*> halfedges;
  vector<vector<Node *>> boundaries_top;
  vector<vector<Node *>> boundaries_saddle;
  vector<Node*> boundary_bottom;
  vector<Node*> saddles;
  vector<Node*> cones;

  vector<vector<vector<Node *>>> isos_node;  // iso -> loop -> Node
  vector<vector<vector<Face *>>> iso_faces;  // iso -> loop -> Face
  vector<vector<Node*>> segments{}; // for merging
  set<Node*> unvisited{};    // for tracing
  vector<Node*> unvisited_vector{}; // for tracing
  vector<Node*> printing_path{};  // for tracing


  float iso_spacing = 10.0;
  float gap_size;



  void reorder_iso();


};


struct Node {
  int idx = -1;
  int idx_iso = -1;   // idx of the isoline (not segment)
  int idx_grad = -1;  // idx of the gradient line, if equals to -1, this is upsampled node
  bool is_cone = false;
  bool is_saddle = false;
  bool on_saddle_boundary = false;
  bool on_saddle_side = false;
  bool is_interp_bridge = false;

  Eigen::RowVector3d pos;
  Eigen::RowVector3d pos_origin;
  Eigen::RowVector3d velocity;

  Node* left = nullptr;
  Node* right = nullptr;
  Node* up = nullptr;
  Node* down = nullptr;
  Node* left_saddle = nullptr;  // for saddle fan
  Node* right_saddle = nullptr; // for saddle fan
  set<Edge*> edges;
  Halfedge* halfedge = nullptr;

  // for interpolation & merging
  int i_unvisited_vector = -1;
  int i_path = -1;
  bool visited_interpolation = false;
  bool is_end = false;
  bool is_left_end = false;
  bool is_right_end = false;
  vector<Node*> ups{};
  vector<Node*> downs{};

  // for printing
  bool is_move = false; // extrude
  float shrinkage = -1; // defaut shrinkage

  // for mapping
  Eigen::RowVector3d color = Eigen::RowVector3d(1.0, 0.0, 0.0);
};

struct Edge {
  int idx = -1;
  string spring = "stretch";      // stretch

  float len_3d;     // initial length on 3D surface
  float len;        // current length
  float len_prev;   // compute convergence
  float rest_len;
  float shrinkage;  // shrinkage = 1 - len_3d / len; how much it shrinks by

  set<Node*> nodes;
  Halfedge* halfedge = nullptr;
  vector<Node*> nodes_interp_as_left{};   // begin from e->stretch
  vector<Node*> nodes_interp_as_right{};
  vector<Node*> nodes_interp{};

  float length();
  Eigen::RowVector3d centroid();
};

struct Halfedge {
  int idx = -1;

  Node* node = nullptr;
  Edge* edge = nullptr;
  Face* face = nullptr;
  Halfedge* prev = nullptr;
  Halfedge* next = nullptr;
  Halfedge* twin = nullptr;

  float length();
  Eigen::RowVector3d vector();
};

struct Face {
  int idx = -1;
  bool is_external = false;
  bool is_saddle = false;

  Halfedge* halfedge = nullptr;

  Eigen::RowVector3d centroid();
  Eigen::RowVector3d normal();

  Node* node(int i);

  // for interpolation
  bool is_up = true;
  bool visited_interpolation = false;
  Edge *e_stretch = nullptr;
  Edge *e_bridge_left = nullptr;
  Edge *e_bridge_right = nullptr;
  Node *n_bridge = nullptr;
  Node *n_stretch_left = nullptr;
  Node *n_stretch_right = nullptr;
  Face* left = nullptr;
  Face* right =nullptr;
  Eigen::RowVector3d pos_stretch_mid = Eigen::RowVector3d(0, 0, 0);
  vector<Eigen::RowVector3d> pos_interps;
  int num_interp = 0;
  Eigen::RowVector3d vec_bridge = Eigen::RowVector3d(0,0,0);
  Eigen::RowVector3d vec_stretch = Eigen::RowVector3d(0,0,0);

  double area();

  double area_origin();

  double opacity();

};

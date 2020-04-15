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


struct Node;
struct Edge;
struct Face;
struct Halfedge;

struct HalfedgeMeshTr;
struct NodeTr;
struct EdgeTr;
struct FaceTr;
struct HalfedgeTr;
struct NodeTr_trace;
struct NodeTr_cycle;

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


struct HalfedgeMeshTr {
  // new trace data
  vector<NodeTr*> nodes;
  vector<EdgeTr*> edges;
  vector<FaceTr*> faces;
  vector<HalfedgeTr*> halfedges;
  vector<Halfedge*> boundary; // boundary of the flattened mesh
  vector<HalfedgeTr*> boundary_tr; // boundary of the rebuilt mesh
  vector<vector<NodeTr_cycle*>> cycles_next;
  FaceTr* face_exterior;

  vector<NodeTr_cycle*> nodes_next;

  vector<vector<NodeTr_trace*>> nodes_tr_traces;
  set<NodeTr*> nodes_accepted;    // source of heat method
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

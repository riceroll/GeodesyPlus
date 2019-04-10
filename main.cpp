#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <queue>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <thread>
#include <string>
#include <boost/algorithm/string/classification.hpp> // Include boost::for is_any_of
#include <boost/algorithm/string/split.hpp> // Include for boost::split

#include <imgui/imgui.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_ray.h>
#include <igl/colormap.h>
#include <igl/avg_edge_length.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Eigen/Dense>

#include "Solver.h"
#include "Constraint.h"
#include "Force.h"

using namespace std;

struct Node;
struct Edge;
struct Face;
struct Halfedge;

struct Node {
  int idx = -1;
  Halfedge* halfedge = nullptr;

  Eigen::RowVector3d pos;
  Eigen::RowVector3d pos_origin;
  int idx_iso = -1; // idx of the isoline (not segment)
  int idx_grad = -1;  // idx of the gradient line, if equals to -1, this is upsampled node
  Node* left = nullptr;
  Node* right = nullptr;
  Node* up = nullptr;
  Node* down = nullptr;
};

struct Edge {
  int idx = -1;
  Halfedge* halfedge = nullptr;
};

struct Face {
  int idx = -1;
  Halfedge* halfedge = nullptr;
};

struct Halfedge {
  int idx = -1;
  Node* node = nullptr;
  Edge* edge = nullptr;
  Face* face = nullptr;
  Halfedge* prev = nullptr;
  Halfedge* next = nullptr;
  Halfedge* twin = nullptr;
};


int main(int argc, char **argv) {
  Eigen::MatrixXd V;  // n_vertices * 3d
  Eigen::MatrixXi F;  // n_faces * 3i
  vector<Node*> graph;
  vector<Face*> faces;
  vector<Halfedge*> halfedges;
  map<set<int>, Edge*> edges; // set of two vertices indices to Edge*

  ShapeOp::Solver* solver;
  Eigen::MatrixXd points;
  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiMenu menu;

  int idx_n_selected = 0;
  float avg_len_iso_edge = -1;
  float avg_len_grad_edge = -1;
  float iso_spacing = 1;
  float grad_spacing = 1;
  bool display_label = false;

  int ii = 0;
  auto redraw = [&](string type = "graph") {
    // type = "graph" / "edges" / "mesh"
    viewer.data().clear();

    viewer.data().points = Eigen::MatrixXd(0, 6);
    viewer.data().lines = Eigen::MatrixXd(0, 9);
    viewer.data().labels_positions = Eigen::MatrixXd(0, 3);
    viewer.data().labels_strings.clear();


    if (type == "graph") {
      viewer.data().points = Eigen::MatrixXd(0, 6);
      viewer.data().lines = Eigen::MatrixXd(0, 9);
      viewer.data().labels_positions = Eigen::MatrixXd(0, 3);
      viewer.data().labels_strings.clear();
      for (auto n : graph) {
        if (n->idx == idx_n_selected)
          viewer.data().add_points(n->pos, Eigen::RowVector3d(0.9, 0.9, 0));
        else
          viewer.data().add_points(n->pos, Eigen::RowVector3d(0, 0.9, 0.9));
        if (display_label)
          viewer.data().add_label(n->pos, to_string(n->idx));
        if (n->right) viewer.data().add_edges(n->pos, n->right->pos, Eigen::RowVector3d(0.9, 0, 0));
        if (n->up) viewer.data().add_edges(n->pos, n->up->pos, Eigen::RowVector3d(0, 0.9, 0));
      }
    }


//    else if (type == "edges") {
//       from A + edges* to points, lines
//       A + edges
//    }

  };

  auto subdivide_edge = [&](Node* n) {
    if (n->right) {
      Eigen::RowVector3d vec = n->right->pos - n->pos;
      int num_insert = floor(vec.norm() / iso_spacing); // number of inserted vertices on one iso line
      if (num_insert == 1) num_insert = 0;
      float t_step = 1.0 / (num_insert + 1);

      Node *n_prev = n;
      for (int i = 1; i <= num_insert; i++) {
        Node *n_new = new Node();
        n_new->idx = graph.size();
        n_new->pos = n->pos + vec * t_step * i;
        n_new->pos_origin = n->pos + vec * t_step * i;
        n_new->idx_iso = n->idx_iso;
        n_new->left = n_prev;
        n_new->right = n_prev->right;
        n_new->left->right = n_new;
        n_new->right->left = n_new;
        n_prev = n_new;
        graph.push_back(n_new);
      }
    }

  };

  //////////////////////////////////// init ////////////////////////////////////
  { // load graph
    string in_file_name;
    if (argv[1]) in_file_name = argv[1];
    else cerr << "arg[1] is required." << endl;
    ifstream ifile(in_file_name);

    string line;
    while (getline(ifile, line, '\n')) {
      auto n = new Node();
      graph.push_back(n);

    }
    ifile.clear();
    ifile.seekg(0, ios::beg);

    while (getline(ifile, line, '\n')) {
      vector<string> items;
      boost::split(items, line, boost::is_any_of(" "), boost::token_compress_on);
      int idx = stoi(items[0]);
      Node *n = graph[idx];
      n->idx = stoi(items[0]); // -1: not defined; -1: inserted
      n->pos = Eigen::RowVector3d(stof(items[1]), stof(items[2]), stof(items[3]));
      n->pos_origin = Eigen::RowVector3d(stof(items[1]), stof(items[2]), stof(items[3]));
      if (items[4] != "-1") n->idx_iso = stoi(items[4]);
      if (items[5] != "-1") n->idx_grad = stoi(items[5]);
      if (items[6] != "-1") n->right = graph[stoi(items[6])];
      if (items[7] != "-1") n->left = graph[stoi(items[7])];
      if (items[8] != "-1") n->up = graph[stoi(items[8])];
      if (items[9] != "-1") n->down = graph[stoi(items[9])];
    }
  }

  { // compute avg_len
    float sum_len_iso_seg = 0;
    float sum_len_grad_seg = 0;
    int n_iso_seg = 0;
    int n_grad_seg = 0;
    for (auto n : graph ) {
      if (n->right) sum_len_iso_seg += (n->pos - n->right->pos).norm();
      if (n->up) sum_len_grad_seg += (n->pos - n->up->pos).norm();
      n_iso_seg ++; n_grad_seg++;

    }
    avg_len_iso_edge = sum_len_iso_seg / n_iso_seg;
    avg_len_grad_edge = sum_len_grad_seg / n_grad_seg;
    cout<<"avg_iso: "<<avg_len_iso_edge<<" "<<"avg_grad: "<<avg_len_grad_edge<<"graph_size: "<<graph.size()<<endl;
  }

  { // visualizer
    viewer.core.background_color = Eigen::Vector4f(1, 1, 1, 1.);
    viewer.core.camera_base_zoom = 0.01;
    viewer.plugins.push_back(&menu);
//    redraw();
  }


  // callbacks
  menu.callback_draw_viewer_menu = [&]()
  {
    if (ImGui::CollapsingHeader("Toolbar", ImGuiTreeNodeFlags_DefaultOpen))
    {
      ImGui::InputFloat("iso_spacing", &iso_spacing);
      ImGui::InputFloat("grad_spacing", &grad_spacing);
      if (ImGui::InputInt("selected_node", &idx_n_selected)) {
        redraw();
      }
      if (ImGui::Checkbox("label", &display_label)) {
        redraw();
      }
      if (ImGui::Button("clear")){
        viewer.data().points = Eigen::MatrixXd(0,6);
        viewer.data().lines = Eigen::MatrixXd(0,9);
      }
      if (ImGui::Button("redraw")) {
        redraw();
      }
      if (ImGui::Button("subdivide isolines")) {
        for (auto n : graph) {
          subdivide_edge(n);

        }
        cout<<"Done. avg_iso: "<<avg_len_iso_edge<<" "<<"avg_grad: "<<avg_len_grad_edge<<"graph_size: "<<graph.size()<<endl;
        redraw();
      }

      if (ImGui::Button("flatten")) {
        vector<int> test;
        cout<<test[2]<<endl;
      }


      if (ImGui::Button("try mesh")) {
        viewer.data().clear();
        V.resize(4,3);
        F.resize(4,3);
        V << 0,0,0,
             10,0,0,
             0,10,0,
             0,0,10;

        F << 0,1,2,
             2,1,3,
             1,3,0,
             0,2,3;
        viewer.data().set_mesh(V, F);
        Eigen::MatrixXd FC, FN;
        FC = Eigen::MatrixXd::Ones(4,3);
        FC = FC * 0.5;
        viewer.data().set_colors(FC);
//        igl::per_face_normals(V, F, FN);
      }


      if (ImGui::Button("triangulate")) {

        int iii = 0;
        for (auto n : graph) {
//        {
//          Node* n = graph[idx_n_selected];

          // detect quads (up right quad of each node)
          if (n->idx_grad != -1 and (n->right and n->up)) {  // original nodes in the graph, has up right
            Node *n_right = n->right;
            Node *n_up = n->up;
            Node *n_up_right = n_up->right;
            while (n_right->idx_grad == -1 or !n_right->up) {
              n_right = n_right->right;
            }
            while (n_up_right->idx_grad == -1 or !n_up_right->down) {
              n_up_right = n_up_right->right;
            }

            if (n_right->up == n_up_right) {  // in a quad
              Node* n_d = n;
              Node* n_u = n->up;
              bool right_most = false;
              // keep moving towards the right edge
              while (true) {
                if (n_u == n_up_right) {
                  if (n_d->right == n_right) {
                    right_most = true;
                  }
                  else {
                    n_d = n_d->right;
                  }
                }
                else if (n_d == n_right) {
                  if (n_u->right == n_up_right) {
                    right_most = true;
                  }
                  else {
                    n_u = n_u->right;
                  }
                }
                else {
                  Eigen::RowVector3d vec_d = n_u->right->pos - n_d->pos;
                  Eigen::RowVector3d vec_u = n_d->right->pos - n_u->pos;

                  if ( (vec_d.norm() < vec_u.norm()) or (n_d == n_right) ) {  // connect the shorter diagonal
                    n_u = n_u->right;
                  } else {
                    n_d = n_d->right;
                  }
                }

                if (right_most) {
                  break;
                }
              }

            }
          }

          // detect top boundary
          if (!n->up) { // detect the up boundary
            vector<Node*> boundary_top;
            bool is_boundary = true;
            Node* n_iter = n;
            while (true) {
              if (n_iter->up) {
                is_boundary = false;
                boundary_top.clear();

                break;
              }
              boundary_top.push_back(n_iter);

              if (n_iter->right == n) {  // close the boundary
                break;
              }
              n_iter = n_iter->right;
            }

            if (is_boundary) {
              Eigen::RowVector3d center_pos = Eigen::RowVector3d(0, 0, 0);
              for (auto nb : boundary_top) {
                center_pos += nb->pos;
              }
              center_pos /= boundary_top.size();

              Node* n_center = new Node();
              n_center->idx = graph.size();
              n_center->pos = center_pos;
              n_center->pos_origin = center_pos;
              n_center->idx_iso = -1; // TODO: indexing for inserted isoline
              n_center->idx_grad = -1;

              graph.push_back(n_center);
            }

            else {
              // TODO: two iso lines sharing a hole but no node has a downward connection
            }
          }

          // detect holes
          if (!n->down) { // detecting the bottom boundary
            vector<Node*> boundary_down;
            bool is_boundary = true;  // not used yet
            bool is_bottom_boundary = true;
            Node* n_iter = n;
            bool is_upper = true;
            set<int> ids_iso;

            while (true) {
              ids_iso.emplace(n_iter->idx_iso);
              if (is_upper) {
                if (n_iter->down) {
                  n_iter = n_iter->down;
                  boundary_down.push_back(n_iter);
                  n_iter = n_iter->right;
                  is_bottom_boundary = false;
                  is_upper = false;
                }
                else if (n_iter->left) {
                  n_iter = n_iter->left;
                }
              }
              else {
                if (n_iter->up) {
                  n_iter = n_iter->up;
                  boundary_down.push_back(n_iter);
                  n_iter = n_iter->left;
                  is_upper = true;
                }
                else if (n_iter->right) {
                  n_iter = n_iter->right;
                }
              }

              boundary_down.push_back(n_iter);

              if (n_iter == n) {  // boundary closed
                break;
              }
            }

            // not bottom, not same path
            if ( (is_boundary) and (!is_bottom_boundary) and (ids_iso.size() > 2 ) ) {
              // TODO: skeletonize the saddle, here using center point temporally

              Eigen::RowVector3d center_pos = Eigen::RowVector3d(0, 0, 0);

              for (auto n : boundary_down) {
                center_pos += n->pos;
              }
              center_pos /= boundary_down.size();

              for (auto n : boundary_down) {
                viewer.data().add_edges(center_pos, n->pos, Eigen::RowVector3d(0.9, 0, 0));
              }

            }
          }

          if (!n->right) {cerr<<n->idx<<" has no right node."<<endl;}
        }

        // draw edges from halfedge mesh
        for (auto ei = edges.begin(); ei != edges.end(); ei++) {
          Edge* e = ei->second;
          Node* n_a = e->halfedge->node;
          if (e->halfedge->twin) {
            Node *n_b = e->halfedge->twin->node;
            viewer.data().add_edges(n_a->pos, n_b->pos, Eigen::RowVector3d(0.9, 0, 0));
          }
          else {

          }
        }

      }

    }
  };


  viewer.callback_key_down =
    [&](igl::opengl::glfw::Viewer& viewer, int key, int mod)->bool
    {
      if (key == GLFW_KEY_LEFT)
        if (graph[idx_n_selected]->left) {
          idx_n_selected = graph[idx_n_selected]->left->idx;
          redraw();
        }

      if (key == GLFW_KEY_RIGHT)
        if (graph[idx_n_selected]->right) {
          idx_n_selected = graph[idx_n_selected]->right->idx;
          redraw();
        }
      if (key == GLFW_KEY_UP)
        if (graph[idx_n_selected]->up) {
          idx_n_selected = graph[idx_n_selected]->up->idx;
          redraw();
        }

      if (key == GLFW_KEY_DOWN)
        if (graph[idx_n_selected]->down) {
          idx_n_selected = graph[idx_n_selected]->down->idx;
          redraw();

        }
    };



  viewer.callback_mouse_down =
    [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
      // intersect ray with xy plane
      Eigen::RowVector3d v_xy;  // intersection between ray and xy plain
      double x = viewer.current_mouse_x;
      double y = viewer.core.viewport(3) - viewer.current_mouse_y;
      Eigen::Vector3d s, dir;
      igl::unproject_ray(Eigen::Vector2f(x,y), viewer.core.view, viewer.core.proj, viewer.core.viewport, s, dir);
      float t = - s[2] / dir[2];
      v_xy = s + t * dir;

      Eigen::RowVector3d color = Eigen::RowVector3d(0.6,0.6,0.8);
//      viewer.data().add_points(v_xy, color);

    };

  viewer.launch();


  return 0;

}

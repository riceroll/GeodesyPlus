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
#include <igl/ray_mesh_intersect.h>
#include <igl/unproject_ray.h>
//#include <igl/colormap.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

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

  float length() {
    return ((*next(nodes.begin(), 0))->pos - (*next(nodes.begin(), 1))->pos ).norm();
  }
  Eigen::RowVector3d centroid() {
    return ((*nodes.begin())->pos + (*next(nodes.begin(), 1))->pos) / 2;
  }
};

struct Halfedge {
  int idx = -1;

  Node* node = nullptr;
  Edge* edge = nullptr;
  Face* face = nullptr;
  Halfedge* prev = nullptr;
  Halfedge* next = nullptr;
  Halfedge* twin = nullptr;

  float length() {
    return this->edge->length();
  }
  Eigen::RowVector3d vector() {
    Eigen::RowVector3d p1 = this->node->pos;
    Eigen::RowVector3d p2 = this->twin->node->pos;
    Eigen::RowVector3d vec = p2 - p1;
    return vec;
  }
};

struct Face {
  int idx = -1;
  bool is_external = false;
  bool is_saddle = false;

  Halfedge* halfedge = nullptr;

  Eigen::RowVector3d centroid() {
    Eigen::RowVector3d p = this->halfedge->node->pos;
    p += this->halfedge->next->node->pos;
    p += this->halfedge->prev->node->pos;
    p /= 3;
    return p;
  }
  Eigen::RowVector3d normal() {
    Eigen::RowVector3d n = this->halfedge->vector().cross(this->halfedge->next->vector());
    n.normalize();
    return n;
  }
  Node* node(int i) {
    Halfedge* h = this->halfedge;
    if (i == 0) return h->node;
    if (i == 1) return h->next->node;
    if (i == 2) return h->next->next->node;
  }

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

  double area() {
    Eigen::RowVector3d v1 = this->halfedge->prev->node->pos - this->halfedge->node->pos;
    Eigen::RowVector3d v2 = this->halfedge->next->node->pos - this->halfedge->node->pos;
    double area = v1.cross(v2).norm() / 2;
    return area;
  }

  double area_origin() {
    Eigen::RowVector3d v1_origin = this->halfedge->prev->node->pos_origin - this->halfedge->node->pos_origin;
    Eigen::RowVector3d v2_origin = this->halfedge->next->node->pos_origin - this->halfedge->node->pos_origin;
    double area_origin = v1_origin.cross(v2_origin).norm() / 2;
    return area_origin;
  }

  double opacity() {
    double opacity = this->area_origin() / this->area();
    if (opacity > 1) return 1.0;
    if (opacity < 0) return 0.0;
    return opacity;
  }

};

int main(int argc, char **argv) {
  // declaration
  Eigen::MatrixXd V_in, TC_in, N_in, FC_in;
  Eigen::MatrixXi F_in, FTC_in, FN_in;

  Eigen::MatrixXd V_out, TC_out, N_out, FC_out;
  Eigen::MatrixXi F_out, FTC_out, FN_out;

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

  ShapeOp::Solver* solver = new ShapeOp::Solver();
  Eigen::MatrixXd points;

  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiMenu menu;

  // param
  int idx_focus = 0;
  string type_focus = "node";
  bool display_label = false;
  int display_mode = 0; // 0: graph, 1: halfedges, 2: mesh, 3: tracing
  int label_type = 0;   // 0: node, 1: isoline, boundary, 2: grad line, 3: spring
  bool display_stress = false; // 0: spring, 1: stress
  bool display_bridge = true;
  bool background_black = true;
  bool is_flattening = false;

  float iso_spacing = 10.0;
  float shrinkage_cone = 0.28;
  float shrinkage_cone_down = 0.1;
  float radius_cone = 5;
  float radius_trim = 0.45;
  int num_iter = 1;
  int filter_threshold = 3;
  int saddle_displacement = 3;
  float damping_flatten = 1.2;
  float damping;
//  float time_step;
  float w_closeness;
  float w_stretch;
  float w_bridge;
  float w_bending;
  float w_flatten;
  float w_angle_stretch;
  float w_angle_shear;
  float w_spreading;
  float w_smooth;
  float gap_size;

  // const
  Eigen::RowVector3d color_grey = Eigen::RowVector3d(0.5, 0.5, 0.5);
  Eigen::RowVector3d color_white = Eigen::RowVector3d(1., 1., 1.);
  Eigen::RowVector3d color_red = Eigen::RowVector3d(0.8, 0.2, 0.2);
  Eigen::RowVector3d color_green = Eigen::RowVector3d(0.2, 0.8, 0.2);
  Eigen::RowVector3d color_blue = Eigen::RowVector3d(0.2, 0.2, 0.8);
  Eigen::RowVector3d color_magenta = Eigen::RowVector3d(0.8, 0.2, 0.8);
  Eigen::RowVector3d color_cyon = Eigen::RowVector3d(0.2, 0.8, 0.8);
  Eigen::RowVector3d color_yellow = Eigen::RowVector3d(0.8, 0.8, 0.2);
  double platform_length = 200;
  double platform_width = platform_length;
  double scale_ratio;

  // mapping
  unsigned char *image_in;
  unsigned char *img_out;
  int w_in, h_in, n_c_in; // width, height, n_channel of image_in
  int w_out = 100;
  int n_c_out = 3;

  auto get_color = [&](float x) {
    double r, g, b;

    const double rone = 0.8;
    const double gone = 1.0;
    const double bone = 1.0;

    if (x == -1) return Eigen::RowVector3d(1.0, 1.0, 1.0);

    x = x / 0.33;
    if ( x > 1) x = 1;
    if ( x < -1) x = -1;
    x = (x + 1) / 2;


    if (x < 1. / 8.)
    {
      r = 0;
      g = 0;
      b = bone*(0.5 + (x) / (1. / 8.)*0.5);
    } else if (x < 3. / 8.)
    {
      r = 0;
      g = gone*(x - 1. / 8.) / (3. / 8. - 1. / 8.);
      b = bone;
    } else if (x < 5. / 8.)
    {
      r = rone*(x - 3. / 8.) / (5. / 8. - 3. / 8.);
      g = gone;
      b = (bone - (x - 3. / 8.) / (5. / 8. - 3. / 8.));
    } else if (x < 7. / 8.)
    {
      r = rone;
      g = (gone - (x - 5. / 8.) / (7. / 8. - 5. / 8.));
      b = 0;
    } else
    {
      r = (rone - (x - 7. / 8.) / (1. - 7. / 8.) * 0.5);
      g = 0;
      b = 0;
    }

    Eigen::RowVector3d c = Eigen::RowVector3d(r, g, b);

    return c;
  };


  auto get_pixel = [&](double u, double v)->Eigen::RowVector3d {
    // (image_in,w_in, h_in, n_c_in, x, y) -> Eigen::RowVector3d(r,g,b)
    u = 1.0 - u;
    v = 1.0 - v;
    int x = u * (w_in - 1);
    int y = v * (h_in - 1);
    unsigned char* px = image_in + w_in * n_c_in * y + n_c_in * x;
    double r = double(int(px[0])) / 255.;
    double g = double(int(px[1])) / 255.;
    double b = double(int(px[2])) / 255.;
    Eigen::RowVector3d color = Eigen::RowVector3d(r, g, b);

    return color;
  };

  auto redraw = [&]() {
    cout<<"start redraw..."<<endl;

//    viewer.data().clear();  // this line causes bug
    viewer.data().V.resize(0,3);
    viewer.data().F.resize(0,3);
    viewer.data().points.resize(0, 6);
    viewer.data().lines.resize(0, 9);
    viewer.data().labels_positions.resize(0, 3);
    viewer.data().labels_strings.clear();

    if (display_label) {

//      if (display_mode == 3 or display_mode == 4) {
//        for (auto n : unvisited_vector) {
//          viewer.data().add_label(n->pos, to_string(n->i_unvisited_vector));
//        }
//        viewer.data().add_points(unvisited_vector[idx_focus]->pos, color_white);
//      }

      for (auto n : nodes) {
        if (display_mode == 3 or display_mode == 4) { // segments or trace
          viewer.data().add_label(n->pos, to_string(n->i_unvisited_vector));
        }
        if (label_type == 0) {        // node
          viewer.data().add_label(n->pos, to_string(n->idx));
        }
        else if (label_type == 1)   // iso line
          viewer.data().add_label(n->pos, to_string(n->idx_iso));
        else if (label_type == 2)   // grad line
          viewer.data().add_label(n->pos, to_string(n->idx_grad));
      }

      if (label_type == 3) {
        for (auto f : faces) {
          viewer.data().add_label(f->centroid(), to_string(f->num_interp));
        }

        for (auto n : printing_path) {
//            viewer.data().add_label(n->pos, to_string(n->is_end));
        }
      }


      for (auto e : edges) {
        if (display_mode == 3 or display_mode == 4) continue; // segments or trace
        viewer.data().add_label(e->centroid(), to_string(e->idx));
      }

      for (auto f : faces) {
        if (display_mode == 3 or display_mode == 4) continue; // segments or trace
        if (label_type != 3) viewer.data().add_label(f->centroid(), to_string(f->idx));
        else viewer.data().add_label(f->centroid(), to_string(f->num_interp));
      }
    }

    if (display_mode == 0) {  // graph
      for (auto n : nodes) {
        if (n->idx == idx_focus)
          viewer.data().add_points(n->pos, Eigen::RowVector3d(0.9, 0.9, 0));
        else
          viewer.data().add_points(n->pos, Eigen::RowVector3d(0, 0.9, 0.9));
        if (n->right) viewer.data().add_edges(n->pos, n->right->pos, Eigen::RowVector3d(0.9, 0, 0));
        if (n->up) viewer.data().add_edges(n->pos, n->up->pos, Eigen::RowVector3d(0, 0.9, 0));
      }
    }

    if (display_mode == 1) { // halfedge

      Eigen::MatrixXd p1s(0, 3);
      Eigen::MatrixXd p2s(0, 3);
      Eigen::MatrixXd colors(0, 3);

      for (auto e : edges) {
        if ((not display_bridge) and (e->spring == "bridge")) continue;

        Node* n_a = *e->nodes.begin();
        Node* n_b = *next(e->nodes.begin(), 1);

        p1s.conservativeResize(p1s.rows() + 1, p1s.cols());
        p1s.row(p1s.rows() - 1) = n_a->pos;
        p2s.conservativeResize(p2s.rows() + 1, p2s.cols());
        p2s.row(p2s.rows() - 1) = n_b->pos;
        colors.conservativeResize(colors.rows() + 1, colors.cols());

        if (display_stress) {
          e->shrinkage = 1.0 - e->len_3d / e->length();   // shrinkage ratio
          Eigen::RowVector3d color_stress = get_color(e->shrinkage);
          colors.row(colors.rows() - 1) = color_stress;
        }
        else {
          if (n_a->on_saddle_boundary and n_b->on_saddle_boundary) {
            colors.row(colors.rows() - 1) = color_cyon;
          }
          else if (e->spring == "stretch") {
            colors.row(colors.rows() - 1) = color_red;
          }
          else if (e->spring == "boundary") {
            colors.row(colors.rows() - 1) = color_blue;
          }
          else if (not ((not display_bridge) and (e->spring == "bridge")) )  {
            colors.row(colors.rows() - 1) = color_green;
          }
        }
      }
      viewer.data().add_edges(p1s, p2s, colors);

      {  // draw focus
        Eigen::RowVector3d color_node;
        if (background_black) color_node = color_white;
        else color_node = color_grey;

        if (type_focus == "node") {
          if (display_mode == 3) {
            viewer.data().add_points(unvisited_vector[idx_focus]->pos, color_node);
          } else {
            viewer.data().add_points(nodes[idx_focus]->pos, color_node);
          }
        } else if (type_focus == "edge") {
          viewer.data().add_points(edges[idx_focus]->centroid(), color_node);
        } else if (type_focus == "face") {
          if (num_iter == 0)
            viewer.data().add_points(faces[idx_focus]->centroid(), color_node);
          else if (num_iter == 1)
            viewer.data().add_points(faces[idx_focus]->e_bridge_right->centroid(), color_node);
          else if (num_iter == 2)
            viewer.data().add_points(faces[idx_focus]->e_bridge_left->centroid(), color_node);
          else if (num_iter == 3)
            viewer.data().add_points(faces[idx_focus]->n_bridge->pos, color_node);
          else if (num_iter == 4)
            viewer.data().add_points(faces[idx_focus]->pos_stretch_mid, color_node);
          else if (num_iter == 5)
            viewer.data().add_points(faces[idx_focus]->e_bridge_left->nodes_interp_as_left[int(gap_size)]->pos,
                                     color_node);
          else if (num_iter == 6)
            viewer.data().add_points(faces[idx_focus]->e_bridge_right->nodes_interp_as_right[int(gap_size)]->pos,
                                     color_node);
          else if (num_iter == 7)
            viewer.data().add_points(faces[idx_focus]->pos_interps[int(gap_size)], color_node);
        } else if (type_focus == "halfedge") {
          viewer.data().add_points(halfedges[idx_focus]->edge->centroid(), color_node);
          viewer.data().add_points(halfedges[idx_focus]->node->pos, color_node);
        }
      }
    }

    if (display_mode == 2) { // connection

      for (auto n : unvisited_vector) {
        if (n->right) {
          viewer.data().add_edges( (n->right->pos - n->pos)/3 +n->pos, n->pos, color_green);
        }
        if (n->left) {
          viewer.data().add_edges( (n->left->pos - n->pos )/3 + n->pos, n->pos, color_red );
        }

        if (not n->ups.empty()) {
          for (int i = 0; i < n->ups.size(); i++) {
            viewer.data().add_edges( (n->ups[i]->pos - n->pos)/3 + n->pos, n->pos, color_yellow);
          }

        }
        else if (n->up) {
          viewer.data().add_edges( (n->up->pos - n->pos )/3 + n->pos, n->pos, color_cyon );
        }

        if (not n->downs.empty()) {
          for (int i = 0; i < n->downs.size(); i++) {
            viewer.data().add_edges( (n->downs[i]->pos - n->pos) / 3 + n->pos, n->pos, color_magenta );
          }
        }
        else if (n->down) {
          viewer.data().add_edges( (n->down->pos - n->pos )/3 + n->pos, n->pos, color_white );
        }

        if (n->right_saddle) {
          viewer.data().add_edges( (n->right_saddle->pos - n->pos) / 3 + n->pos, n->pos, color_blue);
        }
        if (n->left_saddle) {
          viewer.data().add_edges( (n->left_saddle->pos - n->pos) / 3 + n->pos, n->pos, color_blue ) ;
        }
      }
    }

    if (display_mode == 3) {  // segments

      if (saddle_displacement == 5) { // show is_end label
        for (auto n : printing_path) {
          viewer.data().add_label(n->pos, to_string(n->is_end));
        }
      }

      {
        Eigen::MatrixXd p1s_seg(segments.size(), 3);
        Eigen::MatrixXd p2s_seg(segments.size(), 3);
        Eigen::MatrixXd colors_seg(segments.size(), 3);
        int i = 0;
        for (auto segment : segments) {
          p1s_seg.row(i) = segment[0]->pos;
          p2s_seg.row(i) = segment[1]->pos;
          colors_seg.row(i) = color_green;
          i++;
        }
        viewer.data().add_edges(p1s_seg, p2s_seg, colors_seg);
      }


      {
        Eigen::MatrixXd p1s_e(edges.size(), 3);
        Eigen::MatrixXd p2s_e(edges.size(), 3);
        Eigen::MatrixXd colors_e(edges.size(), 3);

        int i = 0;
        for (auto e : edges) {
          Node *n_a = *e->nodes.begin();
          Node *n_b = *next(e->nodes.begin(), 1);

          p1s_e.row(i) = n_a->pos;
          p2s_e.row(i) = n_b->pos;

          if (e->spring == "stretch")
            colors_e.row(i) = color_red / 3;
          else if (e->spring == "boundary")
            colors_e.row(i) = color_blue/ 3;
          else if (not((not display_bridge) and (e->spring == "bridge")))
            colors_e.row(i) = color_green / 3;

          /*
          for (auto n : e->nodes_interp_as_left) {
            viewer.data().add_points(n->pos, color_yellow);
          }
          for (auto n : e->nodes_interp_as_right) {
            viewer.data().add_points(n->pos, color_cyon);
          }
           */

          i++;
        }

        viewer.data().add_edges(p1s_e, p2s_e, colors_e);

      }
    }

    if (display_mode == 4) {  // tracing
      for (auto n : unvisited_vector) {
//        viewer.data().add_points(n->pos, color_blue);
      }

      Eigen::MatrixXd p1s(printing_path.size()-1, 3);
      Eigen::MatrixXd p2s(printing_path.size()-1, 3);
      Eigen::MatrixXd colors(printing_path.size()-1, 3);

      for (int i = 0; i < printing_path.size() - 1 ; i ++) {
        Eigen::RowVector3d color = get_color(printing_path[i+1]->shrinkage);

        p1s.row(i) = printing_path[i]->pos;
        p2s.row(i) = printing_path[i+1]->pos;
        colors.row(i) = color;

//        viewer.data().add_edges(printing_path[i]->pos, printing_path[i+1]->pos, color);
      }

      viewer.data().add_edges(p1s, p2s, colors);
    }

    if (display_mode == 5) {  // mesh
      V_out.resize(nodes.size(), 3);
      F_out.resize(faces.size(), 3);

      for (auto n : nodes) {
        V_out.row(n->idx) << n->pos;
      }

      for (auto f : faces) {
        int i0 = f->halfedge->node->idx;
        int i1 = f->halfedge->next->node->idx;
        int i2 = f->halfedge->next->next->node->idx;
        F_out.row(f->idx) << i0, i1, i2;
      }

      viewer.data().set_mesh(V_out, F_out);
      FC_out.resize(F_out.rows(), 3);

      for (auto i = 0; i < faces.size(); i++) {
        Face* f = faces[i];
        Eigen::RowVector3d color;
        color << 0,0,0;
        for (int i_n = 0; i_n < 3; i_n++) {
          Node* n = f->node(i_n);
          color += n->color;
        }
        color /= 3;
        FC_out.row(i) = color;
      }

//      FC = FC * 0.8;
      viewer.data().set_colors(FC_out);
      igl::per_face_normals(V_out, F_out, N_out);  // might be redundent
    }

    if (display_mode == 6) {  // input mesh
      viewer.data().set_mesh(V_in, F_in);
      viewer.data().set_colors(FC_in);
      igl::per_face_normals(V_in, F_in, N_in);  // might be redundent
    }
  };

  auto subdivide_edge = [&](Node* n) {
    if (n->right) {
      Eigen::RowVector3d vec = n->right->pos - n->pos;
      int num_insert = floor(vec.norm() / iso_spacing); // number of inserted vertices on one iso line
      float t_step = 1.0 / (num_insert + 1);

      Node *n_prev = n;
      for (int i = 1; i <= num_insert; i++) {
        Node *n_new = new Node();
        n_new->idx = nodes.size();
        n_new->pos = n->pos + vec * t_step * i;
        n_new->pos_origin = n_new->pos;
        n_new->idx_iso = n->idx_iso;
        n_new->left = n_prev;
        n_new->right = n_prev->right;
        n_new->left->right = n_new;
        n_new->right->left = n_new;
        n_prev = n_new;
        nodes.push_back(n_new);
      }
    }
  };

  auto add_edge = [&](Node* n_a, Node* n_b, string type) {
    Edge* e = new Edge();

    e->nodes.emplace(n_a);
    e->nodes.emplace(n_b);

    for (auto ee : n_a->edges) {
      if (ee->nodes == e->nodes) {
        return ee;
      }
    }

    if (n_a == n_b) cout<<"n_a ==n_b: "<<type<<endl;

    e->spring = type;
    e->idx = edges.size();
    e->len_3d = e->length();
    e->len = e->len_3d;
    e->len_prev = 0.0000001;
    edges.push_back(e);

    n_a->edges.emplace(e);
    n_b->edges.emplace(e);

    return e;
  };

  auto find_the_other_face = [&](Node* n_a, Node* n_b, vector<Node*>* ns_triplet, vector<Edge*>* es_triplet) {

    Edge* e_c;
    for (auto e : edges) {
      if (e->nodes == set<Node*>{n_a, n_b}) {
        e_c = e;
        break;
      }
    }

    if (e_c->halfedge) {
      if (e_c->halfedge->twin) {
        return false;
      }
    }

    set<Node*> ns;
    int n_faces_found = 0;
    for (auto e_a : n_a->edges) {
      ns = e_a->nodes;
      ns.erase(n_a);
      Node* nn_a = *ns.begin();
      for (auto e_b : n_b->edges) {
        ns = e_b->nodes;
        ns.erase(n_b);
        Node* nn_b = *ns.begin();

        if (nn_a == nn_b) { // find a face, tri: n_a - n_b - nn_a/b
          if (e_a->halfedge and e_b->halfedge) { // face already exists
            Halfedge* h_a = e_a->halfedge;
            Halfedge* h_b = e_b->halfedge;
            if (h_a->face == h_b->face) {
              n_faces_found++;
              if (n_faces_found == 2) return false;
              continue;
            }
            else if (h_a->twin and h_a->twin->face == h_b->face) {
              n_faces_found++;
              if (n_faces_found == 2) return false;
              continue;
            }
            else if (h_b->twin and h_a->face == h_b->face) {
              n_faces_found++;
              if (n_faces_found == 2) return false;
              continue;
            }
            else if (h_b->twin and h_a->twin and h_a->twin->face == h_b->twin->face ) {
              n_faces_found++;
              if (n_faces_found == 2) return false;
              continue;
            }
          }

          (*ns_triplet)[0] = n_a;
          (*ns_triplet)[1] = n_b;
          (*ns_triplet)[2] = nn_a;
          (*es_triplet)[0] = e_c;
          (*es_triplet)[1] = e_b;
          (*es_triplet)[2] = e_a;
          return true;
        }
      }
    }

    return false;
  };

  function<bool (vector<Edge*>, vector<Node*>)> complete_face = [&](vector<Edge*> es, vector<Node*> ns) {
    Face* f = new Face();
    Halfedge* h_a = new Halfedge();
    Halfedge* h_b = new Halfedge();
    Halfedge* h_c = new Halfedge();

    f->idx = faces.size();
    faces.emplace_back(f);
    f->halfedge = h_a;
    f->is_saddle = false;
    for (auto n : ns) {
      if (n->is_saddle) f->is_saddle = true;
    }

    h_a->idx = halfedges.size();
    halfedges.emplace_back(h_a);
    h_a->node = ns[0];
    h_a->edge = es[0];
    h_a->face = f;
    h_a->prev = h_c;
    h_a->next = h_b;
    if (es[0]->halfedge) {
      h_a->twin = es[0]->halfedge;
      h_a->twin->twin = h_a;
    }

    h_b->idx = halfedges.size();
    halfedges.emplace_back(h_b);
    h_b->node = ns[1];
    h_b->edge = es[1];
    h_b->face = f;
    h_b->prev = h_a;
    h_b->next = h_c;
    if (es[1]->halfedge) {
      h_b->twin = es[1]->halfedge;
      h_b->twin->twin = h_b;
    }

    h_c->idx = halfedges.size();
    halfedges.emplace_back(h_c);
    h_c->node = ns[2];
    h_c->edge = es[2];
    h_c->face = f;
    h_c->prev = h_b;
    h_c->next = h_a;
    if (es[2]->halfedge) {
      h_c->twin = es[2]->halfedge;
      h_c->twin->twin = h_c;
    }

    es[0]->halfedge = h_a;
    es[1]->halfedge = h_b;
    es[2]->halfedge = h_c;

    ns[0]->halfedge = h_a;
    ns[1]->halfedge = h_b;
    ns[2]->halfedge = h_c;

    // search h_a->twin->f, h_b->twin->f, h_c->twin->f
    vector<Node*> ns_triplet = {nullptr, nullptr, nullptr};
    vector<Edge*> es_triplet = {nullptr, nullptr, nullptr};

    if (find_the_other_face(h_a->node, h_c->node, &ns_triplet, &es_triplet)) {
//      viewer.data().add_edges( (ns[0]->pos + ns[1]->pos + ns[2]->pos) / 3,
//                               (ns_triplet[0]->pos + ns_triplet[1]->pos + ns_triplet[2]->pos) / 3,
//                               color_red
//      );
      complete_face(es_triplet, ns_triplet);
    }
    if (find_the_other_face(h_c->node, h_b->node, &ns_triplet, &es_triplet)) {
//      viewer.data().add_edges( (ns[0]->pos + ns[1]->pos + ns[2]->pos) / 3,
//                               (ns_triplet[0]->pos + ns_triplet[1]->pos + ns_triplet[2]->pos) / 3,
//                               color_green
//      );
      complete_face(es_triplet, ns_triplet);
    }
    if (find_the_other_face(h_b->node, h_a->node, &ns_triplet, &es_triplet)) {
//      viewer.data().add_edges( (ns[0]->pos + ns[1]->pos + ns[2]->pos) / 3,
//                               (ns_triplet[0]->pos + ns_triplet[1]->pos + ns_triplet[2]->pos) / 3,
//                               color_blue
//      );
      complete_face(es_triplet, ns_triplet);
    }

    return true;
  };

  auto get_convergence = [&]() {
    double sum = 0;
    for (auto e : edges) {
      sum += abs(e->len - e->len_prev) / e->len_prev;
    }
    return ( sum / edges.size() );
  };

  auto get_halfedge = [&](Node* na, Node* nb) -> Halfedge* {
    Halfedge* h_iter = na->halfedge;
    do {
      if (h_iter->twin->node == nb) return h_iter;
      h_iter = h_iter->prev->twin;
    } while (h_iter != na->halfedge);
  };

  auto get_intersection = [&](Eigen::RowVector3d p1, Eigen::RowVector3d p2,
                              Eigen::RowVector3d v1, Eigen::RowVector3d v2) -> Eigen::RowVector3d {
    double den = v1.cross(v2).squaredNorm();
    Eigen::Matrix3d det;
    det.row(0) = p2 - p1;
    det.row(1) = v2;
    det.row(2) = v1.cross(v2);
    double t = det.determinant() / den;

    Eigen::RowVector3d intersect = p1 + t * v1;
    if (t < 0 or t > 1) {
      cout<<"not intersecting the segment"<<endl;
      cout<<t<<endl;
      viewer.data().add_points(intersect, color_red);
      getchar();
    }
    return intersect;
  };

  auto get_halfedges_of_node = [&](Node* n) -> vector<Halfedge*> {
    vector<Halfedge*> halfedges;
    Halfedge* h = n->halfedge;
    do {
      halfedges.push_back(h);
      h = h->twin->next;
    } while (h != n->halfedge);
    return halfedges;
  };

  auto get_halfedges_of_face = [&](Face* f) -> vector<Halfedge*> {
    vector<Halfedge*> halfedges;
    Halfedge* h = f->halfedge;
    do {
      halfedges.push_back(h);
      h = h->next;
    } while (h != f->halfedge);
    return halfedges;
  };

  auto fix_saddle = [&](Node* n_left, Node* n_right) {
    // reduce hole size
//    {
//      Node *n_left_new = n_left;
//      for (int i = 0; i < saddle_displacement; i++) {
//        n_left_new = n_left_new->right;
//      }
//      if (n_left_new->down) {
//        cout << "left_new->down" << endl;
//        getchar();
//      }
//      n_left->down->up = n_left_new;
//      n_left_new->down = n_left->down;
//      n_left->down = nullptr;
//      n_left = n_left_new;
//
//      Node *n_right_new = n_right;
//      for (int i = 0; i < saddle_displacement; i++) {
//        n_right_new = n_right_new->left;
//      }
//      if (n_right_new->down) {
//        cout << "right_new->down" << endl;
//        getchar();
//      }
//      n_right->down->up = n_right_new;
//      n_right_new->down = n_right->down;
//      n_right->down = nullptr;
//      n_right = n_right_new;
//    }

    double dist = (n_left->pos - n_right->pos).norm();
    Eigen::RowVector3d vec = n_right->pos - n_left->pos;
    vec.normalize();
    int num_nodes = int( dist / iso_spacing);
    for (int i = 0; i < num_nodes; i++) {
      Node* n_new = new Node();
      n_new->idx = nodes.size();
      n_new->pos = n_left->pos + vec * (dist / (num_nodes + 1)) * (i + 1);
      n_new->pos_origin = n_new->pos;
      n_new->idx_iso = n_left->idx_iso;
      n_new->idx_grad = -1;
      n_new->on_saddle_side = true;

      if (i == 0) {
        n_new->left = n_left;
        n_left->right_saddle = n_new;
        cout<<"n_left: "<<n_left->idx<<endl;
        cout<<"n_left->right_saddle: "<<n_left->right_saddle->idx<<endl;
      }
      else {
        n_new->left = nodes[nodes.size() - 1];
        n_new->left->right = n_new;
      }
      if (i == num_nodes - 1) {
        n_new->right = n_right;
        n_right->left_saddle = n_new;
      }
      nodes.push_back(n_new);
    }

  };

  auto visit = [&](Node* n_iter) {
    unvisited.erase(n_iter);
    n_iter->i_path = printing_path.size(); printing_path.push_back(n_iter);
  };

  auto get_links = [&](int num_interp_left, int num_interp_right)
                    -> vector<vector<int>> {
    // num_interp_left : # of interpolated nodes on the left
    if (num_interp_left % 2 != 0 or num_interp_right % 2 != 0) {
      cout<<"not even"<<endl; getchar();
    }

    set<int> unlinked_left;
    set<int> unlinked_right;
    vector<vector<int>> potential_links;
    vector<pair<int, double>> idx_dist;
    vector<vector<int>> links;

    num_interp_left /= 2;   // double gap size, insert odd number later
    num_interp_right /= 2;
    double gap_left = 1.0 / (num_interp_left + 1);
    double gap_right = 1.0 / (num_interp_right + 1);

    // init
    for (int i=0; i<num_interp_left; i++)   unlinked_left.insert(i);
    for (int i=0; i<num_interp_right; i++)  unlinked_right.insert(i);

    // (unlinked_left, unlinked_right) -> potential_links, idx_dist
    for (int i_left = 0; i_left < num_interp_left; i_left++) {
      double t_left = (i_left + 1) * gap_left;
      int i_right_prev = int(t_left / gap_right) - 1;
      int i_right_next = i_right_prev + 1;
      double t_right_prev = (i_right_prev + 1) * gap_right;
      double t_right_next = (i_right_next + 1) * gap_right;

      if (i_right_prev < 0 ) {
        // right_prev not exist
      }
      else {
        idx_dist.push_back(pair<int, double>({
          potential_links.size(),
          t_left - t_right_prev
        }));

        potential_links.push_back(vector<int>({i_left, i_right_prev}));
      }

      if (i_right_next > num_interp_right - 1) {
        // overflow
      }
      else {
        idx_dist.push_back(pair<int, double>({
          potential_links.size(),
          t_right_next - t_left
        }));

        potential_links.push_back(vector<int>({i_left, i_right_next}));
      }
    }

    // sort idx_dist
    sort(idx_dist.begin(), idx_dist.end(),
      [](const pair<int, double> a, const pair<int, double> b){
      return a.second < b.second;
    });

    // (idx_dist, potential_links, unlinked_left, unlinked_right) -> links
    for (pair<int,double> i_d : idx_dist) {
      int idx = i_d.first;
      vector<int> p = potential_links[idx];
      if (unlinked_left.count(p[0]) and unlinked_right.count(p[1])) { // both unlinked
        unlinked_left.erase(p[0]);
        unlinked_right.erase(p[1]);
        links.push_back(p);
      }
      else {
        // one of them is already linked
      }
    }

    // insert odd nodes
    vector<vector<int>> links_double;
    for (auto link : links) {
      links_double.push_back(vector<int>({link[0]*2, link[1]*2}));
      links_double.push_back(vector<int>({link[0]*2+1, link[1]*2+1}));
    }

    return links_double;
  };

  auto merge_edge = [&](Edge* e, Face* f, Face* f_left, Halfedge* h) {
    // update e->nodes_interp


    // (e, f, f_left) -> nodes_interp_left, nodes_interp_right
    vector<Node*> nodes_interp_left;
    vector<Node*> nodes_interp_right;
    {
      if (f_left->is_up) {
        nodes_interp_left = e->nodes_interp_as_right;
      } else {
        reverse(e->nodes_interp_as_right.begin(), e->nodes_interp_as_right.end());
        nodes_interp_left = e->nodes_interp_as_right;
        reverse(e->nodes_interp_as_right.begin(), e->nodes_interp_as_right.end());
      }

      if (f->is_up) {
        nodes_interp_right = e->nodes_interp_as_left;
      } else {
        reverse(e->nodes_interp_as_left.begin(), e->nodes_interp_as_left.end());
        nodes_interp_right = e->nodes_interp_as_left;
        reverse(e->nodes_interp_as_left.begin(), e->nodes_interp_as_left.end());
      }
    }

    vector<vector<int>> links = get_links(nodes_interp_left.size(), nodes_interp_right.size());

    // (nodes_interp_left, nodes_interp_right, links) -> e->nodes_interp
    {
      set<int> unvisited_left;
      set<int> unvisited_right;
      for (int i = 0; i < nodes_interp_left.size(); i++) unvisited_left.insert(i);
      for (int i = 0; i < nodes_interp_right.size(); i++) unvisited_right.insert(i);

      // left
      for (int i = 0; i < nodes_interp_left.size(); i++) {
        Node *n_mid = new Node();
        Node *n_left;
        Node *n_right;

        if (i < nodes_interp_left.size() and unvisited_left.count(i)) {   // left exists and unvisted
          n_left = nodes_interp_left[i];
          bool linked = false;
          int i_right = -1;
          for (auto link : links) {
            if (link[0] == i) {
              linked = true;
              i_right = link[1];
              break;
            }
          }
          if (linked) {
            n_right = nodes_interp_right[i_right];
            // link left i with right i_right
            n_mid->pos = (n_left->pos + n_right->pos) / 2.;
            n_mid->left = n_left->left;   // TODO: bug?
            n_mid->right = n_right->right;
            n_mid->left->right = n_mid;
            n_mid->right->left = n_mid;
            // visit left i and right i_right
            unvisited_left.erase(i);
            unvisited_right.erase(i_right);
          } else {
            // link left i
            n_mid->pos = n_left->pos;
            n_mid->left = n_left->left;
            n_mid->left->right = n_mid;
            n_mid->is_end = true;
            n_mid->is_right_end = true;
          }
          e->nodes_interp.push_back(n_mid);

          unvisited.emplace(n_mid); // TODO
          n_mid->i_unvisited_vector = unvisited_vector.size();  // TODO
          n_mid->is_interp_bridge = true; // TODO
          unvisited_vector.push_back(n_mid); // TODO
        }
      }

      // right
      for (int i = 0; i < nodes_interp_right.size(); i++) {
        Node *n_mid = new Node();
        Node *n_left;
        Node *n_right;

        if (i < nodes_interp_right.size() and unvisited_right.count(i)) {   // right exists and unvisted
          n_right = nodes_interp_right[i];
          bool linked = false;
          int i_left = -1;
          for (auto link : links) {
            if (link[1] == i) {
              linked = true;
              i_left = link[0];
              break;
            }
          }
          if (linked) {
            n_left = nodes_interp_right[i_left];
            // link right i with left i_left
            n_mid->pos = (n_left->pos + n_right->pos) / 2.;
            n_mid->left = n_left->left;   // TODO: bug?
            n_mid->right = n_right->right;
            n_mid->left->right = n_mid;
            n_mid->right->left = n_mid;
            // visit left i_left and right i
            unvisited_left.erase(i_left);
            unvisited_right.erase(i);
          } else {
            // link right i
            n_mid->pos = n_right->pos;
            n_mid->right = n_right->right;
            n_mid->right->left = n_mid;
            n_mid->is_end = true;
            n_mid->is_left_end = true;
          }
          e->nodes_interp.push_back(n_mid);

          unvisited.emplace(n_mid); // TODO
          n_mid->i_unvisited_vector = unvisited_vector.size();  // TODO
          n_mid->is_interp_bridge = true; // TODO
          unvisited_vector.push_back(n_mid); // TODO
        }
      }
    }

    // order e->nodes_interp
    {
      sort(e->nodes_interp.begin(), e->nodes_interp.end(),
           [&](const Node*  a, const Node*  b)->bool {
             double d_a = (a->pos - h->twin->node->pos).norm();
             double d_b = (b->pos - h->twin->node->pos).norm();
             return d_a > d_b;
           }
      );
    }
  };

  //////////////// menu funcs /////////////////

  auto initialize_param = [&]() {
    damping = 100.0;
    w_closeness = 1.0;
    w_stretch = 400.0;
    w_bridge = 400.0;
    w_bending = 10.0;
    w_angle_stretch = 100.0;
    w_angle_shear = 0.0;
    w_spreading = 0.0;
    w_flatten = 3.0;
    w_smooth = 0.1;
    gap_size = 0.48;
  };

  auto triangulate = [&]() {
    cout<<"triangulating......";

    for (auto n : nodes) {
//        {
//          Node* n = nodes[idx_focus];

      // detect quads (up right quad of each node)
      if (n->idx_grad != -1 and (n->right and n->up)) {  // original nodes in the graph, has up right
        Node *n_right = n->right;
        Node *n_up = n->up;
        Node *n_up_right = n_up->right;
        if (n_up->right_saddle) n_up_right = n_up->right_saddle;
        while (n_right->idx_grad == -1 or !n_right->up) {
          n_right = n_right->right;
        }
        while (n_up_right->idx_grad == -1 or !n_up_right->down) {
          if (n_up_right->left_saddle) break;
          n_up_right = n_up_right->right;
        }

        bool is_saddle = false;

        if (n_up->right_saddle) {  // not in a quad, in saddle
          is_saddle = true;

          // switch ->right and ->right_saddle
          Node* tmp;
          tmp = n_up->right;
          n_up->right = n_up->right_saddle;
          n_up->right_saddle = tmp;
        }


        Node *n_d = n;
        Node *n_u = n->up;
        bool right_most = false;
        // keep moving towards the right edge, insert edge every time
        while (true) {
          add_edge(n_d, n_u, "bridge");

          Node *n_c;
          if (n_u == n_up_right) {
            n_c = n_d->right;
            add_edge(n_d, n_c, "stretch");
            add_edge(n_u, n_c, "bridge");
            if (n_d->right == n_right) {
              right_most = true;
            } else {
              n_d = n_d->right;
            }
          } else if (n_d == n_right) {
            n_c = n_u->right;
            add_edge(n_c, n_u, "stretch");
            add_edge(n_c, n_d, "bridge");
            if (n_u->right == n_up_right) {
              right_most = true;
            } else {
              n_u = n_u->right;
            }
          } else {
            Eigen::RowVector3d vec_d = n_u->right->pos - n_d->pos;
            Eigen::RowVector3d vec_u = n_d->right->pos - n_u->pos;

            if ((vec_d.norm() < vec_u.norm()) or (n_d == n_right)) {  // connect the shorter diagonal
              n_c = n_u->right;
              add_edge(n_c, n_u, "stretch");
              add_edge(n_c, n_d, "bridge");
              n_u = n_u->right;
            } else {
              n_c = n_d->right;
              add_edge(n_c, n_u, "bridge");
              add_edge(n_c, n_d, "stretch");
              n_d = n_d->right;
            }
          }

          viewer.data().add_edges(n_u->pos, n_d->pos, Eigen::RowVector3d(0.9, 0, 0));

          if (right_most) {
            break;
          }
        }

        if (is_saddle) {
          // switch back
          Node* tmp = n->up->right;
          n->up->right = n->up->right_saddle;
          n->up->right_saddle = tmp;
        }

      }


      // top boundary
      if (!n->up and (not (n->on_saddle_boundary and (not (n->down))))) {   // avoid the case saddle connected with top
        bool in_boundary = false; // already in boundary
        for (auto b : boundaries_top) {
          for (auto n_b : b) {
            if (n == n_b) {
              in_boundary = true;
              break;
            }
          }
        }

        if (!in_boundary) {
          vector<Node *> boundary_top;
          bool is_boundary = true;
          Node *node_iter = n;
          cout<<"extract the boundary: "<<node_iter->idx<<endl;

          while (true) {  // extract the whole boundary
            if (node_iter->up) {
              is_boundary = false;
              boundary_top.clear();
              break;
            }
            boundary_top.push_back(node_iter);

            if (node_iter->right == n) {  // close the boundary
              break;
            }
            node_iter = node_iter->right;
          }

          if (is_boundary) {
            Eigen::RowVector3d center_pos = Eigen::RowVector3d(0, 0, 0);
            for (auto nb : boundary_top) {
              center_pos += nb->pos;
            }
            center_pos /= boundary_top.size();

            Node *n_center = new Node();
            n_center->idx = nodes.size();
            n_center->pos = center_pos;
            n_center->pos_origin = center_pos;
            n_center->idx_iso = -1; // TODO: indexing for inserted isoline
            n_center->idx_grad = -1;
            n_center->is_cone = true;
            nodes.push_back(n_center);
            cones.push_back(n_center);

            int i = 0;
            for (auto nb : boundary_top) {
              add_edge(nb, n_center, "bridge");

              Node* nb_next = boundary_top[(i + 1) % boundary_top.size()];
              add_edge(nb, nb_next, "stretch");   // fix the case saddle and top connected
              i++;
            }
          } else {
            // TODO: two iso lines sharing a hole but no node has a downward connection
          }

          boundaries_top.push_back(boundary_top);
        }
      }

      // bottom_boundary, only one
      if (!n->down and boundary_bottom.empty() ) {
        cout<<"detect boundary"<<endl;

        Node *n_iter = n;
        bool is_bottom_boundary = true;

        // check if closed bottom boundary
        do {
          if ((not n_iter->right) or
              n_iter->right_saddle or n_iter->left_saddle or
              n_iter->down) {
            is_bottom_boundary = false;
            break;
          }
          boundary_bottom.push_back(n_iter);
          n_iter = n_iter->right;
        } while (n_iter != n);

        if (is_bottom_boundary) {
          for (int i = 0; i < boundary_bottom.size(); i++) {
            int j = (i + 1) % boundary_bottom.size();
            Node *n_a = boundary_bottom[i];
            cout<<"boundary_node: "<<n_a->idx<<endl;
            Node *n_b = boundary_bottom[j];
            Edge *e = add_edge(n_a, n_b, "stretch");
            e->spring = "boundary";
//            if (e->idx != -1) e->spring = "boundary";
          }
        }
        else {
          boundary_bottom.clear();
        }

      }

      if (n->right_saddle and (not n->on_saddle_boundary) ) {
        cout<<"collect saddle"<<endl;
        vector<Node*> saddle_boundary;
        bool searching_side = true;
        Node* n_iter = n;

        do {
          n_iter->on_saddle_boundary = true;
          saddle_boundary.push_back(n_iter);

          if (n_iter->right_saddle) {
            searching_side = true;
          }
          if (n_iter->left_saddle) {
            searching_side = false;
          }

          if (n_iter->right_saddle) {
            n_iter->on_saddle_side = true;
            n_iter = n_iter->right_saddle;
          }
          else if (searching_side) {
            n_iter->on_saddle_side = true;
            n_iter = n_iter->right;
          }
          else {
            n_iter = n_iter->left;
          }

        } while (n_iter != n);


        Eigen::RowVector3d center_pos = Eigen::RowVector3d(0, 0, 0);

        for (auto n_iter : saddle_boundary) {
          center_pos += n_iter->pos;
        }
        center_pos /= saddle_boundary.size();

        Node *n_center = new Node();
        n_center->idx = nodes.size();
        n_center->pos = center_pos;
        n_center->pos_origin = center_pos;
        n_center->idx_iso = -1;
        n_center->idx_grad = -1;
        n_center->is_saddle = true;
        nodes.push_back(n_center);
        saddles.push_back(n_center);

        for (auto n_iter : saddle_boundary) {
          add_edge(n_iter, n_center, "bridge");
        }
      }


      if (!n->right or !n->left) { cerr << n->idx << " has no right node." << endl; }
    }

    cout<<"done."<<endl;

    display_mode = 1;   // show halfedge mesh
    redraw();
  };

  auto upsample = [&]() {
    cout<<"upsampling.....";
    for (auto n : nodes) {
      subdivide_edge(n);
    }
    cout<<"done."<<endl;
    triangulate();
    display_mode = 1;
    redraw();
  };

  auto halfedgize = [&]() {
    cout<<"halfedgizing......";
    vector<Node *> ns_triplet;
    vector<Edge *> es_triplet;
    // find first triangle
    for (Node *n_a : nodes) {
      Node *n_b = n_a->up;
      if (!n_b) continue;

      // prepare ns_triplet
      ns_triplet = {n_a, n_b};
      set<Node *> ns_tuple_0, ns_tuple_1, ns_tuple_2;
      ns_tuple_0 = {n_a, n_b};
      if (n_a->left) ns_tuple_1 = {n_b, n_a->left};
      else continue;
      if (n_b->left) ns_tuple_2 = {n_a, n_b->left};
      else continue;
      bool edge_exist = false;
      for (auto e : edges) {
        if (e->nodes == ns_tuple_1) {
          ns_triplet.emplace_back(n_a->left);
          ns_tuple_2 = {n_a, n_a->left};
          edge_exist = true;
          break;
        } else if (e->nodes == ns_tuple_2) {
          ns_triplet.emplace_back(n_b->left);
          ns_tuple_1 = {n_b, n_b->left};
          edge_exist = true;
          break;
        }
      }

      // prepare es_triplet
      es_triplet = {nullptr, nullptr, nullptr};
      for (auto e : edges) {
        if (e->nodes == ns_tuple_0) es_triplet[0] = e;
        if (e->nodes == ns_tuple_1) es_triplet[1] = e;
        if (e->nodes == ns_tuple_2) es_triplet[2] = e;
      }
      if (!edge_exist) continue;
    }

    // function: bfs_halfedge_mesh iterate through whole mesh
    // given triplets, output, complete a face
    complete_face(es_triplet, ns_triplet);

    // external face & boundary halfedges
    {
      Face *f = new Face(); // external face
      Edge *edge;
      Edge *edge_0;
      vector<Edge *> edges_boundary;

      // face
      f->idx = faces.size();
      f->is_external = true;
      faces.emplace_back(f);

      bool found_boundary = false;
      for (auto e : edges) {  // find the first boundary edge
        if (e->spring == "boundary") {
          edge = e;
          edge_0 = e;
          found_boundary = true;
          break;
        }
      }

      // external halfedges
      if (found_boundary) {
        do {  // collect boundary edges, create external halfedges
          edges_boundary.emplace_back(edge);

          Halfedge *h = new Halfedge();
          h->idx = halfedges.size();
          halfedges.emplace_back(h);
          h->edge = edge;
          h->face = f;
          f->halfedge = h;
          h->twin = edge->halfedge;
          h->twin->twin = h;

          // find next boundary edge
          Halfedge *hh = edge->halfedge;
          while (true) {
            hh = hh->next;
            if (hh->edge->spring == "boundary") {
              edge = hh->edge;
              break;
            }
            hh = hh->twin;
          }
        } while (edge != edge_0);

        // connect boundary halfedges
        int i = 0;
        for (auto e : edges_boundary) {
          Halfedge *h = e->halfedge->twin;
          int i_next = (i + 1) % edges_boundary.size();
          int i_prev = (i + edges_boundary.size() - 1) % edges_boundary.size();
          h->next = edges_boundary[i_prev]->halfedge->twin;
          h->prev = edges_boundary[i_next]->halfedge->twin;
          h->node = edges_boundary[i_next]->halfedge->node;
          i++;
        }

        for (auto e : edges) {
          e->rest_len = e->length();
        }
      }
      else {
        cout<<"boundary not found"<<endl;
      }
    }
    cout<<"done."<<endl;
    display_mode = 1;
//    redraw();
  };

  auto hit = [&](Node* n)->Eigen::RowVector3d {
    Eigen::RowVector3d s, dir;  // intersection ray
    s = n->pos;
    s.z() = -1E4;
    dir << 0, 0, 1E4;

    vector<igl::Hit> hits;
    igl::ray_mesh_intersect(s,dir, V_in, F_in, hits);
    if (hits.size() != 1) {
      cout<<"hit "<<hits.size()<<" idx: "<<n->idx<<endl; getchar();
      return Eigen::RowVector3d(1,0,0);
    }
    else {
      return FC_in.row(hits[0].id);
    }

  };

  auto texture_map = [&]() {
    for (auto n : nodes) {
      n->color = hit(n);
    }
  };

  auto step = [&]() {
    display_stress = true;

    cout<<"done."<<endl<<"Updating position......";
    { // flattening velocity
      double d_max = 0;
      for (auto n : nodes) if (abs(n->pos.z()) > d_max) d_max = abs(n->pos.z());

      double xy_max = 0;
      for (auto n : nodes) {
        if (abs(n->pos.x()) > xy_max) xy_max = abs(n->pos.x());
        if (abs(n->pos.y()) > xy_max) xy_max = abs(n->pos.y());
      }

      for (auto n : nodes) {
        float speed = ( w_flatten * exp(log(d_max / xy_max) * (1 / damping_flatten) ) ) * (n->pos.z() / d_max) ;
        n->velocity = Eigen::RowVector3d(0, 0, - speed);
        n->pos += n->velocity;
      }
    }


    cout<<"done."<<endl<<"Updating solver......";
    solver->reset();

    // setPoints
    {
      points.resize(3, nodes.size());
      for (auto n : nodes) {
        points(0, n->idx) = n->pos.x();
        points(1, n->idx) = n->pos.y();
        points(2, n->idx) = n->pos.z();
      }
      solver->setPoints(points);
    }

    // ClosenessConstraint (prevent drifting)
    for (auto n : nodes) {
      std::vector<int> id_vector;
      id_vector.push_back(n->idx);
      auto c = std::make_shared<ShapeOp::ClosenessConstraint>(id_vector, w_closeness, solver->getPoints());
      if (w_closeness > 0.) solver->addConstraint(c);
    }

    // EdgeStrainConstraint
    for (auto e : edges) {
      std::vector<int> id_vector;
      if (not e->halfedge) {
        cout<<"no halfedge"<<e->idx<<endl; getchar();
      }
      if (not e->halfedge->node) {
        cout<<"no halfedge->node"<<e->idx<<endl; getchar();
      }
      if (not e->halfedge->twin) {
        cout<<"no twin"<<e->idx<<endl; cout<<" "<<e->halfedge->node->idx<<endl; getchar();
      }
      if (not e->halfedge->twin->node) {
        cout<<"no twin node"<<e->idx<<endl; cout<<" "<<e->halfedge->node->idx<<endl; getchar();
      }

      id_vector.push_back(e->halfedge->node->idx);
      id_vector.push_back(e->halfedge->twin->node->idx);
      auto weight = w_stretch;
      auto c = std::make_shared<ShapeOp::EdgeStrainConstraint>(id_vector, weight, solver->getPoints(), 1, 1.5);
      if (e->spring == "bridge") {
        weight = w_bridge;
        c = std::make_shared<ShapeOp::EdgeStrainConstraint>(id_vector, weight, solver->getPoints());
      }
      double rest_len =  e->rest_len;
      c->setEdgeLength(rest_len);

      if (weight > 0) {
        solver->addConstraint(c);
      }
    }

    // BendingConstraint
    Eigen::MatrixXd points_flat;
    points_flat.resize(points.rows(), points.cols());
    for (auto e : edges) {
      if (e->spring == "boundary") continue;

      std::vector<int> id_vector;
      int i_0 = e->halfedge->node->idx;
      int i_1 = e->halfedge->twin->node->idx;
      int i_2 = e->halfedge->next->next->node->idx;
      int i_3 = e->halfedge->twin->next->next->node->idx;

      Eigen::RowVector3d v01 = nodes[i_1]->pos - nodes[i_0]->pos;
      Eigen::RowVector3d v02 = nodes[i_2]->pos - nodes[i_0]->pos;
      Eigen::RowVector3d v03 = nodes[i_3]->pos - nodes[i_0]->pos;

      double theta = acos( v01.dot(v02) / (v01.norm() * v02.norm() ) );
      double phi   = acos( v01.dot(v03) / (v01.norm() * v03.norm() ) );

      // positions of 4 flattened nodes
      Eigen::RowVector3d p_0(0, 0, 0);
      Eigen::RowVector3d p_1(v01.norm(), 0, 0);
      Eigen::RowVector3d p_2(cos(theta) * v02.norm(), sin(theta) * v02.norm(), 0);
      Eigen::RowVector3d p_3(cos(phi) * v03.norm(), -sin(phi) * v03.norm(), 0);

      id_vector.push_back(i_0); id_vector.push_back(i_1); id_vector.push_back(i_2); id_vector.push_back(i_3);   // bug??
      points_flat.col(i_0) = p_0;
      points_flat.col(i_1) = p_1;
      points_flat.col(i_2) = p_2;
      points_flat.col(i_3) = p_3;

      auto c = std::make_shared<ShapeOp::BendingConstraint>(id_vector, w_bending, points_flat);
      if (w_bending > 0) {
        solver->addConstraint(c);
      }
    }

    // AngleConstraint for stretch springs
    {
      for (auto n : nodes) {
        std::vector<int> id_vector;
        if (n->idx_iso != -1) { // not the center point nor the saddle point
          std::vector<int> id_vector;
          id_vector.push_back(n->idx);  // vertex 0

          Halfedge *h0 = n->halfedge;
          Halfedge *h = n->halfedge;
          do {
            if (h->twin->node->idx_iso == n->idx_iso) {
              id_vector.push_back(h->twin->node->idx);  // vertex 1 & 2
            }
            h = h->twin->next;
          } while (h != h0);
//          if (id_vector.size() != 3) cout << "WARNING: not 3" << endl;

          auto c = std::make_shared<ShapeOp::AngleConstraint>(id_vector, w_angle_stretch, points, 0, 0);
          if (w_angle_stretch > 0) solver->addConstraint(c);
        }
      }
    }

    // AngleConstraint for shearing (negative effect on flattening)
    for (auto f : faces) {
      int i0, i1, i2;
      Halfedge* h = f->halfedge;
      std::vector<int> id_vector;
      i0 = h->node->idx;
      i1 = h->next->node->idx;
      i2 = h->prev->node->idx;
      id_vector.push_back(i0); id_vector.push_back(i1); id_vector.push_back(i2);
      auto c = std::make_shared<ShapeOp::AngleConstraint>(id_vector, w_angle_shear, solver->getPoints());
      if (w_angle_shear > 0) solver->addConstraint(c);

      h = h->next;
      id_vector.clear();
      i0 = h->node->idx;
      i1 = h->next->node->idx;
      i2 = h->prev->node->idx;
      id_vector.push_back(i0); id_vector.push_back(i1); id_vector.push_back(i2);
      c = std::make_shared<ShapeOp::AngleConstraint>(id_vector, w_angle_shear, solver->getPoints());
      if (w_angle_shear > 0) solver->addConstraint(c);

      h = h->next;
      id_vector.clear();
      i0 = h->node->idx;
      i1 = h->next->node->idx;
      i2 = h->prev->node->idx;
      id_vector.push_back(i0); id_vector.push_back(i1); id_vector.push_back(i2);
      c = std::make_shared<ShapeOp::AngleConstraint>(id_vector, w_angle_shear, solver->getPoints());
      if (w_angle_shear > 0) solver->addConstraint(c);
    }

    // PlaneConstraint
    {
      std::vector<int> id_vector;
      for (auto n : nodes) {
        id_vector.push_back(n->idx);
      }
      auto c = std::make_shared<ShapeOp::PlaneConstraint>(id_vector, w_flatten, solver->getPoints());
//      if (w_flatten > 0) solver->addConstraint(c);
    }

    // flattening force (cannot see any effect?)
    {
      for (auto n : nodes) {
        double d = n->pos.z();  // distance to the target plane
        double m = d * w_flatten;
        Eigen::RowVector3d force(0., 0., -m);
        auto f = std::make_shared<ShapeOp::VertexForce>(force, n->idx);
//        if (w_flatten > 0.) solver->addForces(f);
      }
    }

    // spreading force
    Eigen::RowVector3d n0(0, 0, 1.0);
    for (auto e : edges) {
      // TODO: test and debug spreading force
      Eigen::RowVector3d n1 = e->halfedge->face->normal();
      Eigen::RowVector3d n2 = e->halfedge->twin->face->normal();
      Eigen::RowVector3d nj1 = n1 - n1.dot(n0) * n0;
      Eigen::RowVector3d nj2 = n2 - n2.dot(n0) * n0;
      Eigen::RowVector3d Fj1 = nj1 * w_spreading;
      Eigen::RowVector3d Fj2 = nj2 * w_spreading;
      int i1 = e->halfedge->prev->node->idx;
      int i2 = e->halfedge->twin->prev->node->idx;

      auto f = std::make_shared<ShapeOp::VertexForce>(Fj1 - Fj2, i1);
      if (w_spreading > 0.) solver->addForces(f);
      f = std::make_shared<ShapeOp::VertexForce>(Fj2 - Fj1, i2);
      if (w_spreading > 0.) solver->addForces(f);
    }

    cout<<"done."<<endl<<"Solving......";

    solver->initialize();
    solver->setDamping(damping);
//    solver->setTimeStep(time_step);
    solver->solve(num_iter);

    points = solver->getPoints();
    for (auto n : nodes) {
      n->pos = points.col(n->idx);
    }
    for (auto e : edges) {
      e->len_prev = e->len;
      e->len = e->length();
    }

    cout<<"done."<<endl;
  };

  //////////////////////////////////// init ////////////////////////////////////
  // load graph
  {
    string in_file_name;
    if (argv[1]) in_file_name = argv[1];
    else cerr << "arg[1] is required." << endl;

    ifstream ifile(in_file_name);

    string line;
    while (getline(ifile, line, '\n')) {
      auto n = new Node();
      nodes.push_back(n);
    }

    ifile.clear();
    ifile.seekg(0, ios::beg);

    while (getline(ifile, line, '\n')) {
      vector<string> items;
      boost::split(items, line, boost::is_any_of(" "), boost::token_compress_on);
      if (items[0] == "#") {
        nodes.resize(nodes.size()-1);
        if (items[1] == "center") {
          center.resize(1, 3);
          center << stod(items[2]), stod(items[3]), stod(items[4]);
        }
      }
      else {
        int idx = stoi(items[0]);
        Node *n = nodes[idx];
        n->idx = stoi(items[0]); // -1: not defined; -1: inserted
        n->pos = Eigen::RowVector3d(stof(items[1]), stof(items[2]), stof(items[3]));
        n->pos_origin = Eigen::RowVector3d(stof(items[1]), stof(items[2]), stof(items[3]));
        if (items[4] != "-1") n->idx_iso = stoi(items[4]);
        if (items[5] != "-1") n->idx_grad = stoi(items[5]);
        if (items[6] != "-1") n->right = nodes[stoi(items[6])];
        if (items[7] != "-1") n->left = nodes[stoi(items[7])];
        if (items[8] != "-1") n->up = nodes[stoi(items[8])];
        if (items[9] != "-1") n->down = nodes[stoi(items[9])];
      }
    }
  }

  // load image
  if (argv[3]) {
    image_in = stbi_load(argv[3], &w_in, &h_in, &n_c_in, 0);
  }

  // load mesh
  if (argv[2]) {
    cout<<"reading OBJ...."<<endl;
    igl::readOBJ(argv[2],V_in, TC_in, N_in, F_in, FTC_in, FN_in);

    FC_in.resize(F_in.rows(), 3);
    for (int i = 0; i < F_in.rows(); i++) {
      int iv0 = FTC_in.row(i)[0];
      int iv1 = FTC_in.row(i)[1];
      int iv2 = FTC_in.row(i)[2];

      Eigen::RowVector2d pos_tc0 = TC_in.row(iv0);
      Eigen::RowVector2d pos_tc1 = TC_in.row(iv1);
      Eigen::RowVector2d pos_tc2 = TC_in.row(iv2);

      Eigen::RowVector3d c0 = get_pixel(pos_tc0[0], pos_tc0[1]);
      Eigen::RowVector3d c1 = get_pixel(pos_tc1[0], pos_tc1[1]);
      Eigen::RowVector3d c2 = get_pixel(pos_tc2[0], pos_tc2[1]);

      Eigen::RowVector3d cf = (c0 + c1 + c2) / 3;
      FC_in.row(i) = cf;
    }
    cout<<"V.size: "<<V_in.rows()<<" "<<V_in.cols()<<endl;
    cout<<"TC.size: "<<TC_in.rows()<<" "<<TC_in.cols()<<endl;
    cout<<"N.size: "<<N_in.rows()<<" "<<N_in.cols()<<endl;
    cout<<"F.size: "<<F_in.rows()<<" "<<F_in.cols()<<endl<<endl;
    cout<<"FTC.size: "<<FTC_in.rows()<<" "<<FTC_in.cols()<<endl<<endl;
    cout<<"FN.size: "<<FN_in.rows()<<" "<<FN_in.cols()<<endl<<endl;
    cout<<"done."<<endl;

    // recenter
    V_in.rowwise() -= center;
  }

  // compute avg_len
  {
    float sum_len_iso_seg = 0;
    float sum_len_grad_seg = 0;
    int n_iso_seg = 0;
    int n_grad_seg = 0;
    for (auto n : nodes ) {
      if (n->right) sum_len_iso_seg += (n->pos - n->right->pos).norm();
      if (n->up) sum_len_grad_seg += (n->pos - n->up->pos).norm();
      n_iso_seg ++; n_grad_seg++;
    }
  }

  { // visualizer
    viewer.core.background_color = Eigen::Vector4f(0., 0., 0., 1.);
    viewer.core.camera_base_zoom = 0.01;
    viewer.plugins.push_back(&menu);

    redraw();
  }

  // callbacks
  menu.callback_draw_viewer_menu = [&]()
  {
    if (ImGui::CollapsingHeader("test window", ImGuiTreeNodeFlags_DefaultOpen)) {

    }

    if (ImGui::CollapsingHeader("params", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::InputFloat("isoSpacing", &iso_spacing);
      ImGui::InputFloat("closeness", &w_closeness);
      ImGui::InputFloat("stretch", &w_stretch);
      ImGui::InputFloat("bridge", &w_bridge);
      ImGui::InputFloat("bending", &w_bending);
      ImGui::InputFloat("flatten", &w_flatten);
      ImGui::InputFloat("angleStretch", &w_angle_stretch);
      ImGui::InputFloat("angleShear", &w_angle_shear);
      ImGui::InputFloat("spreading", &w_spreading);
      ImGui::InputFloat("damping", &damping);
      ImGui::InputFloat("dampingSmooth", &damping_flatten);
//      ImGui::InputFloat("timeStep", &time_step);
      ImGui::InputInt("numIter", &num_iter);
      ImGui::InputFloat("smoothing", &w_smooth);
      ImGui::InputFloat("gapSize", &gap_size);
      ImGui::InputInt("maxGap", &filter_threshold);
      ImGui::InputInt("saddle_displacement", &saddle_displacement);
      ImGui::InputInt("resolution", &w_out);
      ImGui::InputFloat("shrinkage_cone", &shrinkage_cone);
      ImGui::InputFloat("shrinkage_cone_down", &shrinkage_cone_down);
      ImGui::InputFloat("radius_cone", &radius_cone);
      ImGui::InputFloat("radius_trim", &radius_trim);
    }

    if (ImGui::CollapsingHeader("display", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::InputInt("index", &idx_focus)) {
        redraw();
      }

      ImGui::Text("displayMode");

      if (ImGui::RadioButton("graph", &display_mode, 0) or
          ImGui::RadioButton("halfedge", &display_mode, 1) or
          ImGui::RadioButton("segments", &display_mode, 3) or
          ImGui::RadioButton("tracing", &display_mode, 4) or
          ImGui::RadioButton("connection", &display_mode, 2) or
          ImGui::RadioButton("mesh", &display_mode, 5) or
          ImGui::RadioButton("input_mesh", &display_mode, 6)
        ) {
        redraw();
      }

      ImGui::Spacing();

      if (display_mode == 1) {
        if (ImGui::Checkbox("stressField", &display_stress) ) {
          redraw();
        }
      }

      if (ImGui::Checkbox("bridgeConstraint", &display_bridge)) {
        redraw();
      }

      if (ImGui::Checkbox("nightMode", &background_black) ) {
        if (background_black) {
          viewer.core.background_color = Eigen::Vector4f(0.0, 0.0, 0.0, 1.0);
        }
        else {
          viewer.core.background_color = Eigen::Vector4f(1.0, 1.0, 1.0, 1.0);
        }
      }

      if (ImGui::Checkbox("label", &display_label)) {
        redraw();
      }
      if (display_label) {
        ImGui::Text("label");
        if (ImGui::RadioButton("node", &label_type, 0) or
            ImGui::RadioButton("iso", &label_type, 1) or
            ImGui::RadioButton("grad", &label_type, 2) or
            ImGui::RadioButton("num_interp", &label_type, 3) or
            ImGui::RadioButton("is_end", &label_type, 4)
          ) {
          redraw();
        }
      }

      if (ImGui::Button("redraw")) {
        redraw();
      }
    }

    if (ImGui::CollapsingHeader("tools", ImGuiTreeNodeFlags_DefaultOpen)) {

      if (ImGui::Button("reset")) {
        for (auto n : nodes) n->pos = n->pos_origin;
        initialize_param();
        redraw();
      }

      if (ImGui::Button("reorder_iso")) {

        while (true) {
          bool is_ordered = true;
          for (auto n : nodes) {
            if (n->idx == -1) continue;

            if (n->down) {
              if (n->down->idx_iso < n->idx_iso - 1) {
                is_ordered = false;
                Node *n_iter = n;
                do {
                  n_iter->idx_iso = n->down->idx_iso + 1;

                  if (n_iter->right) n_iter = n_iter->right;
                  else {
                    cout << "n_iter->right does not exist" << endl;
                    getchar();
                  }
                } while (n_iter != n);
              }
            }
          }
          if (is_ordered) break;
        }

      }


      if (ImGui::Button("saddle")) {
        for (auto n : nodes) {
          if (n->idx_grad != -1 and (n->right and n->up)) {
            Node *n_right = n->right;
            Node *n_up = n->up;
            Node *n_up_right = n_up->right;
            while (n_right->idx_grad == -1 or !n_right->up) {
              n_right = n_right->right;
            }
            while (n_up_right->idx_grad == -1 or !n_up_right->down) {
              n_up_right = n_up_right->right;
            }

            if (n_right->up != n_up_right) { // saddle
              fix_saddle(n_up, n_right->up);
            }
          }
        }
      }

      if (ImGui::Button("upsample")) {
        upsample();
      }

      if (ImGui::Button("subdivide")) {
        for (auto n : nodes) {
          subdivide_edge(n);
        }
      }

      if (ImGui::Button("triangulate")) {
        triangulate();
      }

      if (ImGui::Button("halfedgize")) {
        cout<<"clicked halfedgize"<<endl;
        halfedgize();
      }

      if (ImGui::Button("map")) {
        texture_map();
      }


      ImGui::Text("Convergence: ", to_string(get_convergence()).c_str());

      if (ImGui::Button("step")) {
        for (int i = 0; i < num_iter; i++) {
//          halfedgize();
          step();
          cout<< float(i + 1) / num_iter <<endl;
        }
        redraw();
      }

      if (ImGui::Button("begin")) {
        is_flattening = true;
      }

      if (ImGui::Button("pause")) {
        is_flattening = false;
      }

      if (ImGui::Button("smooth")) {
        for (auto n : nodes) {
          if ((not n->left) or (not n->right)) continue;
          if (n->idx_iso == -1) continue;
          if (not (n->left->idx_iso == n->right->idx_iso and n->right->idx_iso == n->idx_iso)) {
            cout<<"error: left, right, iso different, node->idx "<<n->idx<<endl;
          }
          Edge* el; Edge* er;
          Halfedge* h;  Halfedge* h0;
          int count = 0;

          h0 = n->halfedge;
          h = n->halfedge;
          do {
            if (h->twin->node == n->left) el = h->edge;   // left edge
            if (h->twin->node == n->right) er = h->edge;  // right edge

            h = h->twin->next;
            count++;
            if (count > 20) {
              cout<<"error: n: "<<n->idx<<endl;
              break;
            }
          } while (h != h0);

          Eigen::RowVector3d v_l2r = n->right->pos - n->left->pos;
          float t = er->shrinkage - el->shrinkage;
          n->pos += v_l2r * t * w_smooth;
        }

        redraw();
      }

      if (ImGui::Button("scale")) {
        double x_max = -99999;
        double x_min = 99999;
        double y_max = -99999;
        double y_min = 99999;
        double z_max = -99999;
        double z_min = 99999;
        double x_span, y_span, z_span;

        for (auto n : nodes) {
          if (n->pos.x() > x_max) x_max = n->pos.x();
          if (n->pos.x() < x_min) x_min = n->pos.x();
          if (n->pos.y() > y_max) y_max = n->pos.y();
          if (n->pos.y() < y_min) y_min = n->pos.y();
          if (n->pos.z() > z_max) z_max = n->pos.z();
          if (n->pos.z() < z_min) z_min = n->pos.z();
        }
        x_span = x_max - x_min;
        y_span = y_max - y_min;
        z_span = z_max - z_min;
        cout<<"span: "<<x_span<<" "<<y_span<<endl;
        scale_ratio = min( (platform_length - 1) / x_span, (platform_width - 1) / y_span );
        Eigen::RowVector3d translation = Eigen::RowVector3d ((x_max + x_min) / 2, (y_max + y_min) / 2, 0);
        for (auto n : nodes) {
          n->pos -= translation;
          n->pos *= scale_ratio;
        }
      }

      if (ImGui::Button("interpolate")) {
        int i_iso_max = 0;
        for (auto n : nodes) {
          if (n->idx_iso > i_iso_max) i_iso_max = n->idx_iso;
        }

        // isos_node
        for (int iso_iter = 0; iso_iter <= i_iso_max; iso_iter++ ) {
          vector<vector<Node*>> loops_node;
          while (true) {
            vector<Node*> loop_node;

            bool found = false;
            Node* n_iter = nullptr;
            for (auto n : nodes) {
              if (n->idx_iso == iso_iter and (not n->visited_interpolation)
                  and (not n->on_saddle_boundary) ) {
                n_iter = n;   // begin with any unvisited node in any loop of the iso
                found = true;
                break;
              }
            }
            if (not found) break;

            Node* n_begin = n_iter;
            do {
              n_iter->visited_interpolation = true;
              loop_node.push_back(n_iter);
              if (n_iter->left) n_iter = n_iter->left;
              else cout<<"err: n_iter->left cannot be found";
            } while (n_iter != n_begin);
            loops_node.push_back(loop_node);
          }
          isos_node.push_back(loops_node);
        }

        // isos_face
        for (int iso_iter = 0; iso_iter <= i_iso_max; iso_iter++ ) {
          vector<vector<Face*>> loops_face;

          while (true) {
            vector<Face *> loop_face;
            Face *f_iter;

            // find a random face of an iso value
            bool found = false;
            for (auto f : faces) {
              Halfedge *h_iter = f->halfedge;
              bool is_saddle = false;
              do {
                if (h_iter->node->is_saddle) is_saddle = true;
                h_iter = h_iter->next;
              } while (h_iter != f->halfedge);
              if (is_saddle) {
                f->visited_interpolation = true;
                continue;
              }

              bool connect_down_iso = false;
              bool connect_up_iso = false;
              h_iter = f->halfedge;
              do {
                if (h_iter->node->idx_iso == iso_iter)
                  connect_down_iso = true;
                if (h_iter->node->idx_iso == iso_iter + 1 or h_iter->node->idx_iso == -1)
                  connect_up_iso = true;
                h_iter = h_iter->next;
              } while (h_iter != f->halfedge);

              if (connect_down_iso and connect_up_iso and (not f->visited_interpolation)) {
                f_iter = f;
                found = true;
                break;
              }
            }

            if (not found) break;

            Face *f_begin = f_iter;
            do {
              f_iter->visited_interpolation =true;
              loop_face.push_back(f_iter);

              Halfedge *h_iter = f_iter->halfedge;
              while (true) {
                if (((h_iter->node->idx_iso == iso_iter + 1) or (h_iter->node->is_cone))
                    and h_iter->next->node->idx_iso == iso_iter) { // left edge
                  f_iter = h_iter->twin->face;
                  break;
                }
                h_iter = h_iter->next;
              }

            } while (f_iter != f_begin);

            loops_face.push_back(loop_face);
          }
          iso_faces.push_back(loops_face);
        }

        // compute num_interp
        for (int i_iso = 0; i_iso <= i_iso_max; i_iso++) {
          for (auto iso_faces_loop : iso_faces[i_iso]) {
            for (auto f : iso_faces_loop) {
              // assign face values
              Halfedge *h_iter = f->halfedge;
              do {
                if (h_iter->node->idx_iso == h_iter->next->node->idx_iso) { // edge on isoline
                  if (h_iter->node->idx_iso == i_iso) { // up triangle
                    f->e_stretch = h_iter->edge;
                    f->e_bridge_left = h_iter->prev->edge;
                    f->e_bridge_right = h_iter->next->edge;
                    f->n_bridge = h_iter->next->next->node;
                    f->n_stretch_left = h_iter->node;
                    f->n_stretch_right = h_iter->next->node;
                    f->pos_stretch_mid = f->e_stretch->centroid();
                    f->is_up = true;
                  } else {  // down triangle
                    f->e_stretch = h_iter->edge;
                    f->e_bridge_left = h_iter->next->edge;
                    f->e_bridge_right = h_iter->prev->edge;
                    f->n_bridge = h_iter->next->next->node;
                    f->n_stretch_left = h_iter->next->node;
                    f->n_stretch_right = h_iter->node;
                    f->pos_stretch_mid = f->e_stretch->centroid();
                    f->is_up = false;
                  }
                  break;
                }
                h_iter = h_iter->next;
              } while (h_iter != f->halfedge);

              Eigen::RowVector3d vec_increment;
              double len_bridge = 0; double len_stretch = 0;
              Node *n_iter = f->n_bridge;
              // vec_bridge
              {
                if (n_iter->is_cone) {
                  f->vec_bridge = Eigen::RowVector3d(0, 0, 0);
                }
                else {
                  do {  // add left
                    vec_increment = get_halfedge(n_iter->left, n_iter)->vector();
                    f->vec_bridge += vec_increment;
                    len_bridge += vec_increment.norm();
                    if (n_iter->left) n_iter = n_iter->left;
                    else {
                      cout << "warning: no_left" << endl;
                      break;
                    }
                  } while (len_bridge < (f->n_bridge->pos - f->pos_stretch_mid).norm() / 2);
                  n_iter = f->n_bridge;
                  do {  //add right
                    vec_increment = get_halfedge(n_iter, n_iter->right)->vector();
                    f->vec_bridge += vec_increment;
                    len_bridge += vec_increment.norm();
                    if (n_iter->right) n_iter = n_iter->right;
                    else {
                      cout << "warning: no_right" << endl;
                      break;
                    }
                  } while (len_bridge < (f->n_bridge->pos - f->pos_stretch_mid).norm());
                  f->vec_bridge.normalize();
                }
              }

              // vec_stretch
              {
                n_iter = f->n_stretch_left;
                f->vec_stretch += get_halfedge(f->n_stretch_left, f->n_stretch_right)->vector();
                do {
                  vec_increment = get_halfedge(n_iter->left, n_iter)->vector();
                  f->vec_stretch += vec_increment;
                  len_stretch += vec_increment.norm();
                  if (n_iter->left) n_iter = n_iter->left;
                  else {
                    cout << "warning: no_left" << endl;
                    break;
                  }
                } while (len_stretch < (f->n_bridge->pos - f->pos_stretch_mid).norm() / 2);
                n_iter = f->n_stretch_right;
                do {
                  vec_increment = get_halfedge(n_iter, n_iter->right)->vector();
                  f->vec_stretch += vec_increment;
                  len_stretch += vec_increment.norm();
                  if (n_iter->right) n_iter = n_iter->right;
                  else {
                    cout << "warning: no_right" << endl;
                    break;
                  }
                } while (len_stretch < (f->n_bridge->pos - f->pos_stretch_mid).norm());
                f->vec_stretch.normalize();
              }

              // compute distance & num_interp
              {
                Eigen::RowVector3d normal = f->normal();
                Eigen::RowVector3d vec_stretch_projected = f->vec_stretch - f->vec_stretch.dot(normal) * normal;
                Eigen::RowVector3d vec_bridge_projected = f->vec_bridge - f->vec_bridge.dot(normal) * normal;

                Eigen::RowVector3d vec_parallel = vec_stretch_projected + vec_bridge_projected;
                vec_parallel.normalize();
                // Eigen::RowVector3d vec_perpendicular = normal.cross(vec_parallel);
                Eigen::RowVector3d vec_perpendicular = normal.cross(f->e_stretch->halfedge->vector());
                vec_perpendicular.normalize();

                Eigen::RowVector3d vec_mid = f->pos_stretch_mid - f->n_bridge->pos;
                 double iso_distance = abs(vec_mid.dot(vec_perpendicular));
                f->num_interp = int(iso_distance / (2 * gap_size) );  // double gap_size first, double num after smoothing

//                if (f->n_bridge->is_cone) f->num_interp = 0;  // TODO
              }
            }

            // smooth  num_interp, difference between neighbor faces within 2
            {
              // smoothing num_interp
              vector<int> num_interps;
              vector<int> num_interps_new;
              for (auto f : iso_faces_loop) {
                num_interps.push_back(f->num_interp);
              }

              int times = 0;
              while (true) {
                bool smooth = true;
                num_interps_new = num_interps;

                for (int i = 0; i < num_interps.size(); i++) {
                  int i_next = (num_interps.size() + i + 1) % num_interps.size();
                  int diff = abs(num_interps[i_next] - num_interps[i]);
                  if (diff > min(num_interps[i_next], num_interps[i])) {  // TODO: param
                    smooth = false;
                    if (diff == 1) {
                      num_interps_new[i] = num_interps[i];
                      num_interps_new[i_next] = num_interps[i];
                    } else {
                      int delta_1 = int((diff - 2) / 2);
                      int delta_2 = diff - 2 - delta_1;
                      if (num_interps[i_next] > num_interps[i]) {
                        num_interps_new[i_next] = num_interps[i_next] - delta_1;
                        num_interps_new[i] = num_interps[i] + delta_2;
                      } else {
                        num_interps_new[i_next] = num_interps[i_next] + delta_2;
                        num_interps_new[i] = num_interps[i] - delta_1;
                      }
                    }
                  }
                }
                if (smooth) break;
                if (times++ > 5) {
                  break;
                }
                num_interps = num_interps_new;
              }

              int i = 0;
              for (auto f : iso_faces_loop) {
                f->num_interp = num_interps_new[i] * 2;
                i++;
              }
            }


          }
        }

        // filter short gaps
        for (int i_filter = 0; i_filter < 3; i_filter++) {
          for (int i_iso = 0; i_iso <= i_iso_max; i_iso++) {  // for each iso value
            for (auto iso_faces_loop : iso_faces[i_iso]) {    // for each loop with same iso
              for (int gap_size = 1; gap_size <= filter_threshold; gap_size++) {    // increase window size
                for (int i_begin = 0; i_begin < iso_faces_loop.size(); i_begin++) {    // for each face on the loop
                  vector<Face *> window;
                  int i_end = i_begin + gap_size + 2 - 1;
                  int i = i_begin;
                  do {
                    int i_push = i % iso_faces_loop.size();
                    window.push_back(iso_faces_loop[i_push]);
                    i++;
                  } while (i <= i_end);

                  bool should_filter = true;
                  if (not(window[0]->num_interp == window[window.size() - 1]->num_interp)) should_filter = false;
                  for (int i_window = 1; i_window < window.size() - 1; i_window++) {
                    if (window[i_window]->num_interp != window[1]->num_interp) should_filter = false;
                  }

                  if (should_filter) {
                    for (auto f : window) {
                      f->num_interp = window[0]->num_interp;
                    }
                  }
                }
              }
            }
          }
        }

        // interpolate
        for (int i_iso = 0; i_iso <= i_iso_max; i_iso++) {
          for (auto iso_faces_loop : iso_faces[i_iso]) {
            for (auto f : iso_faces_loop) {

              // interpolate nodes
              for (int i_interp = 0; i_interp < f->num_interp; i_interp++) {
                double weight_bridge = float(i_interp + 1) / (f->num_interp + 1);
                double weight_stretch = 1.0 - weight_bridge;

                // fix
                // Eigen::RowVector3d vec_interp = weight_bridge * f->vec_bridge + weight_stretch * f->vec_stretch;
                Eigen::RowVector3d vec_interp = f->n_stretch_left->pos - f->n_stretch_right->pos;
                vec_interp.normalize();

                Eigen::RowVector3d pos_interp = weight_bridge * f->n_bridge->pos + weight_stretch * f->pos_stretch_mid;
                f->pos_interps.push_back(pos_interp);

                Eigen::RowVector3d vec_left = f->e_bridge_left->halfedge->vector();
                if (f->e_bridge_left->halfedge->node != f->n_stretch_left) {
                  vec_left = f->e_bridge_left->halfedge->twin->vector();
                }
                Eigen::RowVector3d pos_left = get_intersection(f->n_stretch_left->pos, pos_interp,
                                                               vec_left, vec_interp);

                Eigen::RowVector3d vec_right = f->e_bridge_right->halfedge->vector();
                if (f->e_bridge_right->halfedge->node != f->n_stretch_right) {
                  vec_right = f->e_bridge_right->halfedge->twin->vector();
                }
                Eigen::RowVector3d pos_right = get_intersection(f->n_stretch_right->pos, pos_interp,
                                                                vec_right, vec_interp);

                // TODO: interact out of the bridge edges
                auto n_interp_left = new Node();
                auto n_interp_right = new Node();

                // connect interp_left with up and down
                {
                  n_interp_left->pos = pos_left;
                  int i_prev = f->e_bridge_left->nodes_interp_as_left.size() - 1;
                  if (i_prev < 0) {
                    if (f->is_up) {
                      n_interp_left->down = f->n_stretch_left;
                      n_interp_left->down->up = n_interp_left;
                    } else {
                      n_interp_left->down = f->n_bridge;
                    }
                  } else {
                    n_interp_left->down = f->e_bridge_left->nodes_interp_as_left[i_prev];
                    n_interp_left->down->up = n_interp_left;
                  }

                  if (i_interp == f->num_interp - 1) {
                    if (f->is_up) {
                      n_interp_left->up = f->n_bridge;
                    } else {
                      n_interp_left->up = f->n_stretch_left;
                      n_interp_left->up->down = n_interp_left;
                    }
                  }
                }

                // connect interp_right with up and down
                {
                  n_interp_right->pos = pos_right;
                  int i_prev = f->e_bridge_right->nodes_interp_as_right.size() - 1;
                  if (i_prev < 0) {
                    if (f->is_up) {
                      n_interp_right->down = f->n_stretch_right;
                      n_interp_right->down->up = n_interp_right;
                    } else {
                      n_interp_right->down = f->n_bridge;
                    }
                  } else {
                    n_interp_right->down = f->e_bridge_right->nodes_interp_as_right[i_prev];
                    n_interp_right->down->up = n_interp_right;
                  }

                  if (i_interp == f->num_interp - 1) {
                    if (f->is_up) {
                      n_interp_right->up = f->n_bridge;
                    } else {
                      n_interp_right->up = f->n_stretch_right;
                      n_interp_right->up->down = n_interp_right;
                    }
                  }
                }

                f->e_bridge_left->nodes_interp_as_left.push_back(n_interp_left);
                f->e_bridge_right->nodes_interp_as_right.push_back(n_interp_right);
                n_interp_left->right = n_interp_right;
                n_interp_right->left = n_interp_left;
              }

//              if (f->is_up) {
//                viewer.data().add_label(f->centroid(), to_string(f->num_interp));
//              }
//              else {
//                viewer.data().add_label(f->centroid(), to_string(f->num_interp));
//              }

            }
          }
        }
      }

      if (ImGui::Button("merge")) {

        for (auto n : nodes) {
          n->i_unvisited_vector = unvisited_vector.size();
          unvisited_vector.push_back(n);
          unvisited.emplace(n);
        }

        // initialize
        for (auto f : faces) {
          if (f->is_saddle) continue;
          if (f->is_external) continue;

          Edge* e = f->e_bridge_left;

          Face* f_left = (e->halfedge->face == f) ?
            e->halfedge->twin->face :
            e->halfedge->face;
          f->left = f_left;
          f_left->right = f;

          Halfedge* h = (e->halfedge->face == f) ?
            e->halfedge->twin :
            e->halfedge;

          // new
//          merge_edge(e, f, f_left, h);

          // old
          for (int i = 0; i < max(e->nodes_interp_as_left.size(), e->nodes_interp_as_right.size()); i++) {
            Node *n_mid = new Node();
            Node *n_as_left;
            Node *n_as_right;
            bool exist_as_left = false;
            bool exist_as_right = false;

            if (i < e->nodes_interp_as_left.size()) {
              n_as_left = f->is_up ?
                                e->nodes_interp_as_left[i]
                                : e->nodes_interp_as_left[e->nodes_interp_as_left.size() - i - 1];
              exist_as_left = true;
            }

            if (i < e->nodes_interp_as_right.size()) {
              n_as_right = f_left->is_up ?
                                 e->nodes_interp_as_right[i]
                                 : e->nodes_interp_as_right[e->nodes_interp_as_right.size() - i - 1];
              exist_as_right = true;
            }

            if (exist_as_left and exist_as_right) {
              n_mid->pos = (n_as_left->pos + n_as_right->pos) / 2.;
              n_mid->left = n_as_right->left;
              n_mid->right = n_as_left->right;
              n_mid->left->right = n_mid;
              n_mid->right->left = n_mid;
            }
            else if (exist_as_left) {
              n_mid->pos = n_as_left->pos;
              n_mid->right = n_as_left->right;
              n_mid->right->left = n_mid;
              n_mid->is_end = true;
              n_mid->is_left_end = true;
            }
            else if (exist_as_right) {
              n_mid->pos = n_as_right->pos;
              n_mid->left = n_as_right->left;
              n_mid->left->right = n_mid;
              n_mid->is_end = true;
              n_mid->is_right_end = true;
            }
            e->nodes_interp.push_back(n_mid);
            unvisited.emplace(n_mid);
            n_mid->i_unvisited_vector = unvisited_vector.size();
            n_mid->is_interp_bridge = true;
            unvisited_vector.push_back(n_mid);
          }
        }


        cout<<"connect left and right"<<endl;
        // connect left and right
        for (auto f : faces) {
          if (f->is_external) continue;
          if (f->is_saddle) continue;

          // new
          int i_left = 0;
          int i_right = 0;
          while (true) {
            if (i_left >= f->e_bridge_left->nodes_interp.size()) {
              break;
            }
            if (i_right >= f->e_bridge_right->nodes_interp.size()) {
              break;
            }

            Node* n_left = f->e_bridge_left->nodes_interp[i_left];
            Node* n_right = f->e_bridge_right->nodes_interp[i_right];

            if (n_left->is_right_end) {
              i_left += 1;
              continue;
            }

            if (n_right->is_left_end) {
              i_right += 1;
              continue;
            }


            n_left->right = n_right;
            n_right->left = n_left;

            vector<Node*> segment{};
            segment.push_back(n_left);
            segment.push_back(n_right);
            segments.push_back(segment);

            i_left += 1;
            i_right += 1;
          }

          // new 2
//          for (int i = 0; i < max(f->e_bridge_left->nodes_interp.size(), f->e_bridge_right->nodes_interp.size()); i++) {
//            if (i < f->e_bridge_left->nodes_interp.size() and i < f->e_bridge_right->nodes_interp.size()) {
//
//              f->e_bridge_left->nodes_interp[i]->right = f->e_bridge_right->nodes_interp[i];
//              f->e_bridge_right->nodes_interp[i]->left = f->e_bridge_left->nodes_interp[i];
//
//              f->e_bridge_left->nodes_interp[i]->is_end = false;
//              f->e_bridge_left->nodes_interp[i]->is_right_end = false;
//              f->e_bridge_right->nodes_interp[i]->is_end = false;
//              f->e_bridge_right->nodes_interp[i]->is_left_end = false;
//
//              vector<Node *> segment{};
//              segment.push_back(f->e_bridge_left->nodes_interp[i]);
//              segment.push_back(f->e_bridge_left->nodes_interp[i]->right);
//              segments.push_back(segment);
//
////              if (f->e_bridge_left->nodes_interp[i]->is_right_end and
////                  f->e_bridge_left->nodes_interp[i]->right->is_left_end) {
////                f->e_bridge_left->nodes_interp[i]->is_right_end = false;
////                f->e_bridge_left->nodes_interp[i]->is_end = false;
////                f->e_bridge_left->nodes_interp[i]->right->is_left_end = false;
////                f->e_bridge_left->nodes_interp[i]->right->is_end = false;
////              }
//
//            }
//            else if (i < f->e_bridge_left->nodes_interp.size()) { // left exist
//              f->e_bridge_left->nodes_interp[i]->is_end = true;
//              f->e_bridge_left->nodes_interp[i]->is_right_end = true;
//            }
//            else {
//              f->e_bridge_right->nodes_interp[i]->is_end = true;
//              f->e_bridge_right->nodes_interp[i]->is_left_end = true;
//            }
//
//          }

          //old
//          for (int i = 0; i < min(f->e_bridge_left->nodes_interp.size(), f->e_bridge_right->nodes_interp.size()); i++) {
//            f->e_bridge_left->nodes_interp[i]->right = f->e_bridge_right->nodes_interp[i];
//            f->e_bridge_right->nodes_interp[i]->left = f->e_bridge_left->nodes_interp[i];
//
//            vector<Node*> segment{};
//            if (f->e_bridge_left->nodes_interp[i]->is_right_end and
//                f->e_bridge_left->nodes_interp[i]->right->is_left_end) {
//              f->e_bridge_left->nodes_interp[i]->is_right_end = false;
//              f->e_bridge_left->nodes_interp[i]->is_end = false;
//              f->e_bridge_left->nodes_interp[i]->right->is_left_end = false;
//              f->e_bridge_left->nodes_interp[i]->right->is_end = false;
//            }
//
//            segment.push_back(f->e_bridge_left->nodes_interp[i]);
//            segment.push_back(f->e_bridge_left->nodes_interp[i]->right);
//            segments.push_back(segment);
//          }
        }

        cout<<"connect up and down within nodes_interp"<<endl;
        // connect up and down within nodes_interp
        for (auto f : faces) {
          if (f->is_external) continue;
          if (f->is_saddle) continue;

          Edge* e = f->e_bridge_left;
          Halfedge* h = e->halfedge;
          if (h->node->idx_iso == -1 or h->node->idx_iso == h->twin->node->idx_iso + 1 ) {
            h = h->twin;
          }

          for (int i = 0; i < e->nodes_interp.size(); i++) {
            if (i > 0) {
              e->nodes_interp[i]->down = e->nodes_interp[i - 1];
            }
            if (i < e->nodes_interp.size() - 1) {
              e->nodes_interp[i]->up = e->nodes_interp[i + 1];
            }
          }
        }

        cout<<"connect ups and downs"<<endl;
        // connect ups and downs
        for (auto n : nodes) {
          if (n->is_cone or n->is_saddle) { // summit / saddle
            for (auto h : get_halfedges_of_node(n)) {
              if (h->edge->nodes_interp.size() > 0) {
                Node *n_top = h->edge->nodes_interp[h->edge->nodes_interp.size() - 1];
                n_top->up = n;
                n->downs.push_back(n_top);
              }
              else {
                n->downs.push_back(h->twin->node);
                h->twin->node->ups.push_back(n);
              }
            }
          }
          else {
            for (auto h : get_halfedges_of_node(n)) {
              if (h->twin->node->is_cone or h->twin->node->idx_iso > n->idx_iso) {
                if (h->edge->nodes_interp.size() > 0) {
                  n->ups.push_back(h->edge->nodes_interp[0]);
                  h->edge->nodes_interp[0]->down = n;
                }
                else {
                  n->ups.push_back(h->twin->node);
                }
              }
              else if (h->twin->node->idx_iso == n->idx_iso) {
                continue;
              }
              else if (h->twin->node->idx_iso < n->idx_iso) {
                if (h->edge->nodes_interp.size() > 0) {
                  n->downs.push_back(h->edge->nodes_interp[h->edge->nodes_interp.size() - 1]);
                  h->edge->nodes_interp[h->edge->nodes_interp.size() - 1]->up = n;
                }
                else {
                  n->downs.push_back(h->twin->node);
                }
              }
            }
          }
        }

        cout<<"connect two end nodes"<<endl;
        // connect two end nodes of the double back
        for (auto f : faces) {
          if (f->is_external) continue;
          if (f->is_saddle) continue;

          if (f->e_bridge_left->nodes_interp_as_left.size() != f->e_bridge_left->nodes_interp_as_right.size()) {
            int num_interp_new = f->e_bridge_left->nodes_interp.size() - 1;
            for (int i_interp = 0; i_interp < num_interp_new; i_interp++) {
              Node* n_up;
              Node* n_down;
              if (f->is_up) {
                n_up = f->n_bridge;
                n_down = f->n_stretch_left;
              }
              else {
                n_up = f->n_stretch_left;
                n_down =f->n_bridge;
              }

              f->e_bridge_left->nodes_interp[i_interp]->pos =
                n_down->pos * (num_interp_new - i_interp) / (num_interp_new + 1)
                + n_up->pos * (i_interp + 1) / (num_interp_new + 1);
            }
            f->e_bridge_left->nodes_interp[num_interp_new]->pos =
              f->e_bridge_left->nodes_interp[num_interp_new - 1]->pos;

          }
        }

      }

      if (ImGui::Button("trace")) {
        viewer.data().V.resize(0,3);
        viewer.data().F.resize(0,3);
        viewer.data().points.resize(0, 6);
        viewer.data().lines.resize(0, 9);
        viewer.data().labels_positions.resize(0, 3);
        viewer.data().labels_strings.clear();

        Node* n_iter;
        int iso_min;

        while (not unvisited.empty()) {
          // find a random node on the lowest loop
          {
            iso_min = 1e8;
            for (auto n : nodes) {
              if (not unvisited.count(n)) continue;
              if (n->is_cone or n->is_saddle) {
                unvisited.erase(n);
                continue;
              }
              if (n->on_saddle_side) continue;

              if (n->idx_iso < iso_min) {
                iso_min = n->idx_iso;
                n_iter = n;
              }
            }
            if (iso_min > 1e7) break;

            unvisited.erase(n_iter); n_iter->i_path = printing_path.size(); printing_path.push_back(n_iter);
            n_iter->shrinkage = -1;
          }

          // one trace
          while (true) {
            // one loop
            while (true) {
              bool move_horizontally = false;
              Node* n_up_a = n_iter;
              Node* n_down_a = n_iter;
              Node* n_up_b;
              Node* n_down_b;

              // move or stop
              {
                if (n_iter->is_end) { // at the end of one segment
                  if (n_iter->left and unvisited.count(n_iter->left)) {   // move left
                    n_iter = n_iter->left;
                    visit(n_iter);
                    move_horizontally = true;
                  }
                  else if (n_iter->right and unvisited.count(n_iter->right)) {    // move right
                    n_iter = n_iter->right;
                    visit(n_iter);
                    move_horizontally = true;
                  }
                  else {
                    if (n_iter->up and n_iter->up->is_end and unvisited.count(n_iter->up)) {    // move up to next double back segment
                      n_iter = n_iter->up;
                      visit(n_iter);
                    }
                    else {  // move down
                      Node* n_down = n_iter->down;
                      bool found_down = false;
                      while (n_down) {
                        if (unvisited.count(n_down)) {
                          n_iter = n_down;
                          visit(n_iter);
                          found_down = true;
                          break;
                        }
                        n_down = n_down->down;
                      }

                      if (not found_down) break;  // end of the loop
                    }
                  }
                }

                else {  // not end
                  bool has_double_back = false;
                  if (n_iter->is_interp_bridge) {
                    if (n_iter->up and n_iter->up->is_end and
                        unvisited.count(n_iter->up)) {  // move up into double back
                      unvisited.emplace(n_iter);  // put back the beginning of the double back
                      n_iter = n_iter->up;
                      visit(n_iter);
                      has_double_back = true;
                    }
                  }
                  else {
                    for (auto nu : n_iter->ups) {
                      if (nu->is_end and unvisited.count(nu)) {
                        unvisited.emplace(n_iter);
                        n_iter = nu;
                        visit(n_iter);
                        has_double_back = true;
                      }
                    }
                  }

                  if (not has_double_back) {  // no up double back
                    if (n_iter->left and unvisited.count(n_iter->left)) {   // move left
                      n_iter = n_iter->left;
                      visit(n_iter);
                      move_horizontally = true;
                    }
                    else if (n_iter->right and unvisited.count(n_iter->right)) {    // move right
                      n_iter = n_iter->right;
                      visit(n_iter);
                      move_horizontally = true;
                    }

                    else {   // move up
                      Node* n_up;
                      if (n_iter->is_interp_bridge) {
                        n_up = n_iter->up;
                      }
                      else {
                        if (n_iter->ups.size() > 0) {
                          n_up = n_iter->ups[0];
                        }
                        else {
                          break;
                        }
                      }

                      bool found_up = false;
                      while (true) {
                        if (not n_up) break;
                        if (n_up->is_cone or n_up->is_saddle) break;
                        if (unvisited.count(n_up)) {
                          found_up = true;
                          break;
                        }
                        n_up = n_up->up;
                      }

                      if (found_up) {
                        n_iter = n_up;
                        visit(n_iter);
                        unvisited.emplace(n_iter);  // fix the missed segment at the beginning of the loop
                      } else {
                        break;
                      }

                    }
                  }
                }
              }

              /*
              {
                if (n_iter->is_end) {
                  if (n_iter->up->is_end) {
                    if (n_iter->down->is_end) {cout<<"up and down is_end"<<endl; getchar();}
                    n_iter = n_iter->down;   // exit double back
                  }
                  else {
                    if (not n_iter->down->is_end or not unvisited.count(n_iter->down) ) {
                      cout<<"not ->down->is_end"<<endl;
                      cout<<"n_iter: "<<n_iter->i_unvisited_vector<<endl;
                      if (not n_iter->down->is_end) cout<<"not down end"<<endl;
                      if (not unvisited.count(n_iter->down)) cout<<"visited"<<endl;
                      getchar(); break;
                    }
                    n_iter = n_iter->down;
                    unvisited.erase(n_iter); n_iter->i_path = printing_path.size(); printing_path.push_back(n_iter);
                    n_iter->shrinkage = -1;
                  }
                }

                if (n_iter->idx != -1) {  // not saddle or cone
                  for (auto nu : n_iter->ups) {
                    if (nu->is_end and unvisited.count(nu)) {
                      n_iter = nu->up;
                      unvisited.erase(n_iter); n_iter->i_path = printing_path.size(); printing_path.push_back(n_iter);
                      n_iter->shrinkage = -1;
                    }
                  }
                } else
                if (n_iter->up and n_iter->up->is_end and unvisited.count(n_iter->up)) {  // enter double back
                  n_iter = n_iter->up->up;
                  unvisited.erase(n_iter); n_iter->i_path = printing_path.size(); printing_path.push_back(n_iter);
                  n_iter->shrinkage = -1;
                }
              }
              */

              // horizontal traces
              if (move_horizontally) {
//                if (n_iter->right and unvisited.count(n_iter->right)) {
//                  n_iter = n_iter->right;
//                } else if (n_iter->left and unvisited.count(n_iter->left)) {
//                  n_iter = n_iter->left;
//                } else {
//                  break;  // exit the loop
//                }

                double shrinkage;
                if (n_iter->idx != -1) {
                  shrinkage = get_halfedge(n_iter, n_up_a)->edge->shrinkage;
                }
                else {
                  n_up_b = n_iter;
                  n_down_b = n_iter;

                  while (n_up_a->idx == -1) {
//                    if (n_up_a->is_cone or n_up_a->is_saddle) break;
                    if (not n_up_a->up) {
                      cout << "not n_up_a->up " << n_up_a->i_unvisited_vector << endl; getchar();
                      break;
                    }
                    n_up_a = n_up_a->up;
                  }

                  while (n_down_a->idx == -1) {
                    if (not n_down_a->down) {
                      cout << "not n_down_a->up " << n_down_a->i_unvisited_vector << endl; getchar();
                      break;
                    }
                    n_down_a = n_down_a->down;
                  }

                  while (n_up_b->idx == -1) {
//                    if (n_up_b->is_cone or n_up_b->is_saddle) break;
                    if (not n_up_b->up) {
                      cout << "not n_up_b->up " << n_up_b->i_unvisited_vector << endl; getchar();
                      break;
                    }
                    n_up_b = n_up_b->up;
                  }

                  while (n_down_b->idx == -1) {
                    if (not n_down_b->down) {
                      cout << "not n_down_b->up " << n_down_b->i_unvisited_vector << endl; getchar();
                      break;
                    }
                    n_down_b = n_down_b->down;
                  }


                  Face *f_iter;
                  {
                    Halfedge *ha = get_halfedge(n_up_a, n_down_a);
                    Halfedge *hb = get_halfedge(n_up_b, n_down_b);
                    if (n_up_a == n_up_b) {
                      if (ha->twin->next == hb) {
                        f_iter = hb->face;
                      } else if (hb->twin->next == ha) {
                        f_iter = ha->face;
                      } else {
                        cout << "ha, hb..." << endl; getchar();
                        continue;
                      }
                    } else if (n_down_a == n_down_b) {
                      if (ha->next == hb->twin) {
                        f_iter = ha->face;
                      } else if (hb->next = ha->twin) {
                        f_iter = hb->face;
                      } else {
                        cout << "hb, ha..." << endl; getchar();
                        continue;
                      }
                    } else {
                      cout << "n_up, n_down, not equal" << endl;
                      cout<<"n_iter: "<<n_iter->i_unvisited_vector<<endl;
                      cout<<"n_up_a: "<<n_up_a->i_unvisited_vector<<endl;
                      cout<<"n_up_b: "<<n_up_b->i_unvisited_vector<<endl;
                      cout<<"n_down_a: "<<n_down_a->i_unvisited_vector<<endl;
                      cout<<"n_down_b: "<<n_down_b->i_unvisited_vector<<endl;
                      getchar();
                      break;
                    }
                  }


                  double shrinkage_up;
                  double shrinkage_down;
                  Halfedge* h_iter;

                  double shrinkage_bridge = 0;
                  double shrinkage_stretch = 0;
                  int n_shrinkage_bridge = 0;
                  int n_shrinkage_stretch = 0;
                  double len_shrinkage_bridge = 0;
                  double len_shrinkage_stretch =0;

                  if (f_iter->n_bridge->left) {
                    h_iter = get_halfedge(f_iter->n_bridge->left, f_iter->n_bridge);
                    shrinkage_bridge += h_iter->edge->shrinkage * h_iter->length();
                    n_shrinkage_bridge++; len_shrinkage_bridge += h_iter->length();

                    if (f_iter->n_bridge->left->left) {
                      h_iter = get_halfedge(f_iter->n_bridge->left->left, f_iter->n_bridge->left);
                      shrinkage_bridge += h_iter->edge->shrinkage * h_iter->length();
                      n_shrinkage_bridge++; len_shrinkage_bridge += h_iter->length();
                    }
                  }
                  if (f_iter->n_bridge->right) {
                    h_iter = get_halfedge(f_iter->n_bridge->right, f_iter->n_bridge);
                    shrinkage_bridge += h_iter->edge->shrinkage * h_iter->length();
                    n_shrinkage_bridge++;len_shrinkage_bridge += h_iter->length();
                    if (f_iter->n_bridge->right->right) {
                      h_iter = get_halfedge(f_iter->n_bridge->right->right, f_iter->n_bridge->right);
                      shrinkage_bridge += h_iter->edge->shrinkage * h_iter->length();
                      n_shrinkage_bridge++; len_shrinkage_bridge += h_iter->length();
                    }
                  }

                  shrinkage_stretch += f_iter->e_stretch->shrinkage * f_iter->e_stretch->length();
                  n_shrinkage_stretch++; len_shrinkage_stretch += f_iter->e_stretch->length();
                  if (f_iter->n_stretch_left->left) {
                    h_iter = get_halfedge(f_iter->n_stretch_left->left, f_iter->n_stretch_left);
                    shrinkage_stretch += h_iter->edge->shrinkage * h_iter->length();
                    n_shrinkage_stretch++; len_shrinkage_stretch += h_iter->length();
                    if (f_iter->n_stretch_left->left->left) {
                      h_iter =get_halfedge(f_iter->n_stretch_left->left->left, f_iter->n_stretch_left->left);
                      shrinkage_stretch += h_iter->edge->shrinkage * h_iter->length();
                      n_shrinkage_stretch++; len_shrinkage_stretch += h_iter->length();
                    }
                  }
                  if (f_iter->n_stretch_right->right) {
                    h_iter = get_halfedge(f_iter->n_stretch_right->right, f_iter->n_stretch_right);
                    shrinkage_stretch += h_iter->edge->shrinkage * h_iter->length();
                    n_shrinkage_stretch++; len_shrinkage_stretch += h_iter->length();
                    if (f_iter->n_stretch_right->right->right) {
                      h_iter = get_halfedge(f_iter->n_stretch_right->right->right, f_iter->n_stretch_right->right);
                      shrinkage_stretch += h_iter->edge->shrinkage * h_iter->length();
                      n_shrinkage_stretch++; len_shrinkage_stretch += h_iter->length();
                    }
                  }

                  shrinkage_bridge /= len_shrinkage_bridge;
                  shrinkage_stretch /= len_shrinkage_stretch;

                  if (f_iter->n_bridge->is_cone) shrinkage_bridge = shrinkage_stretch;

                  if (f_iter->is_up) {
                    shrinkage_up = shrinkage_bridge;
                    shrinkage_down = shrinkage_stretch;
                  } else {
                    shrinkage_up = shrinkage_stretch;
                    shrinkage_down = shrinkage_bridge;
                  }

                  Node *nn_iter = n_iter;
                  int to_up = 0;
                  int to_down = 0;

                  while (true) {
                    if (nn_iter->idx != -1 or nn_iter->is_cone) break;

                    if (nn_iter->up) {
                      nn_iter = nn_iter->up;
                      to_up++;
                    }
                  }

                  while (true) {
                    if (nn_iter->idx != -1 or nn_iter->is_saddle) break;

                    if (nn_iter->down) {
                      nn_iter = nn_iter->down;
                      to_down++;
                    }
                  }

                  shrinkage = shrinkage_up * to_down / (to_up + to_down) +
                              shrinkage_down * to_up / (to_up + to_down);
                }

                unvisited.erase(n_iter); n_iter->i_path = printing_path.size(); printing_path.push_back(n_iter);
                n_iter->shrinkage = shrinkage;
              }

            }

            // move up
            {
              if (n_iter->idx == -1 and not n_iter->is_cone and not n_iter->is_saddle) {
                if (n_iter->up) {
                  if (n_iter->up->is_cone or n_iter->up->is_saddle) break;
                  n_iter = n_iter->up;
                }
                else break;
              } else {
                if (not n_iter->ups.empty()) {
                  if (n_iter->ups[0]->is_cone or n_iter->ups[0]->is_saddle) break;
                  else {
                    n_iter = n_iter->ups[0];
                  }
                }
                else {
                  cout<<"error: ups.empty()"<<endl; getchar();
                }
              }

              // move up across double-backs
              while (!unvisited.count(n_iter)) {
                viewer.data().add_points(n_iter->pos, color_blue);
                if (n_iter->idx == -1) {
                  if (n_iter->up->is_cone or n_iter->up->is_saddle) break;
                  n_iter = n_iter->up;
                } else {
                  if (n_iter->ups[0]->is_cone or n_iter->ups[0]->is_saddle) break;
                  n_iter = n_iter->ups[0];
                }
              }
              unvisited.erase(n_iter);
              n_iter->i_path = printing_path.size(); printing_path.push_back(n_iter);
              n_iter->shrinkage = -1;

            }
          }
        }

        num_iter = printing_path.size();
        display_mode = 4;
        redraw();
      }

      if (ImGui::Button("trace_saddle")) {
        cout<<"start trace saddles..."<<endl;

        for (auto n_saddle : saddles) {
          cout<<"detect saddle "<<n_saddle->idx<<endl;
          bool is_first = true;

          // collect halfedges_saddle
          vector<Halfedge*> halfedges_saddle;
          {
            halfedges_saddle = get_halfedges_of_node(n_saddle);
            for (int i = 0; i < halfedges_saddle.size(); i++) {
              halfedges_saddle[i] = halfedges_saddle[i]->twin;  // get the halfedge pointing at saddle
            }
          }

          // compute f->num_interp
          int num_interp_max = -1;
          for (auto h : halfedges_saddle) {
            Face *f = h->face;
            Eigen::RowVector3d normal = f->normal();
            Eigen::RowVector3d vec_height = normal.cross(h->prev->vector());
            vec_height.normalize();

            Eigen::RowVector3d vec_mid = h->prev->edge->centroid() - h->next->node->pos;
            double iso_distance = abs(vec_mid.dot(vec_height));
            f->num_interp = int(iso_distance / (2 * gap_size)) * 2;

            if (f->num_interp > num_interp_max) num_interp_max = f->num_interp;
          }

          // assign edges and nodes for the face
          for (auto h : halfedges_saddle) {
            Face* f = h->face;
            f->e_stretch = h->prev->edge;
            f->e_bridge_left = h->next->edge;
            f->e_bridge_right = h->edge;
            f->n_bridge = h->next->node;
            f->n_stretch_left = h->prev->node;
            f->n_stretch_right = h->node;
            f->pos_stretch_mid = f->e_stretch->centroid();
            f->is_up = true;   // TODO
          }

          // interpolate
          for (auto h : halfedges_saddle) {
            Face* f = h->face;
            for (int i_interp = 0; i_interp < f->num_interp; i_interp++) {
              double weight_bridge = float(i_interp + 1) / (f->num_interp + 1);
              double weight_stretch = 1.0 - weight_bridge;

              // fix
              // Eigen::RowVector3d vec_interp = weight_bridge * f->vec_bridge + weight_stretch * f->vec_stretch;
              Eigen::RowVector3d vec_interp = f->n_stretch_left->pos - f->n_stretch_right->pos;
              vec_interp.normalize();

              Eigen::RowVector3d pos_interp = weight_bridge * f->n_bridge->pos + weight_stretch * f->pos_stretch_mid;
              f->pos_interps.push_back(pos_interp);

              Eigen::RowVector3d vec_left = f->e_bridge_left->halfedge->vector();
              if (f->e_bridge_left->halfedge->node != f->n_stretch_left) {
                vec_left = f->e_bridge_left->halfedge->twin->vector();
              }
              Eigen::RowVector3d pos_left = get_intersection(f->n_stretch_left->pos, pos_interp,
                                                             vec_left, vec_interp);

              Eigen::RowVector3d vec_right = f->e_bridge_right->halfedge->vector();
              if (f->e_bridge_right->halfedge->node != f->n_stretch_right) {
                vec_right = f->e_bridge_right->halfedge->twin->vector();
              }
              Eigen::RowVector3d pos_right = get_intersection(f->n_stretch_right->pos, pos_interp,
                                                              vec_right, vec_interp);

              // TODO: interact out of the bridge edges
              auto n_interp_left = new Node();
              auto n_interp_right = new Node();

              // connect interp_left with up and down
              {
                n_interp_left->pos = pos_left;
                int i_prev = f->e_bridge_left->nodes_interp_as_left.size() - 1;
                if (i_prev < 0) {
                  if (f->is_up) {
                    n_interp_left->down = f->n_stretch_left;
                    n_interp_left->down->up = n_interp_left;
                  } else {
                    n_interp_left->down = f->n_bridge;
                  }
                } else {
                  n_interp_left->down = f->e_bridge_left->nodes_interp_as_left[i_prev];
                  n_interp_left->down->up = n_interp_left;
                }

                if (i_interp == f->num_interp - 1) {
                  if (f->is_up) {
                    n_interp_left->up = f->n_bridge;
                  } else {
                    n_interp_left->up = f->n_stretch_left;
                    n_interp_left->up->down = n_interp_left;
                  }
                }
              }

              // connect interp_right with up and down
              {
                n_interp_right->pos = pos_right;
                int i_prev = f->e_bridge_right->nodes_interp_as_right.size() - 1;
                if (i_prev < 0) {
                  if (f->is_up) {
                    n_interp_right->down = f->n_stretch_right;
                    n_interp_right->down->up = n_interp_right;
                  } else {
                    n_interp_right->down = f->n_bridge;
                  }
                } else {
                  n_interp_right->down = f->e_bridge_right->nodes_interp_as_right[i_prev];
                  n_interp_right->down->up = n_interp_right;
                }

                if (i_interp == f->num_interp - 1) {
                  if (f->is_up) {
                    n_interp_right->up = f->n_bridge;
                  } else {
                    n_interp_right->up = f->n_stretch_right;
                    n_interp_right->up->down = n_interp_right;
                  }
                }
              }

              f->e_bridge_left->nodes_interp_as_left.push_back(n_interp_left);
              f->e_bridge_right->nodes_interp_as_right.push_back(n_interp_right);
              n_interp_left->right = n_interp_right;
              n_interp_right->left = n_interp_left;
            }
          }

          // merge
//          for (auto h : halfedges_saddle) {
//            Face* f = h->twin->face;
//            Edge* e = f->e_bridge_left;
//            Face* f_left = (e->halfedge->face == f) ?
//                           e->halfedge->twin->face :
//                           e->halfedge->face;
//            f->left = f_left;
//            f_left->right = f;
//
//            merge_edge(e, f, f_left, h);
//          }

          // collect path
          bool from_bottom = true;
          for (auto h : halfedges_saddle) {
            Edge* e_l = h->prev->prev->edge;
            Edge* e_r = h->edge;
            vector<Node*> nodes_left = e_l->nodes_interp_as_left;
            vector<Node*> nodes_right = e_r->nodes_interp_as_right;

            vector<Node*> path;
            int i = -1;
            bool from_left =true;
            while (true) {

              if (i == -1) {
                if (h->node->on_saddle_side) {
                  path.push_back(h->node);
                  path.push_back(h->prev->node);
                }
                i = 0;
                continue;
              }

              if (i >= nodes_left.size() or i >= nodes_right.size()) break;

              if (from_left) {
                path.push_back(nodes_left[i]);
                path.push_back(nodes_right[i]);
              }
              else {
                path.push_back(nodes_right[i]);
                path.push_back(nodes_left[i]);
              }


              i++;
              from_left = not from_left;
            }

            if (not from_bottom) {
              reverse(path.begin(), path.end());
            }

            for (Node* n : path) {
              n->shrinkage = h->node->shrinkage;
              if (n->shrinkage == -1 and not h->node->downs.empty()) {
                n->shrinkage = h->node->downs[0]->shrinkage;
              }
              if (n->shrinkage == -1 and not h->node->ups.empty()) {
                n->shrinkage = h->node->ups[0]->shrinkage;
              }
              if (n->shrinkage == -1) {
                n->shrinkage = h->twin->next->next->node->shrinkage;
              }
              if (is_first) {
                n->shrinkage = -1;
                n->is_move = true;
                is_first = false;
              }

              printing_path.push_back(n);
            }

            from_bottom = not from_bottom;
          }

//
//          vector<Node*> paths;
//          for (int i_interp = -1; i_interp < num_interp_max; i_interp++) {
//            for (auto h : halfedges_saddle) {
//              Edge* e = h->edge;
//              Node *n_interp;
//              if (i_interp == -1) {
//                n_interp = h->node;
//              }
//              else if (i_interp < e->nodes_interp.size() ) {
//                n_interp = e->nodes_interp[i_interp];
//              }
//              else {
//                n_interp = n_saddle;
//              }
//
//
//              if (is_first) {
//                n_interp->shrinkage = -1;
//                is_first = false;
//              }
//              else {
//                n_interp->shrinkage = h->node->shrinkage;
//                if (n_interp->shrinkage == -1 and not h->node->downs.empty()) {
//                  n_interp->shrinkage = h->node->downs[0]->shrinkage;
//                }
//                if (n_interp->shrinkage == -1) {
//                  n_interp->shrinkage = h->twin->next->next->node->shrinkage;
//                }
//
//               }
//              printing_path.emplace_back(n_interp);
//              paths.emplace_back(n_interp);
//
//            }
//          }

          display_mode = 4;
          redraw();

        }
      }

      if (ImGui::Button("adjust_cone")) {
        for (auto n : printing_path) {
          for (auto nc : cones) {
            double d = (n->pos - nc->pos).norm();
            if (d < radius_cone) {
              if (n->shrinkage == -1) continue;
              double t = 1 - d / radius_cone;
              double s = t * shrinkage_cone + (1-t) * shrinkage_cone_down;
              n->shrinkage = s;
            }
          }
        }
        redraw();

      }

      if (ImGui::Button("trim_center")) {
        vector<Node*> printing_path_new;
        for (auto n : printing_path) {
          bool keep = true;
          for (auto nc :cones) {
            double d = (n->pos - nc->pos).norm();
            if (d < radius_trim) {
              keep = false;
            }
          }
          for (auto nc :saddles) {
            double d = (n->pos - nc->pos).norm();
            if (d < radius_trim) {
              keep = false;
            }
          }

          if (keep) {
            printing_path_new.push_back(n);
          }
        }
        printing_path.clear();
        printing_path = printing_path_new;
        redraw();

      }

      if (ImGui::Button("save")) {
        // scale
        std::string ofile_name = igl::file_dialog_save();
        std::ofstream ofile(ofile_name);
        if (ofile.is_open()) {
          for (auto n : printing_path) {
            Eigen::RowVector3d p = n->pos;
            if (n->is_move) n->shrinkage = -1;
            ofile << p[0] << " " << p[1] << " " << n->is_move << " " << n->shrinkage <<endl;
          }

          for (auto f : faces) {
            if (f->is_external) continue;
            ofile << "f "<< f->node(0)->pos.x() <<" "<< f->node(0)->pos.y() <<" "
                          << f->node(1)->pos.x() <<" "<< f->node(1)->pos.y() <<" "
                          << f->node(2)->pos.x() <<" "<< f->node(2)->pos.y() <<" "
                          << f->node(0)->color[0] <<" "<< f->node(0)->color[1] <<" "<< f->node(0)->color[2]<<" "
                          << f->node(1)->color[0] <<" "<< f->node(1)->color[1] <<" "<< f->node(1)->color[2]<<" "
                          << f->node(2)->color[0] <<" "<< f->node(2)->color[1] <<" "<< f->node(2)->color[2]<<" "
                          << f->opacity() * scale_ratio * scale_ratio << endl;
          }
        }
      }

//      if (ImGui::Button("save_img")) {
//        string ofile_name = igl::file_dialog_save();
//        const char* name = ofile_name.c_str();
//        stbi_write_bmp(name, w_out, w_out, n_c_out, img_out);
//      }

//      if (ImGui::Button("test")) {
//        // init image_out
//        int num_px_x = w_out;
//        int num_px_y = w_out;
//        double w_screen = platform_width;
//        double h_screen = platform_length;
//        double size_px = w_screen / num_px_x;
//
//        int size_img = num_px_x * num_px_y * n_c_out;
//        img_out = (unsigned char*) malloc(size_img);
//        for (int i = 0; i < num_px_x * num_px_y; i++) {
//          if (i + 2 >= num_px_x * num_px_y * n_c_out) {
//            cout<<"oversize"<<endl;
//            getchar();
//          }
//          img_out[i*n_c_out] = 255;
//          img_out[i*n_c_out+1] = 255;
//          img_out[i*n_c_out+2] = 255;
//        }
//
//        auto get_i_px = [&](double x, double y)->Eigen::RowVector2i {
//          int i_px_x = x / size_px;
//          int i_px_y = y / size_px;
//          if (i_px_x < 0 or i_px_x >= num_px_x) {
//            cout<<"overflow"<<endl;getchar();
//            return Eigen::RowVector2i(0, 0);
//          }
//          if (i_px_y < 0 or i_px_y >= num_px_x) {
//            cout<<"overflow"<<endl;getchar();
//            return Eigen::RowVector2i(0, 0);
//          }
//          return Eigen::RowVector2i(i_px_x, i_px_y);
//        };
//
//        auto get_coord = [&](int i_px_x, int i_px_y)->Eigen::RowVector2d {
//          double x = i_px_x * size_px + size_px / 2;
//          double y = i_px_y * size_px + size_px / 2;
//          return Eigen::RowVector2d(x, y);
//        };
//
//        auto paint_px = [&](int i_px_x, int i_px_y, Eigen::RowVector3d c) {
//          int i = i_px_y * num_px_x * 3 + i_px_x * 3;
//          img_out[i] = char(int(c[0] * 255));
//          img_out[i+1] = char(int(c[1] * 255));
//          img_out[i+2] = char(int(c[2] * 255));
//        };
//
//        auto sign = [&](Eigen::RowVector2d p0, Eigen::RowVector2d p1, Eigen::RowVector2d p2) -> double {
//          return (p0.x() - p2.x()) * (p1.y() - p2.y()) - (p1.x() - p2.x()) * (p0.y() - p2.y());
//        };
//        auto in_triangle = [&](Eigen::RowVector2d p, Eigen::RowVector2d p0,
//                              Eigen::RowVector2d p1, Eigen::RowVector2d p2) -> bool {
//          float d1, d2, d3;
//          bool has_neg, has_pos;
//
//          d1 = sign(p, p0, p1);
//          d2 = sign(p, p1, p2);
//          d3 = sign(p, p2, p0);
//
//          has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
//          has_pos = (d1 > 0) || (d2 > 0) || (d3 > 0);
//
//          return !(has_neg && has_pos);
//        };
//
//        for (auto f : faces) {
//          if (f->is_external) continue;
//          cout<<"f_idx: "<<f->idx<<endl;
////          cout<<f->halfedge->node->idx<<endl;
////          cout<<f->halfedge->next->node->idx<<endl;
////          cout<<f->halfedge->next->next->node->idx<<endl;
//
//          // get bbox
//          int i_px_x0 = 1E8;
//          int i_px_y0 = 1E9;
//          int i_px_x1 = -1;
//          int i_px_y1 = -1;
//          for (int i_n = 0; i_n < 3; i_n++) {
//            Node* n = f->node(i_n);
//            Eigen::RowVector2i i_px = get_i_px(n->pos.x(), n->pos.y());
//            if (i_px.x() < i_px_x0) i_px_x0 = i_px.x();
//            if (i_px.y() < i_px_y0) i_px_y0 = i_px.y();
//            if (i_px.x() > i_px_x1) i_px_x1 = i_px.x();
//            if (i_px.y() > i_px_y1) i_px_y1 = i_px.y();
//          }
//
//          int i_px_x = i_px_x0;
//          int i_px_y = i_px_y0;
//          while (true) {
//            cout<<"get_coord("<<i_px_x<<", "<<i_px_y<<")"<<endl;
//            Eigen::RowVector2d p = get_coord(i_px_x, i_px_y) + Eigen::RowVector2d(size_px/2, size_px/2);
//            Eigen::RowVector2d p0;
//            Eigen::RowVector2d p1;
//            Eigen::RowVector2d p2;
//            cout<<"a"<<endl;
//            p0 << f->node(0)->pos[0], f->node(0)->pos[1];
//            p1 << f->node(1)->pos[0], f->node(1)->pos[1];
//            p2 << f->node(2)->pos[0], f->node(2)->pos[1];
//            cout<<"b"<<endl;
//
//            bool in_tri = in_triangle(p, p0, p1, p2);
//            cout<<"check"<<in_tri<<endl;
//            if (in_tri) {
//              cout<<"paint"<<endl;
//              // TODO: bc
//              paint_px(i_px_x, i_px_y, f->node(0)->color);
//              cout<<"done paint"<<endl;
//            }
//            cout<<"hehe"<<endl;
//            i_px_x++;
//            if (i_px_x > i_px_x0) {
//              i_px_x = i_px_x0;
//              i_px_y++;
//            }
//            cout<<"haha"<<endl;
//            if (i_px_y > i_px_y0) break;
//          }
//          cout<<"haha"<<endl;
//        }
//        cout<<"eee"<<endl;
//
//
//        string ofile_name = igl::file_dialog_save();
//        const char* name = ofile_name.c_str();
//        stbi_write_bmp(name, w_out, w_out, n_c_out, img_out);
//        free(img_out);
//
//      }
    }
  };

  viewer.callback_key_down =
    [&](igl::opengl::glfw::Viewer& viewer, int key, int mod)->bool
    {
      if (type_focus == "node") {
        if (key == GLFW_KEY_LEFT)
          if (display_mode == 3 or display_mode == 4) {
            if (unvisited_vector[idx_focus]->left) {
              idx_focus = unvisited_vector[idx_focus]->left->i_unvisited_vector;
            }
          }
          else {
            if (nodes[idx_focus]->left) {
              idx_focus = nodes[idx_focus]->left->idx;
            }
          }
        if (key == GLFW_KEY_RIGHT)
          if (display_mode == 3 or display_mode == 4) {
            if (unvisited_vector[idx_focus]->right) {
              idx_focus = unvisited_vector[idx_focus]->right->i_unvisited_vector;
            }
          }
          else {
            if (nodes[idx_focus]->right) {
              idx_focus = nodes[idx_focus]->right->idx;
            }
          }
        if (key == GLFW_KEY_UP)
          if (nodes[idx_focus]->up) {
            idx_focus = nodes[idx_focus]->up->idx;
          }
        if (key == GLFW_KEY_DOWN)
          if (nodes[idx_focus]->down) {
            idx_focus = nodes[idx_focus]->down->idx;
          }


        // fix saddle
        if (key == GLFW_KEY_P) {
          Node* n = nodes[idx_focus];
          Node* n_new = n->right;
          n->down->up = n_new;
          n_new->down = n->down;
          n->down = nullptr;
          idx_focus = n_new->idx;
        }
        if (key == GLFW_KEY_O) {
          Node* n = nodes[idx_focus];
          Node* n_new = n->left;
          n->down->up = n_new;
          n_new->down = n->down;
          n->down = nullptr;
          idx_focus = n_new->idx;
        }

      }

      if (type_focus == "halfedge") {
        Halfedge* h = halfedges[idx_focus];
        if (key == GLFW_KEY_N) {
          idx_focus = h->next->idx;
        }
        else if (key == GLFW_KEY_P) {
          idx_focus = h->prev->idx;
        }
        else if (key == GLFW_KEY_T) {
          if (h->twin)
            idx_focus = h->twin->idx;
          else
            cout<<"no twin"<<endl;
        }
        else if (key == GLFW_KEY_V) {
          type_focus = "node";
          idx_focus = h->node->idx;
        }
        else if (key == GLFW_KEY_E) {
          type_focus = "edge";
          idx_focus = h->edge->idx;
        }
        else if (key == GLFW_KEY_F) {
          type_focus = "face";
          idx_focus = h->face->idx;
        }
      }

      if (type_focus != "halfedge" and key == GLFW_KEY_H) {
        if (type_focus == "node") {
          idx_focus = nodes[idx_focus]->halfedge->idx;
        }
        else if (type_focus == "edge") {
          idx_focus = edges[idx_focus]->halfedge->idx;
        }
        else if (type_focus == "face") {
          idx_focus = faces[idx_focus]->halfedge->idx;
        }
        type_focus = "halfedge";
      }

      if (display_mode == 3 and key == GLFW_KEY_P) {
        Node* n = unvisited_vector[idx_focus];
        viewer.data().add_points(n->pos, color_white);
        if (n->idx == -1) {
          if (n->up)
            viewer.data().add_points(n->up->pos, color_blue);
          if (n->down)
            viewer.data().add_points(n->down->pos, color_green);
        }
        else {
          if (not n->ups.empty()) {
            for (auto n_up : n->ups) {
              viewer.data().add_points(n_up->pos, color_blue);
            }
          }
          if (not n->downs.empty()) {
            for (auto n_down : n->downs) {
              viewer.data().add_points(n_down->pos, color_green);
            }
          }
        }
      }

      if (display_mode == 3 and key == GLFW_KEY_U) {
        Node* n = unvisited_vector[idx_focus];
        if (n->idx == -1) {
          if (n->up) {
            idx_focus = n->up->i_unvisited_vector;
            cout<<idx_focus<<endl;
          }
          else {
            cout<<"not found"<<endl;
          }
        }
        else {
          if (n->ups[num_iter]) {
            idx_focus = n->ups[num_iter]->i_unvisited_vector;
            cout<<idx_focus<<endl;
          }
          else {
            cout<<"not found"<<endl;
          }
        }
      }

      if (display_mode == 3 and key == GLFW_KEY_D) {
        Node* n = unvisited_vector[idx_focus];
        if (n->idx == -1) {
          if (n->down) {
            if (n->down == n) cout<<"same node"<<endl;
            if (n == unvisited_vector[idx_focus]) cout<<"n ==" <<endl;
            if (n->down == unvisited_vector[idx_focus] )  cout<<"n->down =="<<endl;
            idx_focus = n->down->i_unvisited_vector;
            cout<<idx_focus<<endl;
          }
          else {
            cout<<"not found"<<endl;
          }
        }
        else {
          if (n->downs[num_iter]) {
            idx_focus = n->downs[num_iter]->i_unvisited_vector;
            cout<<idx_focus<<endl;
          }
          else {
            cout<<"not found"<<endl;
          }
        }
      }

      redraw();
    };

  viewer.callback_mouse_down =
    [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
      // intersect ray with xy flatten
      Eigen::RowVector3d v_xy;  // intersection between ray and xy plain
      double x = viewer.current_mouse_x;
      double y = viewer.core.viewport(3) - viewer.current_mouse_y;
      Eigen::Vector3d s, dir;
      igl::unproject_ray(Eigen::Vector2f(x,y), viewer.core.view, viewer.core.proj, viewer.core.viewport, s, dir);
      float t = - s[2] / dir[2];
      v_xy = s + t * dir;

      Eigen::RowVector3d color = Eigen::RowVector3d(0.6,0.6,0.8);
      // viewer.data().add_points(v_xy, color);
    };

  viewer.callback_post_draw =
    [&](igl::opengl::glfw::Viewer& viewer)->bool
    {
      if (is_flattening) {
        step();
        cout<<"convergence: "<<get_convergence()<<endl;
      }
    };

  // setup
  initialize_param();
  viewer.launch();

  return 0;
}

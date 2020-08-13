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
#include <igl/heat_geodesics.h>
//#include <igl/upsample.h>
//#include <igl/colormap.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

// Eigen
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Dense>

// ShapeOp
#include "Solver.h"
#include "Constraint.h"
#include "Force.h"

// Geodesy
#include "geometry.h"
#include "tracer.h"
#include "model.h"

using namespace std;

//////// template funcs ////////
template<typename T>
inline void label_vector(vector<T>& vec)
{
  int i = 0;
  for (auto ele : vec) {
    ele->id = i;
    i++;
  }
}

void add_face(FaceTr* f, HalfedgeMeshTr* m,
              set<NodeTr*> &nodes, set<EdgeTr*> &edges,
              set<FaceTr*> &faces, set<HalfedgeTr*> &halfedges,
              igl::opengl::glfw::Viewer &viewer);

void add_face(FaceTr* f, HalfedgeMeshTr* m,
  set<NodeTr*>& nodes, set<EdgeTr*> &edges,
  set<FaceTr*> &faces, set<HalfedgeTr*> &halfedges,
  igl::opengl::glfw::Viewer &viewer) {

  auto h = f->halfedge;
  halfedges.insert(h);
  halfedges.insert(h->next);
  halfedges.insert(h->prev);
  nodes.insert(h->node);
  nodes.insert(h->next->node);
  nodes.insert(h->prev->node);
  edges.insert(h->edge);
  edges.insert(h->next->edge);
  edges.insert(h->prev->edge);
  faces.insert(h->face);
  faces.insert(f);

  vector<HalfedgeTr*> hs;
  hs.push_back(h->twin);
  hs.push_back(h->next->twin);
  hs.push_back(h->prev->twin);
  for (auto h : hs) {
    cout<<h->face<<endl;
    if (h->face) {
      viewer.data().add_points( (h->node->pos + h->twin->node->pos)/2, Eigen::RowVector3d(0, 0.8, 0.8));
    }
    else {
      cerr<<"no h->face"<<endl;
      viewer.data().add_points( (h->node->pos + h->twin->node->pos)/2, Eigen::RowVector3d(0.8, 0.8, 0.2));
      continue;
    }

    bool no_face = (faces.count(h->face) == 0);
    bool not_exterior = not h->face->is_exterior;
    cout<<"no_face: "<<no_face<<endl;
    cout<<"not_exterior: "<<not_exterior<<endl;

    if (no_face and not_exterior) {
      cout<<"add_face"<<endl;
      add_face(h->face, m, nodes, edges, faces, halfedges, viewer);
    }
  }
}

int main(int argc, char **argv) {

  // declaration
  Eigen::MatrixXd V_in, TC_in, N_in, FC_in;
  Eigen::MatrixXi F_in, FTC_in, FN_in;

  Eigen::MatrixXd V_out, TC_out, N_out, FC_out;
  Eigen::MatrixXi F_out, FTC_out, FN_out;

  Eigen::MatrixXd V_tr, C_tr;
  Eigen::MatrixXi F_tr;

  Eigen::RowVector3d center;

  vector<vector<vector<Node *>>> isos_node;  // iso -> loop -> Node
  vector<vector<vector<Face *>>> iso_faces;  // iso -> loop -> Face

  vector<vector<Node*>> segments{}; // for merging
  set<Node*> unvisited{};    // for tracing
  vector<Node*> unvisited_vector{}; // for tracing
  vector<Node*> printing_path{};  // for tracing

  auto mesh = new HalfedgeMesh();
  auto mesh_tr = new HalfedgeMeshTr();

  auto model = new Model(mesh);

  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiMenu menu;



  // param
  enum DisplayMode{DisplayMode_Graph, DisplayMode_Halfedge, DisplayMode_Connection,
    DisplayMode_Segments, DisplayMode_Tracing, DisplayMode_Mesh,
    DisplayMode_InputMesh, DisplayMode_HalfedgeTracing, DisplayMode_MeshTracing};
  int display_mode = DisplayMode_Graph;

  enum TypeFocus{TypeFocus_Node, TypeFocus_Halfedge, TypeFocus_Face, TypeFocus_Edge};
  int type_focus = TypeFocus_Node;
  int idx_focus = 0;

  enum LabelType{LabelType_Node, LabelType_Edge, LabelType_Face, LabelType_Isoline, LabelType_Gradline, LabelType_NumInterp, LabelType_End};
  int label_type = LabelType_Node;

  bool display_label = false;
  bool is_display_stress = false; // 0: spring, 1: stress
  bool is_display_bridge = true;
  bool is_display_background_black = true;

  float smooth_strength = 0.1;
  float shrinkage_cone = 0.28;
  float shrinkage_cone_down = 0.1;
  float radius_cone = 5;
  float radius_trim = 0.45;
  int test = 0;
  int test2 = 0;
  float test3 = 0;
  int filter_threshold = 3;
  int saddle_displacement = 3;

  bool is_flattening = false;
  bool debug = true;

  // const
  Eigen::RowVector3d color_grey = Eigen::RowVector3d(0.5, 0.5, 0.5);
  Eigen::RowVector3d color_white = Eigen::RowVector3d(1., 1., 1.);
  Eigen::RowVector3d color_red = Eigen::RowVector3d(0.8, 0.2, 0.2);
  Eigen::RowVector3d color_green = Eigen::RowVector3d(0.2, 0.8, 0.2);
  Eigen::RowVector3d color_blue = Eigen::RowVector3d(0.2, 0.2, 0.8);
  Eigen::RowVector3d color_magenta = Eigen::RowVector3d(0.8, 0.2, 0.8);
  Eigen::RowVector3d color_cyon = Eigen::RowVector3d(0.2, 0.8, 0.8);
  Eigen::RowVector3d color_yellow = Eigen::RowVector3d(0.8, 0.8, 0.2);
  auto z_shift = Eigen::RowVector3d(0,0, 1e-2);
  float platform_length = 200;
  float platform_width = platform_length;
  float scale_ratio;

  // mapping
  unsigned char *image_in;
  unsigned char *img_out;
  int w_in, h_in, n_c_in; // width, height, n_channel of image_in
  int w_out = 100;
  int n_c_out = 3;

  // ================  new trace param ====================
  float trace_reso = 30.0;
  float trace_edge_len;
  float geodesic_gap = 0.1;
  float geodesy_gap = 0.5;

  // ================  end of new trace param ====================

  ////////////////// helper ////////////////////////

  auto init_draw = [&]() {
    viewer.data().V.resize(0,3);
    viewer.data().F.resize(0,3);
    viewer.data().points.resize(0, 6);
    viewer.data().lines.resize(0, 9);
    viewer.data().labels_positions.resize(0, 3);
    viewer.data().labels_strings.clear();
  };

  auto get_color = [&](float x, bool stress_map = true) {
    double r, g, b;

    const double rone = 0.8;
    const double gone = 1.0;
    const double bone = 1.0;

    if (x == -1) return Eigen::RowVector3d(1.0, 1.0, 1.0);


    x = x / 0.33;
    if (stress_map) {
      if (x > 1) x = 1;
      if (x < -1) x = -1;
      x = (x + 1) / 2;
    }
    else {
      if (x > geodesic_gap) x = x - geodesic_gap;
      if (x > geodesy_gap) x = 1.0;
      else x = 0;
    }

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

  auto assign_flat_mesh =[&]() {

    V_out.resize(mesh->nodes.size(), 3);
    F_out.resize(mesh->faces.size()-1, 3);

    for (auto n : mesh->nodes) {
      V_out.row(n->idx) << n->pos;
    }

    for (auto f : mesh->faces) {
      if (f->is_external) continue;

      int i0 = f->halfedge->node->idx;
      int i1 = f->halfedge->next->node->idx;
      int i2 = f->halfedge->next->next->node->idx;
      F_out.row(f->idx) << i0, i1, i2;
      if (debug) {
        cout<<"face_id: " << f->idx<<endl;
        cout<<f->centroid()<<endl;
      }
    }
  };

  auto redraw = [&]() {
    cout<<"start redraw...."<<endl;

//    viewer.data().clear();  // this line causes bug
    viewer.data().V.resize(0,3);
    viewer.data().F.resize(0,3);
    viewer.data().points.resize(0, 6);
    viewer.data().lines.resize(0, 9);
    viewer.data().labels_positions.resize(0, 3);
    viewer.data().labels_strings.clear();

    viewer.data().label_color << 0.5, 0.5, 0.5;
    viewer.data().point_size = 10;

    if (display_label) {
      switch (display_mode) {
        case DisplayMode_Segments:
          for (auto n : mesh->nodes) {
            viewer.data().add_label(n->pos, to_string(n->i_unvisited_vector));
          }
          break;

        case DisplayMode_Tracing:
          for (auto n : mesh->nodes) {
            viewer.data().add_label(n->pos, to_string(n->i_unvisited_vector));
          }
          if (label_type == LabelType_NumInterp) {
            for (auto n : printing_path) {
              viewer.data().add_label(n->pos, to_string(n->is_end));
            }
          }
          break;

        case DisplayMode_Halfedge:
            switch (label_type) {
              case LabelType_Node:
                for (auto n : mesh->nodes) {
                  viewer.data().add_label(n->pos, to_string(n->idx));
                }
                break;

              case LabelType_Edge:
                for (auto e : mesh->edges) {
                  viewer.data().add_label(e->centroid(), to_string(e->idx));
                }
                break;

              case LabelType_Face:
                for (auto f : mesh->faces) {
                  viewer.data().add_label(f->centroid(), to_string(f->idx));
                }
                break;

              case LabelType_Isoline:
                for (auto n : mesh->nodes) {
                  viewer.data().add_label(n->pos, to_string(n->idx_iso));
                }
                break;

              case LabelType_Gradline:
                for (auto n : mesh->nodes) {
                  viewer.data().add_label(n->pos, to_string(n->idx_grad));
                }
                break;

              case LabelType_NumInterp:
                for (auto f : mesh->faces) {
                  viewer.data().add_label(f->centroid(), to_string(f->num_interp));
                }
                break;

            }
          break;

        case DisplayMode_Graph:
          for (auto n : mesh->nodes) {
            viewer.data().add_label(n->pos, to_string(n->idx));
          }
      }
    }

    switch (display_mode) {
      case DisplayMode_Graph: {  // graph
        for (auto n : mesh->nodes) {
          if (n->idx == idx_focus)
            viewer.data().add_points(n->pos, Eigen::RowVector3d(0.9, 0.9, 0));
          else
            viewer.data().add_points(n->pos, Eigen::RowVector3d(0, 0.9, 0.9));
          if (n->right) viewer.data().add_edges(n->pos, n->right->pos, Eigen::RowVector3d(0.9, 0, 0));
          if (n->up) viewer.data().add_edges(n->pos, n->up->pos, Eigen::RowVector3d(0, 0.9, 0));
        }
      }
        break;

      case DisplayMode_Mesh: {  // mesh
        assign_flat_mesh();
        viewer.data().set_mesh(V_out, F_out);

        FC_out.resize(F_out.rows(), 3);
        for (int i = 0; i < F_out.rows(); i++) {
          Face *f = mesh->faces[i];
          Eigen::RowVector3d color;
          color << 0, 0, 0;
          for (int i_n = 0; i_n < 3; i_n++) {
            Node *n = f->node(i_n);
            color += n->color;
          }
          color /= 3;
          color << test3, test3, test3;
          FC_out.row(i) = color;
        }

        //      FC = FC * 0.8;
        viewer.data().set_colors(FC_out);
        igl::per_face_normals(V_out, F_out, N_out);  // might be redundent
      }
//        break;

      case DisplayMode_Halfedge: { // halfedge
        Eigen::MatrixXd p1s(0, 3);
        Eigen::MatrixXd p2s(0, 3);
        Eigen::MatrixXd colors(0, 3);

        for (auto e : mesh->edges) {
          if ((not is_display_bridge) and (e->spring == "bridge")) continue;

          Node *n_a = *e->nodes.begin();
          Node *n_b = *next(e->nodes.begin(), 1);

          p1s.conservativeResize(p1s.rows() + 1, p1s.cols());
          p1s.row(p1s.rows() - 1) = n_a->pos;
          p2s.conservativeResize(p2s.rows() + 1, p2s.cols());
          p2s.row(p2s.rows() - 1) = n_b->pos;
          colors.conservativeResize(colors.rows() + 1, colors.cols());

          if (is_display_stress) {
            e->shrinkage = 1.0 - e->len_3d / e->length();   // shrinkage ratio
            Eigen::RowVector3d color_stress = get_color(e->shrinkage);
            colors.row(colors.rows() - 1) = color_stress;
          } else {
            if (n_a->on_saddle_boundary and n_b->on_saddle_boundary) {
              colors.row(colors.rows() - 1) = color_cyon;
            } else if (e->spring == "stretch") {
              colors.row(colors.rows() - 1) = color_red;
            } else if (e->spring == "boundary") {
              colors.row(colors.rows() - 1) = color_blue;
            } else if (not((not is_display_bridge) and (e->spring == "bridge"))) {
              colors.row(colors.rows() - 1) = color_green;
            }
          }
        }
        viewer.data().add_edges(p1s, p2s, colors);

        {  // draw focus
          Eigen::RowVector3d color_node;
          if (is_display_background_black) color_node = color_white;
          else color_node = color_grey;

          switch (type_focus) {
            case TypeFocus_Node: {
              if (display_mode == DisplayMode_Segments) {
                viewer.data().add_points(unvisited_vector[idx_focus]->pos, color_node);
              } else {
                viewer.data().add_points(mesh->nodes[idx_focus]->pos, color_node);
              }
            }
              break;

            case TypeFocus_Edge: {
              viewer.data().add_points(mesh->edges[idx_focus]->centroid(), color_node);
            }
              break;

            case TypeFocus_Face: {
              if (model->num_iter == 0)
                viewer.data().add_points(mesh->faces[idx_focus]->centroid(), color_node);
              else if (model->num_iter == 1)
                viewer.data().add_points(mesh->faces[idx_focus]->e_bridge_right->centroid(), color_node);
              else if (model->num_iter == 2)
                viewer.data().add_points(mesh->faces[idx_focus]->e_bridge_left->centroid(), color_node);
              else if (model->num_iter == 3)
                viewer.data().add_points(mesh->faces[idx_focus]->n_bridge->pos, color_node);
              else if (model->num_iter == 4)
                viewer.data().add_points(mesh->faces[idx_focus]->pos_stretch_mid, color_node);
              else if (model->num_iter == 5)
                viewer.data().add_points(mesh->faces[idx_focus]->e_bridge_left->nodes_interp_as_left[int(mesh->gap_size)]->pos,
                                         color_node);
              else if (model->num_iter == 6)
                viewer.data().add_points(mesh->faces[idx_focus]->e_bridge_right->nodes_interp_as_right[int(mesh->gap_size)]->pos,
                                         color_node);
              else if (model->num_iter == 7)
                viewer.data().add_points(mesh->faces[idx_focus]->pos_interps[int(mesh->gap_size)], color_node);
            }
              break;

            case TypeFocus_Halfedge: {
              viewer.data().add_points(mesh->halfedges[idx_focus]->edge->centroid(), color_node);
              viewer.data().add_points(mesh->halfedges[idx_focus]->node->pos, color_node);
            }
              break;
          }
        }
      }
        break;

      case DisplayMode_Connection: { // connection
        for (auto n : unvisited_vector) {
          if (n->right) {
            viewer.data().add_edges((n->right->pos - n->pos) / 3 + n->pos, n->pos, color_green);
          }
          if (n->left) {
            viewer.data().add_edges((n->left->pos - n->pos) / 3 + n->pos, n->pos, color_red);
          }

          if (not n->ups.empty()) {
            for (int i = 0; i < n->ups.size(); i++) {
              viewer.data().add_edges((n->ups[i]->pos - n->pos) / 3 + n->pos, n->pos, color_yellow);
            }

          } else if (n->up) {
            viewer.data().add_edges((n->up->pos - n->pos) / 3 + n->pos, n->pos, color_cyon);
          }

          if (not n->downs.empty()) {
            for (int i = 0; i < n->downs.size(); i++) {
              viewer.data().add_edges((n->downs[i]->pos - n->pos) / 3 + n->pos, n->pos, color_magenta);
            }
          } else if (n->down) {
            viewer.data().add_edges((n->down->pos - n->pos) / 3 + n->pos, n->pos, color_white);
          }

          if (n->right_saddle) {
            viewer.data().add_edges((n->right_saddle->pos - n->pos) / 3 + n->pos, n->pos, color_blue);
          }
          if (n->left_saddle) {
            viewer.data().add_edges((n->left_saddle->pos - n->pos) / 3 + n->pos, n->pos, color_blue);
          }
        }
      }
        break;

      case DisplayMode_Segments: {  // segments
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
          Eigen::MatrixXd p1s_e(mesh->edges.size(), 3);
          Eigen::MatrixXd p2s_e(mesh->edges.size(), 3);
          Eigen::MatrixXd colors_e(mesh->edges.size(), 3);

          int i = 0;
          for (auto e : mesh->edges) {
            Node *n_a = *e->nodes.begin();
            Node *n_b = *next(e->nodes.begin(), 1);

            p1s_e.row(i) = n_a->pos;
            p2s_e.row(i) = n_b->pos;

            if (e->spring == "stretch")
              colors_e.row(i) = color_red / 3;
            else if (e->spring == "boundary")
              colors_e.row(i) = color_blue / 3;
            else if (not((not is_display_bridge) and (e->spring == "bridge")))
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
        break;

      case DisplayMode_Tracing: {  // tracing
        for (auto n : unvisited_vector) {
//        viewer.data().add_points(n->pos, color_blue);
        }

        Eigen::MatrixXd p1s(printing_path.size() - 1, 3);
        Eigen::MatrixXd p2s(printing_path.size() - 1, 3);
        Eigen::MatrixXd colors(printing_path.size() - 1, 3);

        for (int i = 0; i < printing_path.size() - 1; i++) {
          Eigen::RowVector3d color = get_color(printing_path[i + 1]->shrinkage, true);

          p1s.row(i) = printing_path[i]->pos;
          p2s.row(i) = printing_path[i + 1]->pos;
          colors.row(i) = color;

//        viewer.data().add_edges(printing_path[i]->pos, printing_path[i+1]->pos, color);
        }

        viewer.data().add_edges(p1s, p2s, colors);
      }
        break;


      case DisplayMode_InputMesh: {  // input mesh
        viewer.data().set_mesh(V_in, F_in);
        viewer.data().set_colors(FC_in);
        igl::per_face_normals(V_in, F_in, N_in);  // might be redundent
      }
        break;

      case DisplayMode_HalfedgeTracing: {
        // halfedge
        Eigen::MatrixXd p1s(0, 3);
        Eigen::MatrixXd p2s(0, 3);
        Eigen::MatrixXd colors(0, 3);

        for (auto e : mesh_tr->edges) {
          NodeTr *n_a = e->halfedge->node;
          NodeTr *n_b = e->halfedge->twin->node;

          p1s.conservativeResize(p1s.rows() + 1, p1s.cols());
          p1s.row(p1s.rows() - 1) = n_a->pos;
          p2s.conservativeResize(p2s.rows() + 1, p2s.cols());
          p2s.row(p2s.rows() - 1) = n_b->pos;
          colors.conservativeResize(colors.rows() + 1, colors.cols());
          colors.row(colors.rows() - 1) = color_cyon;
        }
        viewer.data().add_edges(p1s, p2s, colors);

        {  // draw focus
          Eigen::RowVector3d color_node;
          color_node = color_white;

          switch (type_focus) {
            case TypeFocus_Node:
              viewer.data().add_points(mesh_tr->nodes[idx_focus]->pos, color_node);
              break;

            case TypeFocus_Edge:
              viewer.data().add_points(mesh_tr->edges[idx_focus]->centroid(), color_node);
              break;

            case TypeFocus_Face:
              viewer.data().add_points(mesh_tr->faces[idx_focus]->centroid(), color_node);
              break;

            case TypeFocus_Halfedge:
              viewer.data().add_points(mesh_tr->halfedges[idx_focus]->edge->centroid(), color_node);
              viewer.data().add_points(mesh_tr->halfedges[idx_focus]->node->pos, color_node);
              break;

          }

        }
      }
        break;

      case DisplayMode_MeshTracing: {
        viewer.data().set_mesh(V_tr, F_tr);
        viewer.data().set_colors(C_tr);
        igl::per_face_normals(V_in, F_in, N_in);  // might be redundent
      }
        break;
    }

  };

  auto get_convergence = [&]() {
    double sum = 0;
    for (auto e : mesh->edges) {
      sum += abs(e->len - e->len_prev) / e->len_prev;
    }
    return ( sum / mesh->edges.size() );
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



  // hit the vertical vector originated from n to the mesh V_in, F_in
  // return the color from FC_in of the face hitted by the vector
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
    for (auto n : mesh->nodes) {
      n->color = hit(n);
    }
  };

  auto smooth = [&]() {
    for (auto n : mesh->nodes) {
      if ((not n->left) or (not n->right)) continue;
      if (n->idx_iso == -1) continue;
      if (not(n->left->idx_iso == n->right->idx_iso and n->right->idx_iso == n->idx_iso)) {
        cout << "error: left, right, iso different, node->idx " << n->idx << endl;
      }
      Edge *el;
      Edge *er;
      Halfedge *h;
      Halfedge *h0;
      int count = 0;

      h0 = n->halfedge;
      h = n->halfedge;
      do {
        if (h->twin->node == n->left) el = h->edge;   // left edge
        if (h->twin->node == n->right) er = h->edge;  // right edge

        h = h->twin->next;
        count++;
        if (count > 20) {
          cout << "error: n: " << n->idx << endl;
          break;
        }
      } while (h != h0);

      Eigen::RowVector3d v_l2r = n->right->pos - n->left->pos;
      float t = er->shrinkage - el->shrinkage;
      n->pos += v_l2r * t * smooth_strength;
    }

  };

  auto autoscale = [&]() {
    double x_max = -99999;
    double x_min = 99999;
    double y_max = -99999;
    double y_min = 99999;
    double z_max = -99999;
    double z_min = 99999;
    double x_span, y_span, z_span;

    for (auto n : mesh->nodes) {
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
    cout << "span: " << x_span << " " << y_span << endl;
    scale_ratio = min((platform_length - 1) / x_span, (platform_width - 1) / y_span);
    Eigen::RowVector3d translation = Eigen::RowVector3d((x_max + x_min) / 2, (y_max + y_min) / 2, 0);
    for (auto n : mesh->nodes) {
      n->pos -= translation;
      n->pos *= scale_ratio;
      n->pos_origin -= translation;
      n->pos_origin *= scale_ratio;
    }

    for (auto e : mesh->edges) {
      e->len_3d *= scale_ratio;
    }
  };

  auto interpolate = [&]() {
    int i_iso_max = 0;
    for (auto n : mesh->nodes) {
      if (n->idx_iso > i_iso_max) i_iso_max = n->idx_iso;
    }

    // isos_node
    for (int iso_iter = 0; iso_iter <= i_iso_max; iso_iter++) {
      vector<vector<Node *>> loops_node;
      while (true) {
        vector<Node *> loop_node;

        bool found = false;
        Node *n_iter = nullptr;
        for (auto n : mesh->nodes) {
          if (n->idx_iso == iso_iter and (not n->visited_interpolation)
              and (not n->on_saddle_boundary)) {
            n_iter = n;   // begin with any unvisited node in any loop of the iso
            found = true;
            break;
          }
        }
        if (not found) break;

        Node *n_begin = n_iter;
        do {
          n_iter->visited_interpolation = true;
          loop_node.push_back(n_iter);
          if (n_iter->left) n_iter = n_iter->left;
          else cout << "err: n_iter->left cannot be found";
        } while (n_iter != n_begin);
        loops_node.push_back(loop_node);
      }
      isos_node.push_back(loops_node);
    }

    // isos_face
    for (int iso_iter = 0; iso_iter <= i_iso_max; iso_iter++) {
      vector<vector<Face *>> loops_face;

      while (true) {
        vector<Face *> loop_face;
        Face *f_iter;

        // find a random face of an iso value
        bool found = false;
        for (auto f : mesh->faces) {
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
          f_iter->visited_interpolation = true;
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
          double len_bridge = 0;
          double len_stretch = 0;
          Node *n_iter = f->n_bridge;
          // vec_bridge
          {
            if (n_iter->is_cone) {
              f->vec_bridge = Eigen::RowVector3d(0, 0, 0);
            } else {
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
            f->num_interp = int(
              iso_distance / (2 * mesh->gap_size));  // double mesh->gap_size first, double num after smoothing

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
        int iso_faces_i_size = iso_faces[i_iso].size();
        for (int i_iso_faces_loop=0; i_iso_faces_loop<iso_faces_i_size; i_iso_faces_loop++) {    // for each loop with same iso
          auto iso_faces_loop = iso_faces[i_iso][i_iso_faces_loop];
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
      int iso_faces_i_size = iso_faces[i_iso].size();
      for (int i_loop=0; i_loop<iso_faces_i_size; i_loop++) {
        auto iso_faces_loop = iso_faces[i_iso][i_loop];
        for (int i_f=0; i_f<iso_faces_loop.size(); i_f++) {
          auto f = iso_faces_loop[i_f];

          // interpolate nodes
          for (int i_interp = 0; i_interp < f->num_interp; i_interp++) {
            double weight_bridge = float(i_interp + 1) / (f->num_interp + 1);
            double weight_stretch = 1.0 - weight_bridge;

            // fix
            // Eigen::RowVector3d vec_interp = weight_bridge * f->vec_bridge + weight_stretch * f->vec_stretch;
            Eigen::RowVector3d vec_interp = f->n_stretch_left->pos - f->n_stretch_right->pos;
            vec_interp.normalize();

            Eigen::RowVector3d pos_interp =
              weight_bridge * f->n_bridge->pos + weight_stretch * f->pos_stretch_mid;
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

  };

  auto merge = [&]() {

    for (auto n : mesh->nodes) {
      n->i_unvisited_vector = unvisited_vector.size();
      unvisited_vector.push_back(n);
      unvisited.emplace(n);
    }

    // initialize
    for (auto f : mesh->faces) {
      if (f->is_saddle) continue;
      if (f->is_external) continue;

      Edge *e = f->e_bridge_left;

      Face *f_left = (e->halfedge->face == f) ?
                     e->halfedge->twin->face :
                     e->halfedge->face;
      f->left = f_left;
      f_left->right = f;

      Halfedge *h = (e->halfedge->face == f) ?
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
        } else if (exist_as_left) {
          n_mid->pos = n_as_left->pos;
          n_mid->right = n_as_left->right;
          n_mid->right->left = n_mid;
          n_mid->is_end = true;
          n_mid->is_left_end = true;
        } else if (exist_as_right) {
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


    cout << "connect left and right" << endl;
    // connect left and right
    for (auto f : mesh->faces) {
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

        Node *n_left = f->e_bridge_left->nodes_interp[i_left];
        Node *n_right = f->e_bridge_right->nodes_interp[i_right];

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

        vector<Node *> segment{};
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

    cout << "connect up and down within nodes_interp" << endl;
    // connect up and down within nodes_interp
    for (auto f : mesh->faces) {
      if (f->is_external) continue;
      if (f->is_saddle) continue;

      Edge *e = f->e_bridge_left;
      Halfedge *h = e->halfedge;
      if (h->node->idx_iso == -1 or h->node->idx_iso == h->twin->node->idx_iso + 1) {
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

    cout << "connect ups and downs" << endl;
    // connect ups and downs
    for (auto n : mesh->nodes) {
      if (n->is_cone or n->is_saddle) { // summit / saddle
        for (auto h : get_halfedges_of_node(n)) {
          if (h->edge->nodes_interp.size() > 0) {
            Node *n_top = h->edge->nodes_interp[h->edge->nodes_interp.size() - 1];
            n_top->up = n;
            n->downs.push_back(n_top);
          } else {
            n->downs.push_back(h->twin->node);
            h->twin->node->ups.push_back(n);
          }
        }
      } else {
        for (auto h : get_halfedges_of_node(n)) {
          if (h->twin->node->is_cone or h->twin->node->idx_iso > n->idx_iso) {
            if (h->edge->nodes_interp.size() > 0) {
              n->ups.push_back(h->edge->nodes_interp[0]);
              h->edge->nodes_interp[0]->down = n;
            } else {
              n->ups.push_back(h->twin->node);
            }
          } else if (h->twin->node->idx_iso == n->idx_iso) {
            continue;
          } else if (h->twin->node->idx_iso < n->idx_iso) {
            if (h->edge->nodes_interp.size() > 0) {
              n->downs.push_back(h->edge->nodes_interp[h->edge->nodes_interp.size() - 1]);
              h->edge->nodes_interp[h->edge->nodes_interp.size() - 1]->up = n;
            } else {
              n->downs.push_back(h->twin->node);
            }
          }
        }
      }
    }

    cout << "connect two end nodes" << endl;
    // connect two end nodes of the double back
    for (auto f : mesh->faces) {
      if (f->is_external) continue;
      if (f->is_saddle) continue;

      if (f->e_bridge_left->nodes_interp_as_left.size() != f->e_bridge_left->nodes_interp_as_right.size()) {
        int num_interp_new = f->e_bridge_left->nodes_interp.size() - 1;
        for (int i_interp = 0; i_interp < num_interp_new; i_interp++) {
          Node *n_up;
          Node *n_down;
          if (f->is_up) {
            n_up = f->n_bridge;
            n_down = f->n_stretch_left;
          } else {
            n_up = f->n_stretch_left;
            n_down = f->n_bridge;
          }

          f->e_bridge_left->nodes_interp[i_interp]->pos =
            n_down->pos * (num_interp_new - i_interp) / (num_interp_new + 1)
            + n_up->pos * (i_interp + 1) / (num_interp_new + 1);
        }
        f->e_bridge_left->nodes_interp[num_interp_new]->pos =
          f->e_bridge_left->nodes_interp[num_interp_new - 1]->pos;

      }
    }


  };

  auto trace = [&]() {
    viewer.data().V.resize(0, 3);
    viewer.data().F.resize(0, 3);
    viewer.data().points.resize(0, 6);
    viewer.data().lines.resize(0, 9);
    viewer.data().labels_positions.resize(0, 3);
    viewer.data().labels_strings.clear();

    Node *n_iter;
    int iso_min;

    while (not unvisited.empty()) {
      // find a random node on the lowest loop
      {
        iso_min = 1e8;
        for (auto n : mesh->nodes) {
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

        unvisited.erase(n_iter);
        n_iter->i_path = printing_path.size();
        printing_path.push_back(n_iter);
        n_iter->shrinkage = -1;
      }

      // one trace
      while (true) {
        // one loop
        while (true) {
          bool move_horizontally = false;
          Node *n_up_a = n_iter;
          Node *n_down_a = n_iter;
          Node *n_up_b;
          Node *n_down_b;

          // move or stop
          {
            if (n_iter->is_end) { // at the end of one segment
              if (n_iter->left and unvisited.count(n_iter->left)) {   // move left
                n_iter = n_iter->left;
                visit(n_iter);
                move_horizontally = true;
              } else if (n_iter->right and unvisited.count(n_iter->right)) {    // move right
                n_iter = n_iter->right;
                visit(n_iter);
                move_horizontally = true;
              } else {
                if (n_iter->up and n_iter->up->is_end and
                    unvisited.count(n_iter->up)) {    // move up to next double back segment
                  n_iter = n_iter->up;
                  visit(n_iter);
                } else {  // move down
                  Node *n_down = n_iter->down;
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
            } else {  // not end
              bool has_double_back = false;
              if (n_iter->is_interp_bridge) {
                if (n_iter->up and n_iter->up->is_end and
                    unvisited.count(n_iter->up)) {  // move up into double back
                  unvisited.emplace(n_iter);  // put back the beginning of the double back
                  n_iter = n_iter->up;
                  visit(n_iter);
                  has_double_back = true;
                }
              } else {
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
                } else if (n_iter->right and unvisited.count(n_iter->right)) {    // move right
                  n_iter = n_iter->right;
                  visit(n_iter);
                  move_horizontally = true;
                } else {   // move up
                  Node *n_up;
                  if (n_iter->is_interp_bridge) {
                    n_up = n_iter->up;
                  } else {
                    if (n_iter->ups.size() > 0) {
                      n_up = n_iter->ups[0];
                    } else {
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
            } else {
              n_up_b = n_iter;
              n_down_b = n_iter;

              while (n_up_a->idx == -1) {
//                    if (n_up_a->is_cone or n_up_a->is_saddle) break;
                if (not n_up_a->up) {
                  cout << "not n_up_a->up " << n_up_a->i_unvisited_vector << endl;
                  getchar();
                  break;
                }
                n_up_a = n_up_a->up;
              }

              while (n_down_a->idx == -1) {
                if (not n_down_a->down) {
                  cout << "not n_down_a->up " << n_down_a->i_unvisited_vector << endl;
                  getchar();
                  break;
                }
                n_down_a = n_down_a->down;
              }

              while (n_up_b->idx == -1) {
//                    if (n_up_b->is_cone or n_up_b->is_saddle) break;
                if (not n_up_b->up) {
                  cout << "not n_up_b->up " << n_up_b->i_unvisited_vector << endl;
                  getchar();
                  break;
                }
                n_up_b = n_up_b->up;
              }

              while (n_down_b->idx == -1) {
                if (not n_down_b->down) {
                  cout << "not n_down_b->up " << n_down_b->i_unvisited_vector << endl;
                  getchar();
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
                    cout << "ha, hb..." << endl;
                    getchar();
                    continue;
                  }
                } else if (n_down_a == n_down_b) {
                  if (ha->next == hb->twin) {
                    f_iter = ha->face;
                  } else if (hb->next = ha->twin) {
                    f_iter = hb->face;
                  } else {
                    cout << "hb, ha..." << endl;
                    getchar();
                    continue;
                  }
                } else {
                  cout << "n_up, n_down, not equal" << endl;
                  cout << "n_iter: " << n_iter->i_unvisited_vector << endl;
                  cout << "n_up_a: " << n_up_a->i_unvisited_vector << endl;
                  cout << "n_up_b: " << n_up_b->i_unvisited_vector << endl;
                  cout << "n_down_a: " << n_down_a->i_unvisited_vector << endl;
                  cout << "n_down_b: " << n_down_b->i_unvisited_vector << endl;
                  getchar();
                  break;
                }
              }


              double shrinkage_up;
              double shrinkage_down;
              Halfedge *h_iter;

              double shrinkage_bridge = 0;
              double shrinkage_stretch = 0;
              int n_shrinkage_bridge = 0;
              int n_shrinkage_stretch = 0;
              double len_shrinkage_bridge = 0;
              double len_shrinkage_stretch = 0;

              if (f_iter->n_bridge->left) {
                h_iter = get_halfedge(f_iter->n_bridge->left, f_iter->n_bridge);
                shrinkage_bridge += h_iter->edge->shrinkage * h_iter->length();
                n_shrinkage_bridge++;
                len_shrinkage_bridge += h_iter->length();

                if (f_iter->n_bridge->left->left) {
                  h_iter = get_halfedge(f_iter->n_bridge->left->left, f_iter->n_bridge->left);
                  shrinkage_bridge += h_iter->edge->shrinkage * h_iter->length();
                  n_shrinkage_bridge++;
                  len_shrinkage_bridge += h_iter->length();
                }
              }
              if (f_iter->n_bridge->right) {
                h_iter = get_halfedge(f_iter->n_bridge->right, f_iter->n_bridge);
                shrinkage_bridge += h_iter->edge->shrinkage * h_iter->length();
                n_shrinkage_bridge++;
                len_shrinkage_bridge += h_iter->length();
                if (f_iter->n_bridge->right->right) {
                  h_iter = get_halfedge(f_iter->n_bridge->right->right, f_iter->n_bridge->right);
                  shrinkage_bridge += h_iter->edge->shrinkage * h_iter->length();
                  n_shrinkage_bridge++;
                  len_shrinkage_bridge += h_iter->length();
                }
              }

              shrinkage_stretch += f_iter->e_stretch->shrinkage * f_iter->e_stretch->length();
              n_shrinkage_stretch++;
              len_shrinkage_stretch += f_iter->e_stretch->length();
              if (f_iter->n_stretch_left->left) {
                h_iter = get_halfedge(f_iter->n_stretch_left->left, f_iter->n_stretch_left);
                shrinkage_stretch += h_iter->edge->shrinkage * h_iter->length();
                n_shrinkage_stretch++;
                len_shrinkage_stretch += h_iter->length();
                if (f_iter->n_stretch_left->left->left) {
                  h_iter = get_halfedge(f_iter->n_stretch_left->left->left, f_iter->n_stretch_left->left);
                  shrinkage_stretch += h_iter->edge->shrinkage * h_iter->length();
                  n_shrinkage_stretch++;
                  len_shrinkage_stretch += h_iter->length();
                }
              }
              if (f_iter->n_stretch_right->right) {
                h_iter = get_halfedge(f_iter->n_stretch_right->right, f_iter->n_stretch_right);
                shrinkage_stretch += h_iter->edge->shrinkage * h_iter->length();
                n_shrinkage_stretch++;
                len_shrinkage_stretch += h_iter->length();
                if (f_iter->n_stretch_right->right->right) {
                  h_iter = get_halfedge(f_iter->n_stretch_right->right->right, f_iter->n_stretch_right->right);
                  shrinkage_stretch += h_iter->edge->shrinkage * h_iter->length();
                  n_shrinkage_stretch++;
                  len_shrinkage_stretch += h_iter->length();
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

            unvisited.erase(n_iter);
            n_iter->i_path = printing_path.size();
            printing_path.push_back(n_iter);
            n_iter->shrinkage = shrinkage;
          }

        }

        // move up
        {
          if (n_iter->idx == -1 and not n_iter->is_cone and not n_iter->is_saddle) {
            if (n_iter->up) {
              if (n_iter->up->is_cone or n_iter->up->is_saddle) break;
              n_iter = n_iter->up;
            } else break;
          } else {
            if (not n_iter->ups.empty()) {
              if (n_iter->ups[0]->is_cone or n_iter->ups[0]->is_saddle) break;
              else {
                n_iter = n_iter->ups[0];
              }
            } else {
              cout << "error: ups.empty()" << endl;
              getchar();
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
          n_iter->i_path = printing_path.size();
          printing_path.push_back(n_iter);
          n_iter->shrinkage = -1;

        }
      }
    }

    model->num_iter = printing_path.size();
  };

  auto trace_saddle = [&]() {

    cout << "start trace saddles..." << endl;

    for (auto n_saddle : mesh->saddles) {
      cout << "detect saddle " << n_saddle->idx << endl;
      bool is_first = true;

      // collect halfedges_saddle
      vector<Halfedge *> halfedges_saddle;
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
        f->num_interp = int(iso_distance / (2 * mesh->gap_size)) * 2;

        if (f->num_interp > num_interp_max) num_interp_max = f->num_interp;
      }

      // assign edges and nodes for the face
      for (auto h : halfedges_saddle) {
        Face *f = h->face;
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
        Face *f = h->face;
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
        Edge *e_l = h->prev->prev->edge;
        Edge *e_r = h->edge;
        vector<Node *> nodes_left = e_l->nodes_interp_as_left;
        vector<Node *> nodes_right = e_r->nodes_interp_as_right;

        vector<Node *> path;
        int i = -1;
        bool from_left = true;
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
          } else {
            path.push_back(nodes_right[i]);
            path.push_back(nodes_left[i]);
          }


          i++;
          from_left = not from_left;
        }

        if (not from_bottom) {
          reverse(path.begin(), path.end());
        }

        for (Node *n : path) {
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

      display_mode = DisplayMode_Tracing;
      redraw();

    }
  };

  auto adjust_cone = [&]() {
    for (auto n : printing_path) {
      for (auto nc : mesh->cones) {
        double d = (n->pos - nc->pos).norm();
        if (d < radius_cone) {
          if (n->shrinkage == -1) continue;
          double t = 1 - d / radius_cone;
          double s = t * shrinkage_cone + (1 - t) * shrinkage_cone_down;
          n->shrinkage = s;
        }
      }
    }
  };

  auto trim_center = [&]() {
    vector<Node *> printing_path_new;
    for (auto n : printing_path) {
      bool keep = true;
      for (auto nc :mesh->cones) {
        double d = (n->pos - nc->pos).norm();
        if (d < radius_trim) {
          keep = false;
        }
      }
      for (auto nc :mesh->saddles) {
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
  };

  auto save = [&]() {
    // scale
    std::string ofile_name = igl::file_dialog_save();
    std::ofstream ofile(ofile_name);
    if (ofile.is_open()) {
      for (auto n : printing_path) {
        Eigen::RowVector3d p = n->pos;
        if (n->is_move) n->shrinkage = -1;
        ofile << p[0] << " " << p[1] << " " << n->is_move << " " << n->shrinkage << endl;
      }

      for (auto f : mesh->faces) {
        if (f->is_external) continue;
        ofile << "f " << f->node(0)->pos.x() << " " << f->node(0)->pos.y() << " "
              << f->node(1)->pos.x() << " " << f->node(1)->pos.y() << " "
              << f->node(2)->pos.x() << " " << f->node(2)->pos.y() << " "
              << f->node(0)->color[0] << " " << f->node(0)->color[1] << " " << f->node(0)->color[2] << " "
              << f->node(1)->color[0] << " " << f->node(1)->color[1] << " " << f->node(1)->color[2] << " "
              << f->node(2)->color[0] << " " << f->node(2)->color[1] << " " << f->node(2)->color[2] << " "
              << f->opacity() * scale_ratio * scale_ratio << endl;
      }
    }
  };

  auto save_graph = [&]() {
    std::string outname = igl::file_dialog_save();

    std::ofstream ofile(outname);

    for (auto n : mesh->nodes) {
      ofile << n->idx << " " << n->pos.x() << " " << n->pos.y() << " " << n->pos.z() << " " << n->idx_iso << " " << n->idx_grad << " " << n->right->idx << " " << n->left->idx << " ";
      if (n->up) ofile << n->up->idx;
      else ofile << to_string(-1);
      ofile << " ";
      if (n->down) ofile << n->down->idx;
      else ofile << to_string(-1);
      ofile <<std::endl;
    }

    ofile << "# center "<<std::to_string(center.x())<<" "<<std::to_string(center.y())<<" "<<std::to_string(center.z())<<std::endl;

  };

  auto save_obj = [&]() {
    std::string outname = igl::file_dialog_save();

    igl::writeOBJ(outname, V_out, F_out);

  };


// new tracing ===============================================

  //project all the nodes onto x, y plane
  auto project = [&]() {
    for (auto n : mesh->nodes) {
      n -> pos.z()  = 0;
    }
  };

  // get boundary from the projected mesh, use is_external
  auto compute_boundary = [&]() {
    Halfedge* h0;
    bool found_h0 = false;
    for (auto h : mesh->halfedges) {
      if (h->face->is_external) {
        h0 = h;
        found_h0 = true;
        break;
      }
    }
    if (not found_h0) { cout<<"Error: h0 not found in get_boundary"<<endl; }

    Halfedge* h = h0;
    mesh_tr->boundary.emplace_back(h);
    h = h->next;
    while (h != h0) {
      mesh_tr->boundary.emplace_back(h);
      h = h->next;
    }

  };

  //  input : boundary
  //  output : right triangle halfedge mesh
  auto triangulate_tr = [&]() {
    // create right triangle meshes based on the extrema value of the 2D boundary
    int num_squares = int(trace_reso);

    double x_min = 1e8;
    double x_max = -1e8;
    double y_min = 1e8;
    double y_max = -1e8;

    for (auto h : mesh_tr->boundary) {
      double x = h->node->pos.x();
      double y = h->node->pos.y();
      if (x < x_min) { x_min = x; }
      if (x > x_max) { x_max = x; }
      if (y < y_min) { y_min = y; }
      if (y > y_max) { y_max = y; }
    }

    x_min = x_min - 1e-8;
    y_min = y_min - 1e-8;
    x_max = x_max - 1e-8;
    y_max = y_max - 1e-8;

    auto in_boundary = [&](double x, double y) -> bool {
      // check whether a point is in the boundary or not
      int num_left = 0;
      int num_right = 0;

      for (auto h : mesh_tr->boundary) {
        Node* na = h->node;
        Node* nb = h->twin->node;

        double t = ( y - na->pos.y() ) / ( nb->pos.y() - na->pos.y() );
        if ( t == 0 or t == 1) {cout<<"Warning: mesh->nodes intersected horizontally in in_boundary. "<<endl;}
        if ( t > 0 and t < 1) {
          // intersected
          double x_intersect = na->pos.x() + ( nb->pos.x() - na->pos.x() ) * t;
          if (x_intersect < x) { num_left++; }
          else { num_right++; }
        }
      }

      if ( (num_left % 2 == 1) and (num_right % 2 == 1) ) { return true; }
      else { return false; }
    };

    double len_square = max( x_max - x_min, y_max - y_min ) / num_squares;
    trace_edge_len = len_square;

    int num_px = int( (x_max - x_min) / len_square ) + 1;
    int num_py = int( (y_max - y_min) / len_square ) + 1;

    Eigen::MatrixXd V;
    Eigen::MatrixXd C;
    vector<double> xs;
    vector<double> ys;
    vector<vector<NodeTr*>> nodes_tr_mat;
    vector<vector<vector<EdgeTr*>>> edges_tr_mat;    // i_row, i_col =>

    Eigen::MatrixXi is_point_in(num_py, num_px);  // y(row) first

    // construct nodes_tr_mat
    double x, y;
    for (int i_py = 0; i_py < num_py; i_py++) {
      vector<NodeTr*> nodes_tr_row;
      y = y_min + i_py * len_square;
      for (int i_px = 0; i_px < num_px; i_px++ ) {
        x = x_min + i_px * len_square;
        NodeTr* n = new NodeTr();

        if (in_boundary(x, y)) {
          is_point_in(i_py, i_px) = 1;
          xs.emplace_back(x);
          ys.emplace_back(y);
          n->pos << x, y, 0;
        }
        else {
          is_point_in(i_py, i_px) = 0;
        }
        nodes_tr_row.emplace_back(n);
      }

      nodes_tr_mat.emplace_back(nodes_tr_row);
    }

    // construct edges_tr_mat
    // [row] [column] [edge]
    // edge: 0: left, 1: top, 2: diagnoal(bottom_left -> top_right)
    for (int i_py = 0; i_py < num_py; i_py++) {
      vector<vector<EdgeTr*>> edges_row;
      for (int i_px = 0; i_px < num_px; i_px++) {
        vector<EdgeTr*> edges_square;
        EdgeTr* e0 = new EdgeTr();  // left edge
        EdgeTr* e1 = new EdgeTr();  // top edge
        EdgeTr* e2 = new EdgeTr();  // diagonal edge
        edges_square.push_back(e0);
        edges_square.push_back(e1);
        edges_square.push_back(e2);
        edges_row.push_back(edges_square);
      }
      edges_tr_mat.emplace_back(edges_row);
    }

    // iterate squares to construct halfedge mesh without boundary
    for (int i_px = 0; i_px < num_px - 1; i_px++ ) {
      for (int i_py = 0; i_py < num_py - 1; i_py++ ) {
        NodeTr* n0 = nodes_tr_mat[i_py][i_px];  // top_left node
        NodeTr* n1 = nodes_tr_mat[i_py][i_px+1];  // top_right node
        NodeTr* n2 = nodes_tr_mat[i_py+1][i_px];  // bottom_left node
        NodeTr* n3 = nodes_tr_mat[i_py+1][i_px+1];  // bottom_right node

        EdgeTr* e0 = edges_tr_mat[i_py][i_px][0];  // left
        EdgeTr* e1 = edges_tr_mat[i_py][i_px][1];  // top
        EdgeTr* e2 = edges_tr_mat[i_py][i_px][2];  // diagonal
        EdgeTr* e3 = edges_tr_mat[i_py+1][i_px][1];  // bottom
        EdgeTr* e4 = edges_tr_mat[i_py][i_px+1][0];  // right

        if (is_point_in(i_py + 1, i_px) and is_point_in(i_py, i_px + 1)) {  // diagonal in boundary

          // top-left triangle
          if (is_point_in(i_py, i_px)) {

            viewer.data().add_edges(n0->pos, n1->pos, color_green);
            viewer.data().add_edges(n1->pos, n2->pos, color_green);
            viewer.data().add_edges(n2->pos, n0->pos, color_green);

            auto* face = new FaceTr();
            auto* h0 = new HalfedgeTr();  // pointing from top_left to bottom_left
            auto* h1 = new HalfedgeTr();  // bottom_left to top_right
            auto* h2 = new HalfedgeTr();  // top_right to top_left

            h0->face = face;
            h0->node = n0;
            n0->halfedge = h0;
            h0->next = h1;
            h0->prev = h2;
            h0->edge = e0;
            if (h0->edge->halfedge) {
              h0->twin = h0->edge->halfedge;
              h0->twin->twin = h0;
            }
            else {
              h0->edge->halfedge = h0;
            }
            mesh_tr->halfedges.push_back(h0);

            h1->face = face;
            h1->node = n2;
            n2->halfedge = h1;
            h1->next = h2;
            h1->prev = h0;
            h1->edge = e2;
            if (h1->edge->halfedge) {
              h1->twin = h1->edge->halfedge;
              h1->twin->twin = h1;
            }
            else {
              h1->edge->halfedge = h1;
            }
            mesh_tr->halfedges.emplace_back(h1);

            h2->face = face;
            h2->node = n1;
            n1->halfedge = h2;
            h2->next = h0;
            h2->prev = h1;
            h2->edge = e1;
            if (h2->edge->halfedge) {
              h2->twin = h2->edge->halfedge;
              h2->twin->twin = h2;
            }
            else {
              h2->edge->halfedge = h2;
            }
            mesh_tr->halfedges.emplace_back(h2);

            face->halfedge = h0;
            mesh_tr->faces.emplace_back(face);

          }

          // bottom_right triangle
          if (is_point_in(i_py+1, i_px+1)) {
            viewer.data().add_edges(n3->pos, n1->pos, color_green);
            viewer.data().add_edges(n1->pos, n2->pos, color_green);
            viewer.data().add_edges(n2->pos, n3->pos, color_green);

            auto* face = new FaceTr();
            auto* h3 = new HalfedgeTr();  // top_right to bottom_left
            auto* h4 = new HalfedgeTr();  // bottom_left to bottom_right
            auto* h5 = new HalfedgeTr();  // bottom_right to top_right

            h3->face = face;
            h3->node = n1;
            n1->halfedge = h3;
            h3->next = h4;
            h3->prev = h5;
            h3->edge = e2;
            if (h3->edge->halfedge) {
              h3->twin = h3->edge->halfedge;
              h3->twin->twin = h3;
            }
            else {
              h3->edge->halfedge = h3;
            }
            mesh_tr->halfedges.push_back(h3);

            h4->face = face;
            h4->node = n2;
            n2->halfedge = h4;
            h4->next = h5;
            h4->prev = h3;
            h4->edge = e3;
            if (h4->edge->halfedge) {
              h4->twin = h4->edge->halfedge;
              h4->twin->twin = h4;
            }
            else {
              h4->edge->halfedge = h4;
            }
            mesh_tr->halfedges.emplace_back(h4);

            h5->face = face;
            h5->node = n3;
            n3->halfedge = h5;
            h5->next = h3;
            h5->prev = h4;
            h5->edge = e4;
            if (h5->edge->halfedge) {
              h5->twin = h5->edge->halfedge;
              h5->twin->twin = h5;
            }
            else {
              h5->edge->halfedge = h5;
            }
            mesh_tr->halfedges.emplace_back(h5);

            face->halfedge = h5;
            mesh_tr->faces.emplace_back(face);

          }
        }

      }
    }

    // iterate faces to construct the boundary halfedges except for their nexts and prevs
    auto face_exterior = new FaceTr();
    mesh_tr->face_exterior = face_exterior;
    face_exterior->is_exterior = true;
//    mesh_tr->faces_tr.push_back(face_exterior);
    int halfedges_tr_size = mesh_tr->halfedges.size();
    for (int i=0; i<halfedges_tr_size; i++) {
      HalfedgeTr* h = mesh_tr->halfedges[i];
      if (not h->twin) {
        auto h_twin = new HalfedgeTr();
        h_twin->twin = h;
        h->twin = h_twin;
        h_twin->edge = h->edge;
        h_twin->face = face_exterior;
        if (not face_exterior->halfedge) {
          face_exterior->halfedge = h_twin;
        }
        h_twin->node = h->next->node;
        mesh_tr->halfedges.push_back(h_twin);
      }
    }

    // create nexts and prevs for halfedges on the boundary
    for (auto h : mesh_tr->halfedges) {
      if (not h->next) {
        if (not h->twin) {
        }
        HalfedgeTr* h_n = h->twin;
        while (true) {
          if (not h_n->prev) {
            h->next = h_n;
            h_n->prev = h;
            break;
          }
          h_n = h_n->prev->twin;
        }
      }
      if (not h->prev) {
        HalfedgeTr* h_p = h->twin;
        while (true) {
          if (not h_p->next) {
            h_p->next = h;
            h->prev = h_p;
            break;
          }
          h_p = h_p->next->twin;
        }
      }
    }

    // construct edges_tr & nodes
    for (auto h : mesh_tr->halfedges) {
      EdgeTr* e = h->edge;
      NodeTr* n = h->node;

      if (count(mesh_tr->edges.begin(), mesh_tr->edges.end(), e) == 0) {
        mesh_tr->edges.push_back(e);
      }

      if ( count(mesh_tr->nodes.begin(), mesh_tr->nodes.end(), n) == 0 ) {
        mesh_tr->nodes.push_back(n);
      }
    }

    // label nodes, edges, faces, halfedges
    label_vector<NodeTr*>(mesh_tr->nodes);
    label_vector<EdgeTr*>(mesh_tr->edges);
    label_vector<FaceTr*>(mesh_tr->faces);
    label_vector<HalfedgeTr*>(mesh_tr->halfedges);

    // initialize accepted nodes with boundary
    for (auto h : mesh_tr->halfedges) {
      if (h->face->is_exterior) {
        mesh_tr->nodes_accepted.insert(h->node);
      }
    }

  };

  // input: right triangle halfedge mesh
  // output: mesh: V_tr, F_tr
  auto convert_to_mesh = [&]() {
    V_tr.resize(mesh_tr->nodes.size(), 3);
    F_tr.resize(mesh_tr->faces.size(), 3);
    C_tr.resize(mesh_tr->nodes.size(), 3);

    for (auto n : mesh_tr->nodes) {
      int id_n = n->id;
      V_tr.row(id_n) << n->pos;
      C_tr.row(id_n) << color_red;
    }

    for (auto f : mesh_tr->faces) {

      NodeTr* n0 = f->halfedge->node;
      NodeTr* n1 = f->halfedge->next->node;
      NodeTr* n2 = f->halfedge->prev->node;

      int id_n0 = n0->id;
      int id_n1 = n1->id;
      int id_n2 = n2->id;

      int id_f = f->id;
      F_tr.row(id_f) << id_n0, id_n2, id_n1;
    }

    display_mode = DisplayMode_MeshTracing;
    redraw();
  };

  auto compute_geodesic = [&]() {
    redraw();
    double t = trace_edge_len;   // avg_edge_len

    igl::HeatGeodesicsData<double> data;
    const auto precompute = [&]()
    {
      if(!igl::heat_geodesics_precompute(V_tr,F_tr,t,data))
      {
        std::cerr<<"Error: heat_geodesics_precompute failed."<<std::endl;
        exit(EXIT_FAILURE);
      }
    };
    precompute();

    C_tr = Eigen::MatrixXd::Constant(V_tr.rows(),3,0.5);
    Eigen::VectorXd D = Eigen::VectorXd::Zero(data.Grad.cols());

    Eigen::VectorXi sources(mesh_tr->nodes_accepted.size());
    int i = 0;
    for (auto n : mesh_tr->nodes_accepted) {
      sources(i) = n->id;
      i++;
    }

    cout<<"sources: "<<sources<<endl;
    cout<<"data.b: "<<data.b<<endl;

    igl::heat_geodesics_solve(data, sources, D);

    for (int i = 0; i < mesh_tr->nodes.size(); i++) {
      mesh_tr->nodes[i]->geodesic = D(i) / D.maxCoeff();
    }

    C_tr.col(0) << D/D.maxCoeff();

    viewer.data().set_colors(C_tr);
  };

  auto compute_geodesy = [&]() {
    assign_flat_mesh();
    display_mode = DisplayMode_MeshTracing;
    redraw();

    set<int> ids_iso;
    int idx_iso_max = 0;
    for (auto n : mesh->nodes) {
      if (n->idx_iso > idx_iso_max) idx_iso_max = n->idx_iso;
      ids_iso.insert(n->idx_iso);
      if (n->idx_iso == -1) viewer.data().add_points(n->pos, color_red);
    }

    for (auto i : ids_iso) {
      cout<<i<<endl;
    }

    for (auto n : mesh_tr->nodes) {
      // assign normalized Geodesy value
      // project onto the mesh
      // get the triangle and barycentric coordinate
      // get Geodesy value of three points in the triangle
      // use bary coord to compute the interpolated geodesy value, assign to points
      // update the color to Geodesy Value

      Eigen::RowVector3d s, dir;
      int face_id;
      float u,v;
      s = n->pos;
      s.z() = -1E4;
      dir << 0, 0, 1E4;

      vector<igl::Hit> hits;
      igl::ray_mesh_intersect(s, dir, V_out, F_out, hits);
      if (hits.size() != 1) {
        face_id = hits[0].id;
        cout<<"hits: "<<hits.size()<<endl;
        cout<<"hits[0]: "<<hits[0].id<<endl;
        cout<<"hits[1]: "<<hits[1].id<<endl;
        cout<<F_out.rows()<<endl;
//        break;
      }
      else {
        face_id = hits[0].id;
        u = hits[0].u;
        v = hits[0].v;
        cout<<"face_id: "<<face_id<<endl;
      }

      // get the distance from the barycentric vertex to the bottom segment and compute the value.
      Node* n0 = mesh->nodes[F_out(face_id, 0)];
      Node* n1 = mesh->nodes[F_out(face_id, 1)];
      Node* n2 = mesh->nodes[F_out(face_id, 2)];

      double geodesy0 = double(n0->idx_iso) / idx_iso_max;
      double geodesy1 = double(n1->idx_iso) / idx_iso_max;
      double geodesy2 = double(n2->idx_iso) / idx_iso_max;
      if (debug) {
        if (n->id == model->num_iter) {
          viewer.data().add_points(n->pos, color_green);

          viewer.data().add_points(mesh->nodes[F_out(hits[0].id, 0)]->pos, color_blue);
          viewer.data().add_points(mesh->nodes[F_out(hits[0].id, 1)]->pos, color_blue);
          viewer.data().add_points(mesh->nodes[F_out(hits[0].id, 2)]->pos, color_blue);

          if (hits.size() != 1) {
            cout<<"hits: "<<hits.size()<<endl;

            viewer.data().add_points(mesh->nodes[F_out(hits[1].id, 0)]->pos, color_cyon);
            viewer.data().add_points(mesh->nodes[F_out(hits[1].id, 1)]->pos, color_cyon);
            viewer.data().add_points(mesh->nodes[F_out(hits[1].id, 2)]->pos, color_cyon);
          }
        }
      }

      if (geodesy0 < 0) geodesy0 = geodesy1;
      if (geodesy1 < 0) geodesy1 = geodesy2;
      if (geodesy2 < 0) geodesy2 = geodesy0;

      Eigen::RowVector3d pos_hit = u * n1->pos + v * n2->pos + (1 - u - v) * n0->pos;
      Eigen::RowVector3d iso_edge_src;  // source point of the vector of the iso edge
      Eigen::RowVector3d iso_edge_vec;
      Eigen::RowVector3d tip_pos; // the third vertex position
      double geodesy_edge, geodesy_vertex;

      if (geodesy0 == geodesy1) {
        iso_edge_src = n0->pos;
        iso_edge_vec = n1->pos - n0->pos;
        tip_pos = n2->pos;
        geodesy_vertex = geodesy2;
        geodesy_edge = geodesy0;
      }
      else if (geodesy1 == geodesy2) {
        iso_edge_src = n1->pos;
        iso_edge_vec = n2->pos - n1->pos;
        tip_pos = n0->pos;
        geodesy_vertex = geodesy0;
        geodesy_edge = geodesy1;
      }
      else if (geodesy2 == geodesy0) {
        iso_edge_src = n2->pos;
        iso_edge_vec = n0->pos - n2->pos;
        tip_pos = n1->pos;
        geodesy_vertex = geodesy1;
        geodesy_edge = geodesy0;
      }
      else {
        cout<<"no equal edge"<<endl;
        viewer.data().add_points(pos_hit, color_white);
      }

      double height = (tip_pos - iso_edge_src).dot(iso_edge_vec) / iso_edge_vec.norm();
      double dist = (pos_hit - iso_edge_src).dot(iso_edge_vec) / iso_edge_vec.norm();
      n->geodesy = dist / height * geodesy_vertex + (1 - dist / height) * geodesy_edge;

      n->geodesy = u * geodesy1 + v * geodesy2 + (1-u-v) * geodesy0;
    }

    cout<<"nodes_tr_size: "<<mesh_tr->nodes.size()<<endl;
    cout<<"C_tr size: "<<C_tr.rows()<<endl;
    cout<<"V_tr size:"<<V_tr.rows()<<endl;

    for (int i = 0; i < mesh_tr->nodes.size(); i++) {
      C_tr(i,0) = mesh_tr->nodes[i]->geodesy;
    }
    viewer.data().set_colors(C_tr);
  };

  auto compute_geodesy_line = [&](double geodesy_gap) {
    // TODO: might have bug
    vector<NodeTr_trace*> nodes_tr_geodesy;

    for (auto e : mesh_tr->edges) {
      e->node_trace = nullptr;

      NodeTr* n0 = e->halfedge->node;
      NodeTr* n1 = e->halfedge->twin->node;

      double t = (geodesy_gap - n0->geodesy) / (n1->geodesy - n0->geodesy);

      if ((t >= 0) and (t < 1)) {
        auto n = new NodeTr_trace();
        n->pos = n0->pos + t * (n1->pos - n0->pos);
        n->edge = e;
        e->node_trace = n;
        nodes_tr_geodesy.push_back(n);

      }
    }

    auto get_left_right = [&](HalfedgeTr* h, NodeTr_trace* n) {
      if (h->edge->node_trace) {
        NodeTr_trace* n_n = h->edge->node_trace;
        if (h->node->geodesy > h->twin->node->geodesy) {
          n->left = n_n;
          n_n->right = n;
        } else {
          n->right = n_n;
          n_n->left = n;
        }
      }
    };

    for (auto n : nodes_tr_geodesy) {
      n->left = nullptr;
      n->right = nullptr;

      vector<HalfedgeTr*> hs;
      hs.push_back(n->edge->halfedge->next);
      hs.push_back(n->edge->halfedge->prev);
      hs.push_back(n->edge->halfedge->twin->next);
      hs.push_back(n->edge->halfedge->twin->prev);

      for (auto h : hs) {
        get_left_right(h, n);
      }

      if (n->left) {
        viewer.data().add_edges(n->left->pos, n->pos, color_cyon);
      }
      if (n->right) {
        viewer.data().add_edges(n->right->pos, n->pos, color_cyon);
      }

    }


  };

  auto compute_geodesic_line = [&](double geodesic, double geodesy) {
    // only keep one geodesic trace each time
    // change nodes_tr_trace to nodes_tr_cycle

    mesh_tr->nodes_next.clear();
    for (auto e : mesh_tr->edges) {
      e->node_cycle = nullptr;
    }

    for (auto e : mesh_tr->edges) {
      e->node_trace = nullptr;

      NodeTr* n0 = e->halfedge->node;
      NodeTr* n1 = e->halfedge->twin->node;

      double t = (geodesic - n0->geodesic) / (n1->geodesic - n0->geodesic);

      if ((t >= 0) and (t < 1)) {
        double geodesy_n = t * n1->geodesy + (1-t) * n0->geodesy;
        if (geodesy_n < geodesy) {
          auto n = new NodeTr_cycle();
          n->t = t;
          n->halfedge = e->halfedge;
          e->node_cycle = n;
          mesh_tr->nodes_next.push_back(n);
          viewer.data().add_points(n->pos(), color_white);
        }
      }
    }

    auto get_left_right = [&](HalfedgeTr* h, NodeTr_cycle* n) {
      if (h->edge->node_cycle) {
        NodeTr_cycle* n_n = h->edge->node_cycle;
        if (h->node->geodesic > h->twin->node->geodesic) {
          n->left = n_n;
          n_n->right = n;
        } else {
          n->right = n_n;
          n_n->left = n;
        }
      }
    };

    for (auto n : mesh_tr->nodes_next) {
      vector<HalfedgeTr*> hs;
      hs.push_back(n->halfedge->next);
      hs.push_back(n->halfedge->prev);
      hs.push_back(n->halfedge->twin->next);
      hs.push_back(n->halfedge->twin->prev);

      for (auto h : hs) {
        get_left_right(h, n);
      }

    }
  };

  auto draw_geodesic_lines = [&]() {
    redraw();

    int num_edges = 0;
    for (auto n : mesh_tr->nodes_next) {
      if (n->left) {
        num_edges++;
      }
    }

    Eigen::MatrixXd points_a(num_edges, 3);
    Eigen::MatrixXd points_b(num_edges, 3);
    Eigen::MatrixXd colors_edge(num_edges, 3);
    int i = 0;
    for (auto n : mesh_tr->nodes_next) {
      if (n->left) {
        points_a.row(i) = n->pos();
        points_b.row(i) = n->left->pos();
        colors_edge.row(i) = color_blue;
        i++;
      }
    }
    viewer.data().add_edges(points_a, points_b, colors_edge);
  };

  auto trim_mesh = [&]() {
    /* input:
     *  halfedges_tr
     *  edges_tr
     *  nodes
     *  faces_tr
     *  trace_tr: NodeOnHalfedge (*Edge, )
    */

    // TODO : halfedgemesh might have multiple cycles later, and do not need to assure the cycle is closed
    auto is_cycle_closed = [&]() {
      int num_nodes_in_cycle = 0;

      if (mesh_tr->nodes_next.size() == 0) return false;
      auto n = mesh_tr->nodes_next[0];
      auto n0 = n;
      while (true) {
        if (not n->right) return false;
        n = n->right;
        num_nodes_in_cycle++;
        if (n == n0) {
          if (num_nodes_in_cycle == mesh_tr->nodes_next.size()) return true;
          else return false;
        }
      }
    };
    cout<<"closed cycle: "<<is_cycle_closed()<<endl;

    NodeTr_cycle* n = mesh_tr->nodes_next[0];
    auto n_init = n;
    NodeTr_cycle* n_next = n->right;

    NodeTr_cycle* n_left;
    NodeTr_cycle* n_right;
    NodeTr* n0;
    NodeTr* n1;
    NodeTr* n2;
    NodeTr* n3;
    NodeTr* n4;
    EdgeTr* e0;
    EdgeTr* e1;
    EdgeTr* e2;
    EdgeTr* e3;
    EdgeTr* e4;
    EdgeTr* e5;
    EdgeTr* e6;
    EdgeTr* e7;
    EdgeTr* e8;
    HalfedgeTr* h0;
    HalfedgeTr* h1;
    HalfedgeTr* h2;
    HalfedgeTr* h3;
    HalfedgeTr* h4;
    HalfedgeTr* h5;
    HalfedgeTr* h6;
    HalfedgeTr* h7;
    HalfedgeTr* h8;
    HalfedgeTr* h9;
    HalfedgeTr* h10;
    HalfedgeTr* h11;
    HalfedgeTr* h12;
    HalfedgeTr* h13;
    HalfedgeTr* h14;
    HalfedgeTr* ha;
    HalfedgeTr* hb;
    HalfedgeTr* ha_init;
    HalfedgeTr* hb_init;
    FaceTr* f0;
    FaceTr* f1;
    FaceTr* f2;
    FaceTr* f3;

    auto trim_triangle = [&](HalfedgeTr* h11, HalfedgeTr* h12,
      HalfedgeTr* h13, HalfedgeTr* h14) {
      {
        bool n_nn = (n->halfedge->next == n_next->halfedge);
        bool n_nnt = (n->halfedge->next == n_next->halfedge->twin);
        bool nt_nn = (n->halfedge->twin->next == n_next->halfedge);
        bool nt_nnt = (n->halfedge->twin->next == n_next->halfedge->twin);
        bool nn_n = (n_next->halfedge->next == n->halfedge);
        bool nn_nt = (n_next->halfedge->next == n->halfedge->twin);
        bool nnt_n = (n_next->halfedge->twin->next == n->halfedge);
        bool nnt_nt = (n_next->halfedge->twin->next == n->halfedge->twin);

        if (n_nn or n_nnt or nt_nn or nt_nnt) {
          n_left = n_next;
          n_right = n;
          if (n_nn) {
            h1 = n->halfedge;
            h2 = n_next->halfedge;
          }
          if (n_nnt) {
            h1 = n->halfedge;
            h2 = n_next->halfedge->twin;
          }
          if (nt_nn) {
            h1 = n->halfedge->twin;
            h2 = n_next->halfedge;
          }
          if (nt_nnt) {
            h1 = n->halfedge->twin;
            h2 = n_next->halfedge->twin;
          }
          n->id_halfedge_right = 6;
        } else {
          n_left = n;
          n_right = n_next;
          if (nn_n) {
            h1 = n_next->halfedge;
            h2 = n->halfedge;
          }
          if (nn_nt) {
            h1 = n_next->halfedge;
            h2 = n->halfedge->twin;
          }
          if (nnt_n) {
            h1 = n_next->halfedge->twin;
            h2 = n->halfedge;
          }
          if (nnt_nt) {
            h1 = n_next->halfedge->twin;
            h2 = n->halfedge->twin;
          }
          n->id_halfedge_right = 8;
        }
      } // n_left, n_right, h1, h2
      h0 = h2->next;
      n2 = h1->node;
      n3 = h2->node;
      n4 = h0->node;
      e0 = h0->edge;
      e1 = h1->edge;
      e2 = h2->edge;
      f0 = h0->face;

      // allocate
      n0 = new NodeTr();
      n1 = new NodeTr();
      e3 = new EdgeTr();
      e4 = new EdgeTr();
      e5 = new EdgeTr();
      e6 = new EdgeTr();
      e7 = new EdgeTr();
      e8 = new EdgeTr();
      h3 = new HalfedgeTr();
      h4 = new HalfedgeTr();
      h5 = new HalfedgeTr();
      h6 = new HalfedgeTr();
      h7 = new HalfedgeTr();
      h8 = new HalfedgeTr();
      h9 = new HalfedgeTr();
      h10 = new HalfedgeTr();
      f1 = new FaceTr();
      f2 = new FaceTr();
      f3 = new FaceTr();

      // reassign
      {
        h0->node = n4;
        h0->edge = e0;
        h0->face = f1;
        h0->prev = h4;
        h0->next = h3;
        h0->twin = h0->twin;
        h0->twin->twin = h0;

        h3->node = n2;
        h3->edge = e3;
        h3->face = f1;
        h3->prev = h0;
        h3->next = h4;
        h3->twin = h14;

        h4->node = n0;
        h4->edge = e4;
        h4->face = f1;
        h4->prev = h3;
        h4->next = h0;
        h4->twin = h5;

        h5->node = n4;
        h5->edge = e4;
        h5->face = f2;
        h5->prev = h7;
        h5->next = h6;
        h5->twin = h4;

        h6->node = n0;
        h6->edge = e5;
        h6->face = f2;
        h6->prev = h5;
        h6->next = h7;
        h6->twin = h8;

        h7->node = n1;
        h7->edge = e6;
        h7->face = f2;
        h7->prev = h6;
        h7->next = h5;
        h7->twin = h11;

        h8->node = n1;
        h8->edge = e5;
        h8->face = f3;
        h8->prev = h10;
        h8->next = h9;
        h8->twin = h6;

        h9->node = n0;
        h9->edge = e7;
        h9->face = f3;
        h9->prev = h8;
        h9->next = h10;
        h9->twin = h13;

        h10->node = n3;
        h10->edge = e8;
        h10->face = f3;
        h10->prev = h9;
        h10->next = h8;
        h10->twin = h12;
      } // h0, h3 ~ h10
      n0->halfedge = h6;
      n0->pos = n_right->pos();
      n1->halfedge = h8;
      n1->pos = n_left->pos();
      n2->halfedge = h3;
      n3->halfedge = h10;
      n4->halfedge = h0;
      e0->halfedge = h0;
      e3->halfedge = h3;
      e4->halfedge = h4;
      e5->halfedge = h6;
      e6->halfedge = h7;
      e7->halfedge = h9;
      e8->halfedge = h10;
      f1->halfedge = h0;
      f2->halfedge = h5;
      f3->halfedge = h8;
      if (n->id_halfedge_right == 6) n->halfedge_right = h6;
      else n->halfedge_right = h8;
    };

    auto next_triangle = [&]() {
      // convey variables to the next triangle
      if (n_next == n_right) {
        ha = h3;
        hb = h9;
      }
      else {
        ha = h10;
        hb = h7;
      }
      n = n_next;
      n_next = n->right;
    };

    // TODO : begin triangle
    // assign
    trim_triangle(nullptr, nullptr, nullptr, nullptr);
    next_triangle();
    if (n_next == n_right) {
      ha_init = h7;
      hb_init = h10;
    }
    else {
      ha_init = h9;
      hb_init = h3;
    }

    viewer.data().add_edges(n0->pos, n1->pos, color_white);
    viewer.data().add_edges(n0->pos, n4->pos, color_white);

    // iteration
    int i = 0;
    while (n != n_init) {
      h11 = h12 = h14 = h13 = nullptr;
      if (n == n_left) {
        h11 = ha;
        h12 = hb;
      }
      else {
        h13 = ha;
        h14 = hb;
      }
      trim_triangle(h11, h12,h13, h14);
      next_triangle();

      if (debug) {
        if (i == test) {
          h0->draw(viewer, true, 0);
          h3->draw(viewer, true, 3);
          h9->draw(viewer, true, 9);
          h7->draw(viewer, true, 7);
          h10->draw(viewer, true, 10);


          viewer.data().add_edges(n0->pos, n1->pos, color_white);
          viewer.data().add_edges(n0->pos, n4->pos, color_white);


          if (test2 == 0) {
            viewer.data().add_label(n->pos(), "n");
            viewer.data().add_label(n_next->pos(), "n_next");
          }

          if (test2 == 1) {
            viewer.data().add_label(n_left->pos(), "n_left");
            viewer.data().add_label(n_right->pos(), "n_right");
          }

          if (test2 == 2) {
            viewer.data().add_label(n0->pos, "n0");
            viewer.data().add_label(n1->pos, "n1");
          }

          if (test2 == 3) {
            viewer.data().add_label(n2->pos, "n2");
          }

          if (test2 == 4) {
            viewer.data().add_label(n3->pos, "n3");
          }

          if (test2 == 5) {
            viewer.data().add_label(n4->pos, "n4");
          }

          if (test2 == 6) {
            viewer.data().add_label(h0->node->pos, "h0");
          }

          if (test2 == 7) {
            viewer.data().add_label(h1->node->pos, "h1");
          }

          if (test2 == 8) {
            viewer.data().add_label(h2->node->pos, "h2");
          }

          Eigen::RowVector3d o = Eigen::RowVector3d(0, 0, 0);
          Eigen::RowVector3d x = Eigen::RowVector3d(1, 0, 0);
          Eigen::RowVector3d y = Eigen::RowVector3d(0, 1, 0);
          Eigen::RowVector3d z = Eigen::RowVector3d(0, 0, 1);
          viewer.data().add_edges(o, x, color_red);
          viewer.data().add_edges(o, y, color_green);
          viewer.data().add_edges(o, z, color_blue);

        }
      }


      if (n_next == n_init) { // last triangle
        ha->twin = ha_init;
        ha_init->twin = ha;
        hb->twin = hb_init;
        hb_init->twin = hb;
      }

      i++;
    }


    // rebuild mesh
    mesh_tr->nodes.clear();
    mesh_tr->edges.clear();
    mesh_tr->faces.clear();
    mesh_tr->halfedges.clear();
    set<NodeTr*> nodes;
    set<EdgeTr*> edges;
    set<FaceTr*> faces;
    set<HalfedgeTr*> halfedges;
    FaceTr* f;
    mesh_tr->boundary_tr.clear();
    mesh_tr->nodes_next.clear();

    n = n_init;

    while (true) {
      auto h = n->halfedge_right;
      h = h->twin;
      mesh_tr->boundary_tr.push_back(h);
      h->face = mesh_tr->face_exterior;
      mesh_tr->face_exterior->halfedge = h;

      f = h->face;
      n = n->right;
      if (n == n_init) break;
    }   // boundary

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
      mesh->nodes.push_back(n);
    }

    ifile.clear();
    ifile.seekg(0, ios::beg);

    while (getline(ifile, line, '\n')) {
      vector<string> items;
      boost::split(items, line, boost::is_any_of(" "), boost::token_compress_on);
      if (items[0] == "#") {
        mesh->nodes.resize(mesh->nodes.size()-1);
        if (items[1] == "center") {
          center.resize(1, 3);
          center << stod(items[2]), stod(items[3]), stod(items[4]);
        }
      }
      else {
        int idx = stoi(items[0]);
        Node *n = mesh->nodes[idx];
        n->idx = stoi(items[0]); // -1: not defined; -1: inserted
        n->pos = Eigen::RowVector3d(stof(items[1]), stof(items[2]), stof(items[3]));
        n->pos_origin = Eigen::RowVector3d(stof(items[1]), stof(items[2]), stof(items[3]));
        if (items[4] != "-1") n->idx_iso = stoi(items[4]);
        if (items[5] != "-1") n->idx_grad = stoi(items[5]);
        if (items[6] != "-1") n->right = mesh->nodes[stoi(items[6])];
        if (items[7] != "-1") n->left = mesh->nodes[stoi(items[7])];
        if (items[8] != "-1") n->up = mesh->nodes[stoi(items[8])];
        if (items[9] != "-1") n->down = mesh->nodes[stoi(items[9])];
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
    for (auto n : mesh->nodes ) {
      if (n->right) sum_len_iso_seg += (n->pos - n->right->pos).norm();
      if (n->up) sum_len_grad_seg += (n->pos - n->up->pos).norm();
      n_iso_seg ++; n_grad_seg++;
    }
  }

  // visualizer
  {
    viewer.core().background_color = Eigen::Vector4f(0., 0., 0., 1.);
    viewer.core().camera_base_zoom = 0.01;
    viewer.plugins.push_back(&menu);

    redraw();
  }

  // callbacks
  menu.callback_draw_viewer_menu = [&]()
  {
    if (ImGui::CollapsingHeader("params", ImGuiTreeNodeFlags_DefaultOpen)) {
//      ImGui::InputFloat("timeStep", &time_step);

      ImGui::InputFloat("isoSpacing", &mesh->iso_spacing);
      ImGui::InputFloat("closeness", &model->w_closeness);
      ImGui::InputFloat("stretch", &model->w_stretch);
      ImGui::InputFloat("bridge", &model->w_bridge);
      ImGui::InputFloat("bending", &model->w_bending);
      ImGui::InputFloat("flatten", &model->w_flatten);
      ImGui::InputFloat("angleStretch", &model->w_angle_stretch);
      ImGui::InputFloat("angleShear", &model->w_angle_shear);
      ImGui::InputFloat("spreading", &model->w_spreading);
      ImGui::InputFloat("damping", &model->damping);
      ImGui::InputFloat("dampingSmooth", &model->damping_flatten);
      ImGui::InputFloat("platformLength", &platform_length);
      ImGui::InputInt("numIter", &model->num_iter);
      ImGui::InputInt("test", &test);
      ImGui::InputInt("test2", &test2);
      ImGui::InputFloat("test3", &test3);
      if (ImGui::Checkbox("forward", &model->forward) ) {

      }
//      ImGui::InputFloat("smoothing", &smooth_strength);
//      ImGui::InputFloat("gapSize", &mesh->gap_size);
//      ImGui::InputInt("maxGap", &filter_threshold);
//      ImGui::InputInt("saddle_displacement", &saddle_displacement);
//      ImGui::InputInt("resolution", &w_out);
//      ImGui::InputFloat("shrinkage_cone", &shrinkage_cone);
//      ImGui::InputFloat("shrinkage_cone_down", &shrinkage_cone_down);
//      ImGui::InputFloat("radius_cone", &radius_cone);
//      ImGui::InputFloat("radius_trim", &radius_trim);


      ImGui::InputFloat("trace_reso", &trace_reso);
      ImGui::InputFloat("geodesic_gap", &geodesic_gap);
      ImGui::InputFloat("geodesy_gap", &geodesy_gap);
      ImGui::Checkbox("debug", &mesh->debug);
    }

    if (ImGui::CollapsingHeader("display", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::InputInt("index", &idx_focus)) {
        redraw();
      }

      ImGui::Text("displayMode");
      if (ImGui::RadioButton("graph", &display_mode, DisplayMode_Graph) or
          ImGui::RadioButton("halfedge", &display_mode, DisplayMode_Halfedge) or
          //          ImGui::RadioButton("segments", &display_mode, DisplayMode_Segments) or
          //          ImGui::RadioButton("tracing", &display_mode, DisplayMode_Tracing) or
          //          ImGui::RadioButton("connection", &display_mode, DisplayMode_Connection) or
          ImGui::RadioButton("mesh", &display_mode, DisplayMode_Mesh)
          //         or ImGui::RadioButton("input_mesh", &display_mode, DisplayMode_InputMesh)
          or ImGui::RadioButton("halfedgeTracing", &display_mode, DisplayMode_HalfedgeTracing)
          or ImGui::RadioButton("meshTracing", &display_mode, DisplayMode_MeshTracing)
        ) {
        redraw();
      }

//      ImGui::Spacing();
      if (display_mode == DisplayMode_Halfedge) {
        if (ImGui::Checkbox("stressField", &is_display_stress) ) {
          redraw();
        }
        if (ImGui::Checkbox("bridgeConstraint", &is_display_bridge)) {
          redraw();
        }
      }

      if (ImGui::Checkbox("nightMode", &is_display_background_black) ) {
        if (is_display_background_black) {
          viewer.core().background_color = Eigen::Vector4f(0.0, 0.0, 0.0, 1.0);
        }
        else {
          viewer.core().background_color = Eigen::Vector4f(1.0, 1.0, 1.0, 1.0);
        }
      }

      if (ImGui::Checkbox("label", &display_label)) {
        redraw();
      }
      if (display_label) {
        ImGui::Text("label");
        if (ImGui::RadioButton("node", &label_type, LabelType_Node) or
            ImGui::RadioButton("edge", &label_type, LabelType_Edge) or
            ImGui::RadioButton("face", &label_type, LabelType_Face) or
            ImGui::RadioButton("iso", &label_type, LabelType_Isoline) or
            ImGui::RadioButton("grad", &label_type, LabelType_Gradline) or
            ImGui::RadioButton("num_interp", &label_type, LabelType_NumInterp) or
            ImGui::RadioButton("is_end", &label_type, LabelType_End)
          ) {
          redraw();
        }
      }
    }

//    if (ImGui::Button("one-key")) {
//      mesh->upsample();
//      mesh->halfedgize();
//      project();
//
//      compute_boundary();
//      triangulate_tr();
//      convert_to_mesh();
//      compute_geodesic();
//      compute_geodesy();
//    }

    if (ImGui::Button("geodesicLine")) {
      redraw();
      mesh_tr->nodes_tr_traces.clear();
      compute_geodesic_line(geodesic_gap, geodesy_gap);
      draw_geodesic_lines();
    }

    if (ImGui::Button("trim-mesh")) {
      mesh_tr->trim_mesh(viewer, test2);
    }

    if (ImGui::CollapsingHeader("tools", ImGuiTreeNodeFlags_DefaultOpen)) {
      {
        if (ImGui::Button("reset")) {
          mesh->reset();
          redraw();
        }
      } // reset() hidden

      if (ImGui::Button("reorder_iso")) {
        mesh->reorder_iso();
      }

      if (ImGui::Button("saddle")) {
        mesh->fix_saddles();
      }

      if (ImGui::Button("upsample")) {
        mesh->upsample();

        display_mode = DisplayMode_Halfedge;
        redraw();
      }

      if (ImGui::Button("halfedgize")) {
        mesh->halfedgize();
        display_mode = DisplayMode_Halfedge;
      }

      {
//        if (ImGui::Button("map")) {
//          texture_map();
//        }
      } // map() hidden

      if (ImGui::Button("step")) {
        is_display_stress = true;
        for (int i = 0; i < model->num_iter; i++) {
//          step();
          model->step();
        }
        redraw();
      }

      if (ImGui::CollapsingHeader("hidden", ImGuiTreeNodeFlags_OpenOnDoubleClick)) {

        if (ImGui::Button("begin")) {
          is_flattening = true;
        }

        if (ImGui::Button("pause")) {
          is_flattening = false;
        }

        if (ImGui::Button("smooth")) {
          smooth();
          redraw();
        }

        if (ImGui::Button("scale")) {
          autoscale();
          redraw();
        }

        if (ImGui::Button("interpolate")) {
          interpolate();
        }

        if (ImGui::Button("merge")) {
          merge();
        }

        if (ImGui::Button("trace")) {
          trace();
          display_mode = DisplayMode_Tracing;
          redraw();
        }

        if (ImGui::Button("trace_saddle")) {
          trace_saddle();
        }

        if (ImGui::Button("adjust_cone")) {
          adjust_cone();
          redraw();
        }

        if (ImGui::Button("trim_center")) {
          trim_center();
          redraw();
        }

        if (ImGui::Button("save")) {
          save();
        }

        if (ImGui::Button("save_graph")) {
          save_graph();
        }


        if (ImGui::Button("save_obj")) {
          save_obj();
        }

      }

/*
      { // might be useful
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
      */
    }
  };

  viewer.callback_key_down =
    [&](igl::opengl::glfw::Viewer& viewer, int key, int mod)->bool
    {
      switch (display_mode) {
        case DisplayMode_Graph:
          switch (key) {
            case GLFW_KEY_LEFT:
              if (mesh->nodes[idx_focus]->left) {
                idx_focus = mesh->nodes[idx_focus]->left->idx;
              }
              break;

            case GLFW_KEY_RIGHT:
              if (mesh->nodes[idx_focus]->right) {
                idx_focus = mesh->nodes[idx_focus]->right->idx;
              }
              break;

            case GLFW_KEY_UP:
              if (mesh->nodes[idx_focus]->up) {
                idx_focus = mesh->nodes[idx_focus]->up->idx;
              }
              break;

            case GLFW_KEY_DOWN:
              if (mesh->nodes[idx_focus]->down) {
                idx_focus = mesh->nodes[idx_focus]->down->idx;
              }
              break;

            // fix saddle
            case GLFW_KEY_P: {
              Node *n = mesh->nodes[idx_focus];
              Node *n_new = n->right;
              n->down->up = n_new;
              n_new->down = n->down;
              n->down = nullptr;
              idx_focus = n_new->idx;
            }
              break;

            case GLFW_KEY_O: {
              Node *n = mesh->nodes[idx_focus];
              Node *n_new = n->left;
              n->down->up = n_new;
              n_new->down = n->down;
              n->down = nullptr;
              idx_focus = n_new->idx;
            }
              break;
          }
          break;

        case DisplayMode_Halfedge:
          switch (type_focus) {
            case TypeFocus_Node:
              switch (key) {
                case GLFW_KEY_H:
                  type_focus = TypeFocus_Halfedge;
                  idx_focus = mesh->nodes[idx_focus]->halfedge->idx;
                  break;

                case GLFW_KEY_LEFT:
                  if (mesh->nodes[idx_focus]->left) {
                    idx_focus = mesh->nodes[idx_focus]->left->idx;
                  }
                  break;

                case GLFW_KEY_RIGHT:
                  if (mesh->nodes[idx_focus]->right) {
                    idx_focus = mesh->nodes[idx_focus]->right->idx;
                  }
                  break;

                case GLFW_KEY_UP:
                  if (mesh->nodes[idx_focus]->up) {
                    idx_focus = mesh->nodes[idx_focus]->up->idx;
                  }
                  break;

                case GLFW_KEY_DOWN:
                  if (mesh->nodes[idx_focus]->down) {
                    idx_focus = mesh->nodes[idx_focus]->down->idx;
                  }
                  break;

              }
              break;

            case TypeFocus_Edge:
              if (key == GLFW_KEY_H) {
                type_focus = TypeFocus_Halfedge;
                idx_focus = mesh->edges[idx_focus]->halfedge->idx;
              }
              break;

            case TypeFocus_Face:
              if (key == GLFW_KEY_H) {
                type_focus = TypeFocus_Halfedge;
                idx_focus = mesh->faces[idx_focus]->halfedge->idx;
              }
              break;

            case TypeFocus_Halfedge:
              Halfedge *h = mesh->halfedges[idx_focus];
              switch (key) {
                case GLFW_KEY_N:
                  idx_focus = h->next->idx;
                  break;

                case GLFW_KEY_P:
                  idx_focus = h->prev->idx;
                  break;

                case GLFW_KEY_T:
                  if (h->twin)
                    idx_focus = h->twin->idx;
                  else
                    cout << "no twin" << endl;
                  break;

                case GLFW_KEY_V:
                  type_focus = TypeFocus_Node;
                  idx_focus = h->node->idx;
                  break;

                case GLFW_KEY_E:
                  type_focus = TypeFocus_Edge;
                  idx_focus = h->edge->idx;
                  break;

                case GLFW_KEY_F:
                  type_focus = TypeFocus_Face;
                  idx_focus = h->face->idx;
                  break;
                }
              break;
          }
          break;

        case DisplayMode_Segments:
          switch (key) {
            case GLFW_KEY_P:
              {
                Node *n = unvisited_vector[idx_focus];
                viewer.data().add_points(n->pos, color_white);
                if (n->idx == -1) {
                  if (n->up)
                    viewer.data().add_points(n->up->pos, color_blue);
                  if (n->down)
                    viewer.data().add_points(n->down->pos, color_green);
                } else {
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
              break;

            case GLFW_KEY_U:
              {
                Node *n = unvisited_vector[idx_focus];
                if (n->idx == -1) {
                  if (n->up) {
                    idx_focus = n->up->i_unvisited_vector;
                    cout << idx_focus << endl;
                  } else {
                    cout << "not found" << endl;
                  }
                } else {
                  if (n->ups[model->num_iter]) {
                    idx_focus = n->ups[model->num_iter]->i_unvisited_vector;
                    cout << idx_focus << endl;
                  } else {
                    cout << "not found" << endl;
                  }
                }
              }
              break;

            case GLFW_KEY_D:
              {
                Node *n = unvisited_vector[idx_focus];
                if (n->idx == -1) {
                  if (n->down) {
                    if (n->down == n) cout << "same node" << endl;
                    if (n == unvisited_vector[idx_focus]) cout << "n ==" << endl;
                    if (n->down == unvisited_vector[idx_focus]) cout << "n->down ==" << endl;
                    idx_focus = n->down->i_unvisited_vector;
                    cout << idx_focus << endl;
                  } else {
                    cout << "not found" << endl;
                  }
                } else {
                  if (n->downs[model->num_iter]) {
                    idx_focus = n->downs[model->num_iter]->i_unvisited_vector;
                    cout << idx_focus << endl;
                  } else {
                    cout << "not found" << endl;
                  }
                }
              }
              break;

            case GLFW_KEY_LEFT:
              if (unvisited_vector[idx_focus]->left) {
                idx_focus = unvisited_vector[idx_focus]->left->i_unvisited_vector;
              }
              break;

            case GLFW_KEY_RIGHT:
              if (unvisited_vector[idx_focus]->right) {
                idx_focus = unvisited_vector[idx_focus]->right->i_unvisited_vector;
              }
              break;
          }
          break;

        case DisplayMode_Tracing:
          switch (key) {
            case GLFW_KEY_LEFT:
              if (unvisited_vector[idx_focus]->left) {
                idx_focus = unvisited_vector[idx_focus]->left->i_unvisited_vector;
              }
              break;

            case GLFW_KEY_RIGHT:
              if (unvisited_vector[idx_focus]->right) {
                idx_focus = unvisited_vector[idx_focus]->right->i_unvisited_vector;
              }
              break;
          }
          break;

        case DisplayMode_HalfedgeTracing:
          switch (type_focus) {
            case TypeFocus_Node:
              if (key == GLFW_KEY_H) {
                idx_focus = mesh_tr->nodes[idx_focus]->halfedge->id;
                type_focus = TypeFocus_Halfedge;
              }
              break;

            case TypeFocus_Edge:
              if (key == GLFW_KEY_H) {
                idx_focus = mesh_tr->edges[idx_focus]->halfedge->id;
                type_focus = TypeFocus_Halfedge;
              }
              break;

            case TypeFocus_Face:
              if (key == GLFW_KEY_H) {
                idx_focus = mesh_tr->faces[idx_focus]->halfedge->id;
                type_focus = TypeFocus_Halfedge;
              }
              break;

            case TypeFocus_Halfedge:
              HalfedgeTr *h = mesh_tr->halfedges[idx_focus];
              switch (key) {
                case GLFW_KEY_N:
                  idx_focus = h->next->id;
                  break;
                case GLFW_KEY_P:
                  idx_focus = h->prev->id;
                  break;
                case GLFW_KEY_T:
                  if (h->twin) {
                    idx_focus = h->twin->id;
                  }
                  else {
                    cout << "key_down_n: no twin" << endl;
                  }
                  break;

                case GLFW_KEY_V:
                  idx_focus = h->node->id;
                  type_focus = TypeFocus_Node;
                  break;

                case GLFW_KEY_E:
                  idx_focus = h->edge->id;
                  type_focus = TypeFocus_Edge;
                  break;

                case GLFW_KEY_F:
                  idx_focus = h->face->id;
                  type_focus = TypeFocus_Face;
                  break;
              }
              break;
          }
          break;
      }
      redraw();
    };

  viewer.callback_mouse_down =
    [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
      // intersect ray with xy flatten
      Eigen::RowVector3d v_xy;  // intersection between ray and xy plain
      double x = viewer.current_mouse_x;
      double y = viewer.core().viewport(3) - viewer.current_mouse_y;
      Eigen::Vector3d s, dir;
      igl::unproject_ray(Eigen::Vector2f(x,y), viewer.core().view, viewer.core().proj, viewer.core().viewport, s, dir);
      float t = - s[2] / dir[2];
      v_xy = s + t * dir;

      Eigen::RowVector3d color = Eigen::RowVector3d(0.6,0.6,0.8);
      // viewer.data().add_points(v_xy, color);
    };

  viewer.callback_post_draw =
    [&](igl::opengl::glfw::Viewer& viewer)->bool
    {
      if (is_flattening) {
        is_display_stress = true;
        model->step();
        cout<<"convergence: "<<get_convergence()<<endl;
      }
    };

  // setup
  viewer.launch();

  return 0;
}


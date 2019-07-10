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
//#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_ray.h>
//#include <igl/colormap.h>
//#include <igl/avg_edge_length.h>

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
  int idx_iso = -1; // idx of the isoline (not segment)
  int idx_grad = -1;  // idx of the gradient line, if equals to -1, this is upsampled node

  Eigen::RowVector3d pos;
  Eigen::RowVector3d pos_origin;
  Eigen::RowVector3d velocity;

  Node* left = nullptr;
  Node* right = nullptr;
  Node* up = nullptr;
  Node* down = nullptr;
  set<Edge*> edges;
  Halfedge* halfedge = nullptr;
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
};

int main(int argc, char **argv) {
  // declaration
  Eigen::MatrixXd V;  // n_vertices * 3d
  Eigen::MatrixXi F;  // n_faces * 3i
  vector<Node*> nodes;
  vector<Edge*> edges;
  vector<Face*> faces;
  vector<Halfedge*> halfedges;
  vector<vector<Node *>> boundaries_top;
  vector<vector<Node *>> boundaries_bottom;
  vector<vector<Node *>> boundaries_saddle;

  ShapeOp::Solver* solver = new ShapeOp::Solver();
  Eigen::MatrixXd points;

  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiMenu menu;

  // param
  int idx_focus = 0;
  string type_focus = "node";
  bool display_label = false;
  int display_mode = 0; // 0: graph, 1: halfedges, 2: mesh
  int label_type = 0; // 0: node, 1: isoline, 2: grad line, 3: spring, 1: boundary
  bool display_stress = false; // 0: spring, 1: stress
  bool display_bridge = true;
  bool background_black = true;
  bool is_flattening = false;

  float iso_spacing = 10.0;
  int lower_bound = 0;
  int upper_bound = 0;
  int n_iter = 1;
  float w_smooth = 0.1;
  float damping_flatten = 1.2;

  float damping;
  float time_step;
  float w_closeness;
  float w_stretch;
  float w_bridge;
  float w_bending;
  float w_flatten;
  float w_angle_stretch;
  float w_angle_shear;
  float w_spreading;


  // const
  Eigen::RowVector3d color_black = Eigen::RowVector3d(0., 0., 0.);
  Eigen::RowVector3d color_white = Eigen::RowVector3d(1., 1., 1.);
  Eigen::RowVector3d color_red = Eigen::RowVector3d(0.8, 0.3, 0.3);
  Eigen::RowVector3d color_green = Eigen::RowVector3d(0.3, 0.8, 0.3);
  Eigen::RowVector3d color_blue = Eigen::RowVector3d(0.3, 0.3, 0.8);

  auto color_map = [&](float x) {
    double r, g, b;

    const double rone = 0.8;
    const double gone = 1.0;
    const double bone = 1.0;

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

  auto redraw = [&]() {
//    viewer.data().clear();  // this line causes bug
    viewer.data().V.resize(0,3);
    viewer.data().F.resize(0,3);
    viewer.data().points.resize(0, 6);
    viewer.data().lines.resize(0, 9);
    viewer.data().labels_positions.resize(0, 3);
    viewer.data().labels_strings.clear();

    if (display_label) {
      for (auto n : nodes) {
        if (label_type == 0)
          viewer.data().add_label(n->pos, to_string(n->idx));
        else if (label_type == 1)
          viewer.data().add_label(n->pos, to_string(n->idx_iso));
        else if (label_type == 2)
          viewer.data().add_label(n->pos, to_string(n->idx_grad));
      }
      for (auto e : edges) {
        if (label_type != 3)
          viewer.data().add_label(e->centroid(), to_string(e->idx));
        else if (label_type == 3) {
          if (e->spring == "boundary")
            viewer.data().add_label(e->centroid(), to_string(1));
          else
            viewer.data().add_label(e->centroid(), to_string(0));
        }
      }
    }

    if (display_mode == 0) {  // graph
      viewer.data().points = Eigen::MatrixXd(0, 6);
      viewer.data().lines = Eigen::MatrixXd(0, 9);
      viewer.data().labels_positions = Eigen::MatrixXd(0, 3);
      viewer.data().labels_strings.clear();
      for (auto n : nodes) {
        if (n->idx == idx_focus)
          viewer.data().add_points(n->pos, Eigen::RowVector3d(0.9, 0.9, 0));
        else
          viewer.data().add_points(n->pos, Eigen::RowVector3d(0, 0.9, 0.9));
        if (n->right) viewer.data().add_edges(n->pos, n->right->pos, Eigen::RowVector3d(0.9, 0, 0));
        if (n->up) viewer.data().add_edges(n->pos, n->up->pos, Eigen::RowVector3d(0, 0.9, 0));
      }
    }

    else if (display_mode == 1) { // halfedge

      for (auto e : edges) {
        Node* n_a = *e->nodes.begin();
        Node* n_b = *next(e->nodes.begin(), 1);

        if (display_stress) {
          e->shrinkage = 1.0 - e->len_3d / e->length();   // shrinkage ratio
          Eigen::RowVector3d color_stress = color_map(e->shrinkage);
          if (not ((not display_bridge) and (e->spring == "bridge")) ) {
            viewer.data().add_edges(n_a->pos, n_b->pos, color_stress);
          }
        }
        else {
          if (e->spring == "stretch") viewer.data().add_edges(n_a->pos, n_b->pos, color_red);
          else if (not ((not display_bridge) and (e->spring == "bridge")) )  viewer.data().add_edges(n_a->pos, n_b->pos, color_green);
          else if (e->spring == "boundary") viewer.data().add_edges(n_a->pos, n_b->pos, color_blue);
        }
      }

      Eigen::RowVector3d color_node;
      if (background_black) color_node = color_white;
      else color_node = color_black;

      if (type_focus == "node") {
        viewer.data().add_points(nodes[idx_focus]->pos, color_node);
      }
      else if (type_focus == "edge") {
        viewer.data().add_points(edges[idx_focus]->centroid(), color_node);
      }

      else if (type_focus == "face") {
        viewer.data().add_points(faces[idx_focus]->centroid(), color_node);
      }

      else if (type_focus == "halfedge") {
        viewer.data().add_points(halfedges[idx_focus]->edge->centroid(), color_node);
        viewer.data().add_points(halfedges[idx_focus]->node->pos, color_node);
      }
    }

    else if (display_mode == 2) { // surface
      V.resize(nodes.size(), 3);
      F.resize(faces.size(), 3);

      for (auto n : nodes) {
        V.row(n->idx) << n->pos;
      }

      for (auto f : faces) {
        int i0 = f->halfedge->node->idx;
        int i1 = f->halfedge->next->node->idx;
        int i2 = f->halfedge->next->next->node->idx;

        F.row(f->idx) << i0, i1, i2;
      }

      viewer.data().set_mesh(V, F);
      Eigen::MatrixXd FC, FN;
      FC = Eigen::MatrixXd::Ones(faces.size(), 3);
      FC = FC * 0.8;
      viewer.data().set_colors(FC);
      igl::per_face_normals(V, F, FN);  // might be redundent
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
      viewer.data().add_edges( (ns[0]->pos + ns[1]->pos + ns[2]->pos)/3,
                               (ns_triplet[0]->pos + ns_triplet[1]->pos + ns_triplet[2]->pos)/3,
                               Eigen::RowVector3d(0.9,0.9,0)
      );
      complete_face(es_triplet, ns_triplet);
    }

    if (find_the_other_face(h_c->node, h_b->node, &ns_triplet, &es_triplet)) {
      viewer.data().add_edges( (ns[0]->pos + ns[1]->pos + ns[2]->pos)/3,
                               (ns_triplet[0]->pos + ns_triplet[1]->pos + ns_triplet[2]->pos)/3,
                               Eigen::RowVector3d(0,0.9,0.9)
      );
      complete_face(es_triplet, ns_triplet);
    }
    if (find_the_other_face(h_b->node, h_a->node, &ns_triplet, &es_triplet)) {
      viewer.data().add_edges( (ns[0]->pos + ns[1]->pos + ns[2]->pos)/3,
                               (ns_triplet[0]->pos + ns_triplet[1]->pos + ns_triplet[2]->pos)/3,
                               Eigen::RowVector3d(0.9,0,0.9)
      );
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

  //////////////// menu funcs /////////////////

  auto initialize_param = [&]() {
    damping = 100.0;
    w_closeness = 1.0;
    w_stretch = 100.0;
    w_bridge = 400.0;
    w_bending = 10.0;
    w_angle_stretch = 10.0;
    w_angle_shear = 0.0;
    w_spreading = 0.0;
    w_flatten = 10.0;
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
        while (n_right->idx_grad == -1 or !n_right->up) {
          n_right = n_right->right;
        }
        while (n_up_right->idx_grad == -1 or !n_up_right->down) {
          n_up_right = n_up_right->right;
        }

        if (n_right->up == n_up_right) {  // in a quad
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
        }
      }


      if (!n->up) {
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
            nodes.push_back(n_center);

            for (auto nb : boundary_top) {
              add_edge(nb, n_center, "bridge");
            }
          } else {
            // TODO: two iso lines sharing a hole but no node has a downward connection
          }


          boundaries_top.push_back(boundary_top);
        }
      }

      // detect holes
      if (!n->down) { // detecting the bottom boundary

        vector<Node *> boundary_bottom;
        bool is_bottom_boundary = true;
        Node *n_iter = n;
        bool is_upper = true;
        set<int> ids_iso;
        do {  // searching the saddle path
          ids_iso.emplace(n_iter->idx_iso);
          if (is_upper) {
            if (n_iter->down) {
              n_iter = n_iter->down;
              boundary_bottom.push_back(n_iter);
              n_iter = n_iter->right;
              is_bottom_boundary = false;
              is_upper = false;
            } else if (n_iter->left) {
              n_iter = n_iter->left;
            }
          } else {
            if (n_iter->up) {
              n_iter = n_iter->up;
              boundary_bottom.push_back(n_iter);
              n_iter = n_iter->left;
              is_upper = true;
            } else if (n_iter->right) {
              n_iter = n_iter->right;
            }
          }
          boundary_bottom.push_back(n_iter);
        } while (n_iter != n);

        // saddle: not bottom, not same path
        if ((!is_bottom_boundary) and (ids_iso.size() > 2)) {
          // TODO: skeletonize the saddle, here using center point temporally

          bool in_boundary = false;
          for (auto b : boundaries_saddle) {
            for (auto n_b : b) {
              if (n == n_b) {
                in_boundary = true;
                break;
              }
            }
          }

          if (!in_boundary) {

            boundaries_saddle.push_back(boundary_bottom);
            Eigen::RowVector3d center_pos = Eigen::RowVector3d(0, 0, 0);

            for (auto n : boundary_bottom) {
              center_pos += n->pos;
            }
            center_pos /= boundary_bottom.size();

            Node *n_center = new Node();
            n_center->idx = nodes.size();
            n_center->pos = center_pos;
            n_center->pos_origin = center_pos;
            n_center->idx_iso = -1;
            n_center->idx_grad = -1;
            nodes.push_back(n_center);

            for (auto n : boundary_bottom) {
              add_edge(n, n_center, "bridge");
            }
          }
        }
        else if (is_bottom_boundary) {
          bool in_boundary = false;
          for (auto b : boundaries_bottom) {
            for (auto n_b : b) {
              if (n == n_b) {
                in_boundary = true;
                break;
              }
            }
          }

          if (!in_boundary) {
            boundaries_bottom.push_back(boundary_bottom);
            for (int i = 0; i < boundary_bottom.size(); i++) {
              int j = (i + 1) % boundary_bottom.size();
              Node *n_a = boundary_bottom[i];
              Node *n_b = boundary_bottom[j];
              viewer.data().add_edges(n_a->pos, n_b->pos, Eigen::RowVector3d(0, 0, 0.9));
              Edge *e = add_edge(n_a, n_b, "stretch");
              if (e->idx != -1) e->spring = "boundary"; //
            }
          }
        }

      }

      if (!n->right) { cerr << n->idx << " has no right node." << endl; }
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
      Face *f = new Face();
      Edge *edge;
      Edge *edge_0;
      vector<Edge *> edges_boundary;

      // face
      f->idx = faces.size();
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
        // compute edge rest_len
        for (auto e : edges) {
          e->rest_len = e->length();
        }
      }
      else {
        cout<<"boundary not found"<<endl;
      }
    }
    cout<<"done."<<endl;
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
      if (weight > 0) solver->addConstraint(c);
    }

    // BendingConstraint
    Eigen::MatrixXd points_flat;
    points_flat.resize(points.rows(), points.cols());
    for (auto e : edges) {
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
      if (w_bending > 0 and e->idx >= lower_bound and e->idx <= upper_bound) {
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
          if (id_vector.size() != 3) cout << "WARNING: not 3" << endl;

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
    solver->setTimeStep(time_step);
    solver->solve(n_iter);

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
      ImGui::InputInt("lower_bound", &lower_bound);
      ImGui::InputInt("upper_bound", &upper_bound);
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
      ImGui::InputFloat("timeStep", &time_step);
      ImGui::InputInt("numIter", &n_iter);
      ImGui::InputFloat("smoothing", &w_smooth);
    }

    if (ImGui::CollapsingHeader("display", ImGuiTreeNodeFlags_DefaultOpen)) {
      if (ImGui::InputInt("index", &idx_focus)) {
        redraw();
      }

      ImGui::Text("displayMode");

      if (ImGui::RadioButton("graph", &display_mode, 0) or
          ImGui::RadioButton("halfedge", &display_mode, 1) or
          ImGui::RadioButton("surface", &display_mode, 2)
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
            ImGui::RadioButton("isBoundary", &label_type, 3)
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

      if (ImGui::Button("upsample")) {
        upsample();
      }

      if (ImGui::Button("triangulate")) {
        triangulate();
      }

      if (ImGui::Button("halfedgize")) {
        halfedgize();
      }

      ImGui::Text("Convergence: ", to_string(get_convergence()).c_str());

      if (ImGui::Button("step")) {
        for (int i = 0; i < n_iter; i++) {
//          halfedgize();
          step();
          cout<< float(i + 1) / n_iter <<endl;
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
            cout<<"error: left, right, iso different"<<endl;
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

      if (ImGui::Button("interpolate")) {



      }

    }
  };

  viewer.callback_key_down =
    [&](igl::opengl::glfw::Viewer& viewer, int key, int mod)->bool
    {
      if (type_focus == "node") {
        if (key == GLFW_KEY_LEFT)
          if (nodes[idx_focus]->left) {
            idx_focus = nodes[idx_focus]->left->idx;
          }
        if (key == GLFW_KEY_RIGHT)
          if (nodes[idx_focus]->right) {
            idx_focus = nodes[idx_focus]->right->idx;
          }
        if (key == GLFW_KEY_UP)
          if (nodes[idx_focus]->up) {
            idx_focus = nodes[idx_focus]->up->idx;
          }
        if (key == GLFW_KEY_DOWN)
          if (nodes[idx_focus]->down) {
            idx_focus = nodes[idx_focus]->down->idx;
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

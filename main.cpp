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
  set<Edge*> edges;

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
  set<Node*> nodes;
  Halfedge* halfedge = nullptr;
  string spring = "stretch";      // stretch,
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
  Eigen::RowVector3d vector() {
    Eigen::RowVector3d p1 = this->node->pos;
    Eigen::RowVector3d p2 = this->twin->node->pos;
    Eigen::RowVector3d vec = p2 - p1;
    return vec;
  }
  float length() {
    // test
    return (this->node->pos - this->twin->node->pos).norm();
  }
};

struct Face {
  int idx = -1;
  Halfedge* halfedge = nullptr;
  bool is_external = false;

  Eigen::RowVector3d centroid() {
    Eigen::RowVector3d p = this->halfedge->node->pos;
    p += this->halfedge->next->node->pos;
    p += this->halfedge->prev->node->pos;
    p /= 3;
    return p;
  }

  Eigen::RowVector3d normal() {
    // test
    Eigen::RowVector3d n = this->halfedge->vector().cross(this->halfedge->next->vector());
    n.normalize();
    return n;
  }
};


int main(int argc, char **argv) {
  Eigen::MatrixXd V;  // n_vertices * 3d
  Eigen::MatrixXi F;  // n_faces * 3i
  vector<Node*> nodes;
  vector<Edge*> edges;
  vector<Face*> faces;
  vector<Halfedge*> halfedges;

  ShapeOp::Solver* solver = new ShapeOp::Solver();
  Eigen::MatrixXd points;

  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiMenu menu;

  int idx_focus = 0;
  string type_focus = "node";
  float avg_len_iso_edge = -1;
  float avg_len_grad_edge = -1;
  float iso_spacing = 1;
  float grad_spacing = 1;
  bool display_label = false;
  int display_mode = 0; // 0: graph, 1: edges, 2: nodes

  float damping = 100;
  float w_closeness = 1000;
  float w_stretch = 1.0;
  float w_bridge = 100.0;
  int n_iter = 100;

  Eigen::RowVector3d color_black = Eigen::RowVector3d(0, 0, 0);
  Eigen::RowVector3d color_red = Eigen::RowVector3d(0.8, 0.3, 0.3);
  Eigen::RowVector3d color_green = Eigen::RowVector3d(0.3, 0.8, 0.3);
  Eigen::RowVector3d color_blue = Eigen::RowVector3d(0.3, 0.3, 0.8);

  auto redraw = [&]() {
    // 0: graph, 1: edges, 2: nodes
    viewer.data().clear();
    viewer.data().points = Eigen::MatrixXd(0, 6);
    viewer.data().lines = Eigen::MatrixXd(0, 9);
    viewer.data().labels_positions = Eigen::MatrixXd(0, 3);
    viewer.data().labels_strings.clear();

    if (display_label) {
      for (auto n : nodes) {
        viewer.data().add_label(n->pos, to_string(n->idx));
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

    else if (display_mode == 1) { // edges
      for (auto e : edges) {
        Node* n_a = *e->nodes.begin();
        Node* n_b = *next(e->nodes.begin(), 1);

        if (e->spring == "stretch")
          viewer.data().add_edges(n_a->pos, n_b->pos, Eigen::RowVector3d(0, 0.9, 0));
        else if (e->spring == "bridge")
          viewer.data().add_edges(n_a->pos, n_b->pos, Eigen::RowVector3d(0.9, 0, 0));
        else if (e->spring == "boundary")
          viewer.data().add_edges(n_a->pos, n_b->pos, Eigen::RowVector3d(0, 0, 0.9));
      }

    }

    else if (display_mode == 2){  // nodes

      for (auto n : nodes) {
        for (auto e : n->edges) {
          Node* n_a = *e->nodes.begin();
          Node* n_b = *next(e->nodes.begin(), 1);

          if (e->spring == "stretch")
            viewer.data().add_edges(n_a->pos, n_b->pos, Eigen::RowVector3d(0, 0.9, 0));
          else if (e->spring == "bridge")
            viewer.data().add_edges(n_a->pos, n_b->pos, Eigen::RowVector3d(0.9, 0, 0));
          else if (e->spring == "boundary")
            viewer.data().add_edges(n_a->pos, n_b->pos, Eigen::RowVector3d(0, 0, 0.9));
        }
      }

    }

    else if (display_mode == 3) { // halfedge

      for (auto e : edges) {
        Node* n_a = *e->nodes.begin();
        Node* n_b = *next(e->nodes.begin(), 1);

        if ( e->spring == "stretch") viewer.data().add_edges(n_a->pos, n_b->pos, color_red);
        else if ( e->spring == "bridge") viewer.data().add_edges(n_a->pos, n_b->pos, color_green);
        else if ( e->spring == "boundary") viewer.data().add_edges(n_a->pos, n_b->pos, color_blue);

      }

      if (type_focus == "node") {
        viewer.data().add_points(nodes[idx_focus]->pos, color_black);
      }
      else if (type_focus == "edge") {
        Eigen::RowVector3d color;
        viewer.data().add_points(edges[idx_focus]->centroid(), color_black);
      }

      else if (type_focus == "face") {
        viewer.data().add_points(faces[idx_focus]->centroid(), color_black);
      }

      else if (type_focus == "halfedge") {
        viewer.data().add_points(halfedges[idx_focus]->edge->centroid(), color_black);
        viewer.data().add_points(halfedges[idx_focus]->node->pos, color_black);
      }

    }

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
        n_new->idx = nodes.size();
        n_new->pos = n->pos + vec * t_step * i;
        n_new->pos_origin = n->pos + vec * t_step * i;
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

  auto add_triangle = [&](Node* n_a, Node* n_b, Node* n_c) {
    Edge* e_ab = new Edge();
    Edge* e_bc = new Edge();
    Edge* e_ca = new Edge();
    e_ab->nodes.emplace(n_a);
    e_ab->nodes.emplace(n_b);
    e_bc->nodes.emplace(n_b);
    e_bc->nodes.emplace(n_c);
    e_ca->nodes.emplace(n_c);
    e_ca->nodes.emplace(n_a);

    e_ab->idx = edges.size();
    edges.push_back(e_ab);
    e_bc->idx = edges.size();
    edges.push_back(e_bc);
    e_ca->idx = edges.size();
    edges.push_back(e_ca);

    n_a->edges.emplace(e_ab);
    n_a->edges.emplace(e_ca);
    n_b->edges.emplace(e_ab);
    n_b->edges.emplace(e_bc);
    n_c->edges.emplace(e_bc);
    n_c->edges.emplace(e_ca);
  };

  auto add_edge = [&](Node* n_a, Node* n_b, string type) {
    Edge* e = new Edge();

    e->nodes.emplace(n_a);
    e->nodes.emplace(n_b);

    for (auto ee : n_a->edges) {
      if (ee->nodes == e->nodes) {
        return 0;
      }
    }

    if (n_a == n_b) cout<<type<<endl;

    e->spring = type;
    e->idx = edges.size();
    edges.push_back(e);

    n_a->edges.emplace(e);
    n_b->edges.emplace(e);
  };

  auto get_edge = [&](Node* n_a, Node* n_b) {
    set<Node*> nodes_tmp;
    nodes_tmp.emplace(n_a);
    nodes_tmp.emplace(n_b);
    for (auto e : edges) {
      if (e->nodes == nodes_tmp) {
        return e;
      }
    }
    Edge* ee = new Edge();
    return ee;
    cerr<<"edge not found: "<<n_a->idx<<" "<<n_b->idx<<endl;
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
        if (nn_a == nn_b) { // find a face
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

  //////////////////////////////////// init ////////////////////////////////////
  { // load graph
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

  { // compute avg_len
    float sum_len_iso_seg = 0;
    float sum_len_grad_seg = 0;
    int n_iso_seg = 0;
    int n_grad_seg = 0;
    for (auto n : nodes ) {
      if (n->right) sum_len_iso_seg += (n->pos - n->right->pos).norm();
      if (n->up) sum_len_grad_seg += (n->pos - n->up->pos).norm();
      n_iso_seg ++; n_grad_seg++;

    }
    avg_len_iso_edge = sum_len_iso_seg / n_iso_seg;
    avg_len_grad_edge = sum_len_grad_seg / n_grad_seg;
    cout<<"avg_iso: "<<avg_len_iso_edge<<" "<<"avg_grad: "<<avg_len_grad_edge<<"graph_size: "<<nodes.size()<<endl;
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
      if (ImGui::InputInt("selected_node", &idx_focus)) {
        redraw();
      }

      if (ImGui::RadioButton("graph", &display_mode, 0) or
        ImGui::RadioButton("edges", &display_mode, 1) or
        ImGui::RadioButton("nodes", &display_mode, 2) or
        ImGui::RadioButton("halfedge", &display_mode, 3)
      ) {
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
        for (auto n : nodes) {
          subdivide_edge(n);

        }
        cout<<"Done. avg_iso: "<<avg_len_iso_edge<<" "<<"avg_grad: "<<avg_len_grad_edge<<"graph_size: "<<nodes.size()<<endl;
        redraw();
      }

      if (ImGui::Button("flatten")) {
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
        for (auto n : nodes) {
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

          // detect top boundary
          if (!n->up) { // detect the up boundary
            vector<Node *> boundary_top;
            bool is_boundary = true;
            Node *n_iter = n;
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
          }

          // detect holes
          if (!n->down) { // detecting the bottom boundary
            vector<Node *> boundary_down;
            bool is_boundary = true;  // not used yet
            bool is_bottom_boundary = true;
            Node *n_iter = n;
            bool is_upper = true;
            set<int> ids_iso;

            while (true) {  // searching and setting flags
              ids_iso.emplace(n_iter->idx_iso);
              if (is_upper) {
                if (n_iter->down) {
                  n_iter = n_iter->down;
                  boundary_down.push_back(n_iter);
                  n_iter = n_iter->right;
                  is_bottom_boundary = false;
                  is_upper = false;
                } else if (n_iter->left) {
                  n_iter = n_iter->left;
                }
              } else {
                if (n_iter->up) {
                  n_iter = n_iter->up;
                  boundary_down.push_back(n_iter);
                  n_iter = n_iter->left;
                  is_upper = true;
                } else if (n_iter->right) {
                  n_iter = n_iter->right;
                }
              }

              boundary_down.push_back(n_iter);

              if (n_iter == n) {  // boundary closed
                break;
              }
            }

            // not bottom, not same path
            if ((is_boundary) and (!is_bottom_boundary) and (ids_iso.size() > 2)) {
              // TODO: skeletonize the saddle, here using center point temporally

              Eigen::RowVector3d center_pos = Eigen::RowVector3d(0, 0, 0);

              for (auto n : boundary_down) {
                center_pos += n->pos;
              }
              center_pos /= boundary_down.size();

              Node *n_center = new Node();
              n_center->idx = nodes.size();
              n_center->pos = center_pos;
              n_center->pos_origin = center_pos;
              n_center->idx_iso = -1;
              n_center->idx_grad = -1;
              nodes.push_back(n_center);

              for (auto n : boundary_down) {
                add_edge(n, n_center, "bridge");
              }
            }

            else if (is_bottom_boundary) {
              for (int i = 0; i < boundary_down.size(); i++) {
                int j = (i + 1) % boundary_down.size();
                Node *n_a = boundary_down[i];
                Node *n_b = boundary_down[j];
                viewer.data().add_edges(n_a->pos, n_b->pos, Eigen::RowVector3d(0, 0, 0.9));
                Edge *e = get_edge(n_a, n_b);
                if (e->idx != -1) e->spring = "boundary";
              }
            }
          }

          if (!n->right) { cerr << n->idx << " has no right node." << endl; }
        }
      }

      if (ImGui::Button("halfedgize")) { // create half edge mesh
        vector<Node *> ns_triplet;
        vector<Edge *> es_triplet;
        // find first triangle
        cout<<"looking for first triangle"<<endl;
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
          Face* f = new Face();
          Edge* edge;
          Edge* edge_0;
          vector<Edge*> edges_boundary;

          // face
          f->is_external = true;
          f->idx = faces.size();
          faces.emplace_back(f);

          for (auto e : edges) {  // find the first boundary edge
            if (e->spring == "boundary") {
              edge = e;
              edge_0 = e;
            }
          }

          do {  // collect boundary edges, create external halfedges
            // TODO: endless loop
            edges_boundary.emplace_back(edge);

            Halfedge* h = new Halfedge();
            h->idx = halfedges.size();
            halfedges.emplace_back(h);
            h->edge = edge;
            h->face = f;
            f->halfedge = h;
            h->twin = edge->halfedge;
            h->twin->twin = h;

            // find next boundary edge
            Halfedge* hh = edge->halfedge;
            while (true) {
              hh = hh->next;
              if (hh->edge->spring == "boundary"){
                edge = hh->edge;
                break;
              }
              hh = hh->twin;
            }
          } while (edge != edge_0);

          // connect boundary halfedges
          int i = 0;
          for (auto e : edges_boundary) {
            Halfedge* h = e->halfedge->twin;
            int i_next = (i + 1) % edges_boundary.size();
            int i_prev = (i + edges_boundary.size() - 1) % edges_boundary.size();
            h->next = edges_boundary[i_prev]->halfedge->twin;
            h->prev = edges_boundary[i_next]->halfedge->twin;
            h->node = edges_boundary[i_next]->halfedge->node;
            i++;
          }

        }

      }

      if (ImGui::Button("model")) {
        // set points
        {
          points.resize(3, nodes.size());
          for (auto n : nodes) {
            points(0, n->idx) = n->pos[0];
            points(1, n->idx) = n->pos[1];
            points(2, n->idx) = n->pos[2];
          }
          solver->setPoints(points);
        }

        // set constraints
        {
          // Closeness
          for (auto n : nodes) {
            std::vector<int> id_vector;
            id_vector.push_back(n->idx);
            ShapeOp::Scalar weight = w_closeness;
            auto points = solver->getPoints();
            auto c = std::make_shared<ShapeOp::ClosenessConstraint>(id_vector, weight, points);
            solver->addConstraint(c);
          }


          // Stretch
          for (auto e : edges) {
            std::vector<int> id_vector;
            cout<<"a"<<endl;
            cout<<e->halfedge->node->idx<<endl;
            id_vector.push_back(e->halfedge->node->idx);
            cout<<"b"<<endl;
            id_vector.push_back(e->halfedge->twin->node->idx);
            cout<<"c"<<endl;
            auto weight = w_stretch;
            cout<<"d"<<endl;
            if (e->spring == "bridge") weight = w_bridge;
            cout<<"e"<<endl;
            auto c = std::make_shared<ShapeOp::EdgeStrainConstraint>(id_vector, weight, solver->getPoints());
            cout<<"f"<<endl;
            double len_origin =  e->halfedge->length();
            cout<<"g"<<endl;
            c->setEdgeLength(len_origin);
            cout<<"h"<<endl;
            solver->addConstraint(c);
          }

        }

        // solve
        {
          solver->setDamping(damping);

          solver->initialize();
          solver->solve(n_iter);

          points = solver->getPoints();
          for (auto n : nodes) {
            n->pos = points.col(n->idx);
          }
        }

        redraw();

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
      // intersect ray with xy plane
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

  viewer.launch();

  return 0;
}

#include "geometry.h"

void HalfedgeMesh::reorder_iso() {
  while (true) {
    bool is_ordered = true;
    for (auto n : this->nodes) {
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
};

void HalfedgeMesh::fix_saddle(Node* n_left, Node* n_right) {
  double dist = (n_left->pos - n_right->pos).norm();
  Eigen::RowVector3d vec = n_right->pos - n_left->pos;
  vec.normalize();
  int num_nodes = int( dist / this->iso_spacing);
  for (int i = 0; i < num_nodes; i++) {
    Node* n_new = new Node();
    n_new->idx = this->nodes.size();
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
      n_new->left = this->nodes[this->nodes.size() - 1];
      n_new->left->right = n_new;
    }
    if (i == num_nodes - 1) {
      n_new->right = n_right;
      n_right->left_saddle = n_new;
    }
    this->nodes.push_back(n_new);
  }
}

void HalfedgeMesh::fix_saddles() {
  int nodes_size = this->nodes.size();
  for (int i=0; i<nodes_size; i++) {
    Node* n = this->nodes[i];
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
        this->fix_saddle(n_up, n_right->up);
      }
    }
  }
}

void HalfedgeMesh::subdivide_edge(Node* n) {
  if (n->right) {
    Eigen::RowVector3d vec = n->right->pos - n->pos;
    int num_insert = floor(vec.norm() / this->iso_spacing); // number of inserted vertices on one iso line
    float t_step = 1.0 / (num_insert + 1);

    Node *n_prev = n;
    for (int i = 1; i <= num_insert; i++) {
      Node *n_new = new Node();
      n_new->idx = this->nodes.size();
      n_new->pos = n->pos + vec * t_step * i;
      n_new->pos_origin = n_new->pos;
      n_new->idx_iso = n->idx_iso;
      n_new->left = n_prev;
      n_new->right = n_prev->right;
      n_new->left->right = n_new;
      n_new->right->left = n_new;
      n_prev = n_new;
      this->nodes.push_back(n_new);
    }
  }
}

Edge* HalfedgeMesh::add_edge(Node* n_a, Node* n_b, string type) {
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
  e->idx = this->edges.size();
  e->len_3d = e->length();
  e->len = e->len_3d;
  e->len_prev = 0.0000001;
  this->edges.push_back(e);

  n_a->edges.emplace(e);
  n_b->edges.emplace(e);

  return e;
}

void HalfedgeMesh::triangulate() {
  cout<<"triangulating......"<<endl;

  int nodes_size = this->nodes.size();
  for (int i = 0; i < nodes_size; i++) {
    Node* n = this->nodes[i];

    // detect quads (up right quad of each node)
    if (debug) {  cout<<n->idx<<endl; }

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

//          viewer.data().add_edges(n_u->pos, n_d->pos, Eigen::RowVector3d(0.9, 0, 0));

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
      if (debug) {cout<<"detect top boundary"<<endl;}
      bool in_boundary = false; // already in boundary
      for (auto b : this->boundaries_top) {
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

          node_iter = node_iter->right;
          if (node_iter == n) {  // close the boundary
            break;
          }
        }

        if (is_boundary) {
          Eigen::RowVector3d center_pos = Eigen::RowVector3d(0, 0, 0);
          for (auto nb : boundary_top) {
            center_pos += nb->pos;
          }
          center_pos /= boundary_top.size();

          Node *n_center = new Node();
          n_center->idx = this->nodes.size();
          n_center->pos = center_pos;
          n_center->pos_origin = center_pos;
          n_center->idx_iso = -1; // TODO: indexing for inserted isoline
          n_center->idx_grad = -1;
          n_center->is_cone = true;
          this->nodes.push_back(n_center);
          this->cones.push_back(n_center);

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

        this->boundaries_top.push_back(boundary_top);
      }
      else {
        if (debug) {cout<<"already in boundary"<<endl;}
      }
    }

    // bottom_boundary, only one
    if (!n->down and this->boundary_bottom.empty() ) {
      if (debug) {cout<<"detect bottom boundary"<<endl;}

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
        this->boundary_bottom.push_back(n_iter);
        n_iter = n_iter->right;
      } while (n_iter != n);

      if (is_bottom_boundary) {
        for (int i = 0; i < this->boundary_bottom.size(); i++) {
          int j = (i + 1) % this->boundary_bottom.size();
          Node *n_a = boundary_bottom[i];
          cout<<"boundary_node: "<<n_a->idx<<endl;
          Node *n_b = boundary_bottom[j];
          Edge *e = add_edge(n_a, n_b, "stretch");
          e->spring = "boundary";
//            if (e->idx != -1) e->spring = "boundary";
        }
      }
      else {
        this->boundary_bottom.clear();
      }

    }

    if (n->right_saddle and (not n->on_saddle_boundary) ) {
      if (debug) cout<<"collect saddle"<<endl;
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
      n_center->idx = this->nodes.size();
      n_center->pos = center_pos;
      n_center->pos_origin = center_pos;
      n_center->idx_iso = -1;
      n_center->idx_grad = -1;
      n_center->is_saddle = true;
      this->nodes.push_back(n_center);
      this->saddles.push_back(n_center);

      for (auto n_iter : saddle_boundary) {
        add_edge(n_iter, n_center, "bridge");
      }
    }

    if (!n->right or !n->left) { cerr << n->idx << " has no right node." << endl; }
  }

  cout<<"done."<<endl;
}

void HalfedgeMesh::upsample() {
  cout<<"upsampling.....";
  int nodes_size = this->nodes.size();
  for (int i=0; i<nodes_size; i++) {
    Node* n = this->nodes[i];
    subdivide_edge(n);
  }
  cout<<"done."<<endl;
  triangulate();
}

bool HalfedgeMesh::find_the_other_face(Node* n_a, Node* n_b, vector<Node*>* ns_triplet, vector<Edge*>* es_triplet) {

  Edge* e_c;
  for (auto e : this->edges) {
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

bool HalfedgeMesh::complete_face(vector<Edge*> es, vector<Node*> ns) {
  Face* f = new Face();
  Halfedge* h_a = new Halfedge();
  Halfedge* h_b = new Halfedge();
  Halfedge* h_c = new Halfedge();

  f->idx = this->faces.size();
  this->faces.emplace_back(f);
  f->halfedge = h_a;
  f->is_saddle = false;
  for (auto n : ns) {
    if (n->is_saddle) f->is_saddle = true;
  }

  h_a->idx = this->halfedges.size();
  this->halfedges.emplace_back(h_a);
  h_a->node = ns[0];
  h_a->edge = es[0];
  h_a->face = f;
  h_a->prev = h_c;
  h_a->next = h_b;
  if (es[0]->halfedge) {
    h_a->twin = es[0]->halfedge;
    h_a->twin->twin = h_a;
  }

  h_b->idx = this->halfedges.size();
  this->halfedges.emplace_back(h_b);
  h_b->node = ns[1];
  h_b->edge = es[1];
  h_b->face = f;
  h_b->prev = h_a;
  h_b->next = h_c;
  if (es[1]->halfedge) {
    h_b->twin = es[1]->halfedge;
    h_b->twin->twin = h_b;
  }

  h_c->idx = this->halfedges.size();
  this->halfedges.emplace_back(h_c);
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
    if (debug) {
      Eigen::RowVector3d p0 = (ns[0]->pos + ns[1]->pos + ns[2]->pos) / 3;
      Eigen::RowVector3d p1 = (ns_triplet[0]->pos + ns_triplet[1]->pos + ns_triplet[2]->pos) / 3;
    }
    complete_face(es_triplet, ns_triplet);
  }
  if (find_the_other_face(h_c->node, h_b->node, &ns_triplet, &es_triplet)) {
    if (debug) {
      Eigen::RowVector3d p0 = (ns[0]->pos + ns[1]->pos + ns[2]->pos) / 3;
      Eigen::RowVector3d p1 = (ns_triplet[0]->pos + ns_triplet[1]->pos + ns_triplet[2]->pos) / 3;
    }
    complete_face(es_triplet, ns_triplet);
  }
  if (find_the_other_face(h_b->node, h_a->node, &ns_triplet, &es_triplet)) {
    if (debug) {
      Eigen::RowVector3d p0 = (ns[0]->pos + ns[1]->pos + ns[2]->pos) / 3;
      Eigen::RowVector3d p1 = (ns_triplet[0]->pos + ns_triplet[1]->pos + ns_triplet[2]->pos) / 3;
    }
    complete_face(es_triplet, ns_triplet);
  }
}

void HalfedgeMesh::halfedgize() {
  cout<<"halfedgizing......";
  vector<Node *> ns_triplet;
  vector<Edge *> es_triplet;
  // find first triangle
  for (Node *n_a : this->nodes) {
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
    for (auto e : this->edges) {
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
    for (auto e : this->edges) {
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
    f->idx = this->faces.size();
    f->is_external = true;
    this->faces.emplace_back(f);

    bool found_boundary = false;
    for (auto e : this->edges) {  // find the first boundary edge
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
        h->idx = this->halfedges.size();
        this->halfedges.emplace_back(h);
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

      for (auto e : this->edges) {
        e->rest_len = e->length();
      }
    }
    else {
      cout<<"boundary not found"<<endl;
    }
  }
  cout<<"done."<<endl;
};

float Edge::length() {
  return ((*next(nodes.begin(), 0))->pos - (*next(nodes.begin(), 1))->pos ).norm();
}
Eigen::RowVector3d Edge::centroid() {
  return ((*nodes.begin())->pos + (*next(nodes.begin(), 1))->pos) / 2;
}

float Halfedge::length() {
  return this->edge->length();
}

Eigen::RowVector3d Halfedge::vector() {
  Eigen::RowVector3d p1 = this->node->pos;
  Eigen::RowVector3d p2 = this->twin->node->pos;
  Eigen::RowVector3d vec = p2 - p1;
  return vec;
}

Node* Face::node(int i) {
  Halfedge* h = this->halfedge;
  if (i == 0) return h->node;
  if (i == 1) return h->next->node;
  if (i == 2) return h->next->next->node;
}

Eigen::RowVector3d Face::centroid() {
  Halfedge* h0 = this->halfedge;
  Halfedge* h = h0;
  int num_nodes = 0;
  Eigen::RowVector3d p;
  p << 0,0,0;

  while (true) {
    p += h->node->pos;
    num_nodes++;

    h = h->next;
    if (h == h0) break;
  }
  p /= num_nodes;

  return p;
}

Eigen::RowVector3d Face::normal() {
  Eigen::RowVector3d n = this->halfedge->vector().cross(this->halfedge->next->vector());
  n.normalize();
  return n;
}

double Face::area() {
  Eigen::RowVector3d v1 = this->halfedge->prev->node->pos - this->halfedge->node->pos;
  Eigen::RowVector3d v2 = this->halfedge->next->node->pos - this->halfedge->node->pos;
  double area = v1.cross(v2).norm() / 2;
  return area;
}

double Face::area_origin() {
  Eigen::RowVector3d v1_origin = this->halfedge->prev->node->pos_origin - this->halfedge->node->pos_origin;
  Eigen::RowVector3d v2_origin = this->halfedge->next->node->pos_origin - this->halfedge->node->pos_origin;
  double area_origin = v1_origin.cross(v2_origin).norm() / 2;
  return area_origin;
}

double Face::opacity() {
  double opacity = this->area_origin() / this->area();
  if (opacity > 1) return 1.0;
  if (opacity < 0) return 0.0;
  return opacity;
}
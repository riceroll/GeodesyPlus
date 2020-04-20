#include "tracer.h"



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

void HalfedgeTr::draw(igl::opengl::glfw::Viewer &viewer, bool label=false, int id=0) {
    if (not this->twin) {
      cerr<<"no twin"<<endl;
    }
    else {
      Eigen::RowVector3d axis = this->twin->node->pos - this->node->pos;
      auto z_ax = Eigen::RowVector3d(0,0,1);
      Eigen::RowVector3d shift = z_ax.cross(axis);
      shift.normalize();
      Eigen::MatrixXd psa(2,3);
      Eigen::MatrixXd psb(2,3);
      psa.row(0) = this->node->pos + axis * 0.2 + shift * axis.norm() * 0.1;
      psa.row(1) = this->node->pos + axis * 0.8 + shift * axis.norm() * 0.1;

      psb.row(0) = this->node->pos + axis * 0.8 + shift * axis.norm() * 0.1;
      psb.row(1) = this->node->pos + axis * 0.7 + shift * axis.norm() * 0.2;
      viewer.data().add_edges(psa, psb, Eigen::RowVector3d(0, 0.8, 0.8));

      if (label) {
        viewer.data().add_label( (psa.row(0) + psb.row(0))/ 2, to_string(id));
      }

    }
  }

Eigen::RowVector3d EdgeTr::centroid() {
    return (halfedge->node->pos + halfedge->twin->node->pos) / 2;
  }

Eigen::RowVector3d FaceTr::centroid() {
    Eigen::RowVector3d p = this->halfedge->node->pos;
    p += this->halfedge->next->node->pos;
    p += this->halfedge->prev->node->pos;
    p /= 3;
    return p;
  }

Eigen::RowVector3d NodeTr_cycle::pos() {
    NodeTr* n0 = this->halfedge->node;
    NodeTr* n1 = this->halfedge->twin->node;
    return n0->pos + this->t * (n1->pos - n0->pos);
  }


void HalfedgeMeshTr::trim_mesh(igl::opengl::glfw::Viewer& viewer, int test2) {

};

/*
void HalfedgeMeshTr::trim_mesh2(igl::opengl::glfw::Viewer& viewer, int test2) {//
//   *  halfedges_tr
//   *  edges_tr
//   *  nodes
//   *  faces_tr
//   *  trace_tr: NodeOnHalfedge (*Edge, )

  // const
  Eigen::RowVector3d color_grey = Eigen::RowVector3d(0.5, 0.5, 0.5);
  Eigen::RowVector3d color_white = Eigen::RowVector3d(1., 1., 1.);
  Eigen::RowVector3d color_red = Eigen::RowVector3d(0.8, 0.2, 0.2);
  Eigen::RowVector3d color_green = Eigen::RowVector3d(0.2, 0.8, 0.2);
  Eigen::RowVector3d color_blue = Eigen::RowVector3d(0.2, 0.2, 0.8);
  Eigen::RowVector3d color_magenta = Eigen::RowVector3d(0.8, 0.2, 0.8);
  Eigen::RowVector3d color_cyon = Eigen::RowVector3d(0.2, 0.8, 0.8);
  Eigen::RowVector3d color_yellow = Eigen::RowVector3d(0.8, 0.8, 0.2);

  // TODO : halfedgemesh might have multiple cycles later, and do not need to assure the cycle is closed
  auto is_cycle_closed = [&]() {
    int num_nodes_in_cycle = 0;

    if (this->nodes_next.size() == 0) return false;
    auto n = this->nodes_next[0];
    auto n0 = n;
    while (true) {
      if (not n->right) return false;
      n = n->right;
      num_nodes_in_cycle++;
      if (n == n0) {
        if (num_nodes_in_cycle == this->nodes_next.size()) return true;
        else return false;
      }
    }
  };
  cout<<"closed cycle: "<<is_cycle_closed()<<endl;

  NodeTr_cycle* n = this->nodes_next[0];
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


    {
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
  this->nodes.clear();
  this->edges.clear();
  this->faces.clear();
  this->halfedges.clear();
  set<NodeTr*> nodes;
  set<EdgeTr*> edges;
  set<FaceTr*> faces;
  set<HalfedgeTr*> halfedges;
  FaceTr* f;
  this->boundary_tr.clear();
  this->nodes_next.clear();

  n = n_init;

  while (true) {
    auto h = n->halfedge_right;
    h = h->twin;
    this->boundary_tr.push_back(h);
    h->face = this->face_exterior;
    this->face_exterior->halfedge = h;

    f = h->face;
    n = n->right;
    if (n == n_init) break;
  }   // boundary



  Eigen::RowVector3d o = Eigen::RowVector3d(0, 0, 0);
  Eigen::RowVector3d x = Eigen::RowVector3d(1, 0, 0);
  Eigen::RowVector3d y = Eigen::RowVector3d(0, 1, 0);
  Eigen::RowVector3d z = Eigen::RowVector3d(0, 0, 1);
  viewer.data().add_edges(o, x, color_red);
  viewer.data().add_edges(o, y, color_green);
  viewer.data().add_edges(o, z, color_blue);

};
*/
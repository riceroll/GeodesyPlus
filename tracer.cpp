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
        viewer.data().add_label( (psa + psb)/ 2, to_string(id));
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

Eigen::RowVector3d NodeTr_tracepos;

Eigen::RowVector3d NodeTr_cycle::pos() {
    NodeTr* n0 = this->halfedge->node;
    NodeTr* n1 = this->halfedge->twin->node;
    return n0->pos + this->t * (n1->pos - n0->pos);
  }

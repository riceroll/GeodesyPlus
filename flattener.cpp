#include "flattener.h"

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
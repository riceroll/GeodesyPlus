#include "model.h"


void Model::step() {
  cout<<"done."<<endl<<"Updating position......";
  { // flattening velocity
    double d_max = 0;
    for (auto n : mesh->nodes) if (abs(n->pos.z()) > d_max) d_max = abs(n->pos.z());

    double xy_max = 0;
    for (auto n : mesh->nodes) {
      if (abs(n->pos.x()) > xy_max) xy_max = abs(n->pos.x());
      if (abs(n->pos.y()) > xy_max) xy_max = abs(n->pos.y());
    }

    for (auto n : mesh->nodes) {
      float speed = ( w_flatten * exp(log(d_max / xy_max) * (1 / damping_flatten) ) ) * (n->pos.z() / d_max);
      n->velocity = Eigen::RowVector3d(0, 0, - speed);
      n->pos += n->velocity;
    }
  }

  cout<<"done."<<endl<<"Updating solver......";
  solver->reset();

  // setPoints
  {
    solver_points.resize(3, mesh->nodes.size());
    for (auto n : mesh->nodes) {
      solver_points(0, n->idx) = n->pos.x();
      solver_points(1, n->idx) = n->pos.y();
      solver_points(2, n->idx) = n->pos.z();
    }
    solver->setPoints(solver_points);
  }

  // ClosenessConstraint (prevent drifting)
  for (auto n : mesh->nodes) {
    std::vector<int> id_vector;
    id_vector.push_back(n->idx);
    auto c = std::make_shared<ShapeOp::ClosenessConstraint>(id_vector, w_closeness, solver->getPoints());
    if (w_closeness > 0.) solver->addConstraint(c);
  }

  // EdgeStrainConstraint
  for (auto e : mesh->edges) {
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
    if (forward) {
      c = std::make_shared<ShapeOp::EdgeStrainConstraint>(id_vector, weight, solver->getPoints(), 0.7, 0.8);
    }
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
  points_flat.resize(solver_points.rows(), solver_points.cols());
  for (auto e : mesh->edges) {
    if (e->spring == "boundary") continue;

    std::vector<int> id_vector;
    int i_0 = e->halfedge->node->idx;
    int i_1 = e->halfedge->twin->node->idx;
    int i_2 = e->halfedge->next->next->node->idx;
    int i_3 = e->halfedge->twin->next->next->node->idx;

    Eigen::RowVector3d v01 = mesh->nodes[i_1]->pos - mesh->nodes[i_0]->pos;
    Eigen::RowVector3d v02 = mesh->nodes[i_2]->pos - mesh->nodes[i_0]->pos;
    Eigen::RowVector3d v03 = mesh->nodes[i_3]->pos - mesh->nodes[i_0]->pos;

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
    for (auto n : mesh->nodes) {
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

        auto c = std::make_shared<ShapeOp::AngleConstraint>(id_vector, w_angle_stretch, solver_points, 0, 0);
        if (w_angle_stretch > 0) solver->addConstraint(c);
      }
    }
  }

  // AngleConstraint for shearing (negative effect on flattening)
  for (auto f : mesh->faces) {
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
    for (auto n : mesh->nodes) {
      id_vector.push_back(n->idx);
    }
    auto c = std::make_shared<ShapeOp::PlaneConstraint>(id_vector, w_flatten, solver->getPoints());
//      if (w_flatten > 0) solver->addConstraint(c);
  }

  // flattening force (cannot see any effect?)
  {
    for (auto n : mesh->nodes) {
      double d = n->pos.z();  // distance to the target plane
      double m = d * w_flatten;
      Eigen::RowVector3d force(0., 0., -m);
      auto f = std::make_shared<ShapeOp::VertexForce>(force, n->idx);
//        if (w_flatten > 0.) solver->addForces(f);
    }
  }

  // spreading force
  Eigen::RowVector3d n0(0, 0, 1.0);
  for (auto e : mesh->edges) {
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

  solver_points = solver->getPoints();
  for (auto n : mesh->nodes) {
    n->pos = solver_points.col(n->idx);
  }
  for (auto e : mesh->edges) {
    e->len_prev = e->len;
    e->len = e->length();
  }

  cout<<"done."<<endl;
};

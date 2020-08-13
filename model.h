#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <string>

// Eigen
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/Dense>

// ShapeOp
#include "Solver.h"
#include "Constraint.h"
#include "Force.h"

#include "geometry.h"

using namespace std;

#pragma once


struct Model {
  ShapeOp::Solver* solver = new ShapeOp::Solver();
  Eigen::MatrixXd solver_points;
  HalfedgeMesh* mesh;

  int num_iter = 1;
  float damping_flatten = 1.2;
  float damping = 100.0;

  float w_closeness = 1.0;
  float w_stretch = 400.0;
  float w_bridge = 400.0;
  float w_bending = 10.0;
  float w_flatten = 3.0;
  float w_angle_stretch = 100.0;
  float w_angle_shear = 0.0;
  float w_spreading = 0.0;

  bool forward = false;

  Model(HalfedgeMesh* mesh) {
    this->mesh = mesh;
  }

  void step();
};

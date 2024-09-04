#ifndef JPNN_H
#define JPNN_H

#include <cassert>
#include <cmath>
#include <vector>
#include <map>
#include <random>
#include <string>
#include <functional>
#include <memory>
#include <iostream>

using number = float;

class Mat {
 public:
  Mat(int rows, int cols);
  number& operator()(int row, int col);
  number operator()(int row, int col) const;
  int rows() const;
  int cols() const;

 protected:
  int rows_;
  int cols_;
  std::unique_ptr<number[]> data_;
};

std::ostream& operator<<(std::ostream& os, const Mat& m);

class Row : public Mat {
 public:
  // Row just a matrix with one row.
  Row(int ncol) : Mat(1, ncol) {}
  number& operator()(int col);
  number operator()(int col) const;
  void from_mat(const Mat& m, int row);
};

namespace matrix {
Mat& assign(Mat& dst, number x);
Mat& assign(Mat& dst, const std::vector<number>& xs);
Mat& assign(Mat& dst, const Mat& m);
Mat& assign(Row& dst, const Mat& m, int row);
Mat& add(Mat& dst, const Mat& m1, const Mat& m2);
Mat& sub(Mat& dst, const Mat& m1, const Mat& m2);
Mat& dot(Mat& dst, const Mat& m1, const Mat& m2);
Mat& mul(Mat& dst, const Mat& m1, const Mat& m2);
Mat& mul(Mat& dst, const Mat& m, number x);
number max(const Mat& m);
number sum(const Mat& m);

void rand_seed(int seed);
number rand_number();
Mat& random(Mat& dst);

} // namespace matrix

enum class Act {
  NONE,
  SIGMOID,
  SOFTMAX,
};

class Layer {
 public:
  Layer(int insz, int outsz, Act af = Act::SIGMOID);
  void forward(Row& output);
  void backward(Row& output);
  void reset_grads();

 public:
  Row X, A_grad;
  Mat W, W_grad;
  Row B, B_grad;
  Row Z, Z_grad;

  Act act;
};

class NN {
 public:
  void add_layer(int insz, int outsz, Act act = Act::SIGMOID);
  void random();
  void forward(const Row& input_row, Row& output_row);
  void backward(const Row& pred, const Row& want);
  void backward(const Row& loss);
  inline int nlayers() { return ls_.size(); }
  void start_learn();
  void end_learn(number);
  friend std::ostream& operator<<(std::ostream& os, NN& nn);

 private:
  std::vector<Layer> ls_;
  size_t run_ = 0;
};

#endif // !JPNN_H

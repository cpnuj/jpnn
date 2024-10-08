#include "jpnn.h"

Mat::Mat(int rows, int cols) : rows_(rows), cols_(cols) {
  auto data = new number[rows_*cols_];
  data_.reset(data);
  for (int i = 0; i < rows_*cols_; ++i) {
    data_[i] = 0;
  }
}

number& Mat::operator()(int row, int col) {
  return data_[row * cols_ + col];
}

number Mat::operator()(int row, int col) const {
  return data_[row * cols_ + col];
}

int Mat::rows() const { return rows_; }
int Mat::cols() const { return cols_; }

std::ostream& operator<<(std::ostream& os, const Mat& m) {
  os << "[" << std::endl;
  for (int i = 0; i < m.rows(); ++i) {
    os << std::string(2, ' ');
    for (int j = 0; j < m.cols(); ++j) {
      os << m(i, j) << ", ";
    }
    os << std::endl;
  }
  os << "]" << std::endl;
  return os;
}

number& Row::operator()(int col) {
  return data_[col];
}

number Row::operator()(int col) const {
  return data_[col];
}

void Row::from_mat(const Mat& m, int row) {
  for (int k = 0; k < cols_; ++k) {
    data_[k] = m(row, k);
  }
}

namespace matrix {

// Assign number x to all matrix elements.
Mat& assign(Mat& dst, number x) {
  for (int i = 0; i < dst.rows(); ++i) {
    for (int j = 0; j < dst.cols(); ++j) {
      dst(i, j) = x;
    }  
  }
  return dst;
}

Mat& assign(Mat& dst, const std::vector<number> xs) {
  for (int i = 0; i < dst.rows(); ++i) {
    for (int j = 0; j < dst.cols(); ++j) {
      dst(i, j) = xs[dst.cols()*i+j];
    }  
  }
  return dst;
}

Mat& assign(Mat& dst, const Mat& m) {
  for (int i = 0; i < dst.rows(); ++i) {
    for (int j = 0; j < dst.cols(); ++j) {
      dst(i, j) = m(i, j);
    }  
  }
  return dst;
}

Mat& add(Mat& dst, const Mat& m1, const Mat& m2) {
  for (int i = 0; i < dst.rows(); ++i) {
    for (int j = 0; j < dst.cols(); ++j) {
      dst(i, j) = m1(i, j) + m2(i, j);
    }  
  }
  return dst;
}

Mat& sub(Mat& dst, const Mat& m1, const Mat& m2) {
  for (int i = 0; i < dst.rows(); ++i) {
    for (int j = 0; j < dst.cols(); ++j) {
      dst(i, j) = m1(i, j) - m2(i, j);
    }  
  }
  return dst;
}

Mat& dot(Mat& dst, const Mat& m1, const Mat& m2) {
  for (int i = 0; i < dst.rows(); ++i) {
    for (int j = 0; j < dst.cols(); ++j) {
        dst(i, j) = 0; // reset
      for (int k = 0; k < m1.cols(); ++k) {
        dst(i, j) += m1(i, k) * m2(k, j);
      }
    }  
  }
  return dst;
}

Mat& mul(Mat& dst, const Mat& m, number x) {
  for (int i = 0; i < dst.rows(); ++i) {
    for (int j = 0; j < dst.cols(); ++j) {
      dst(i, j) = m(i, j) * x;
    }  
  }
  return dst;
}

} // namespace matrix


inline number sigmoid(number x) {
  return 1 / (1 + expf(-x));
}

Row& activate(Act a, Row& dst, const Row& src) {
  switch (a) {
  case Act::SIGMOID: {
    for (int i = 0; i < src.cols(); ++i) {
      dst(i) = sigmoid(src(i));
    }
    break;
  }
  }
  return dst;
}

Row& derivative(Act a, Row& dst, const Row& src) {
  switch (a) {
  case Act::SIGMOID: {
    for (int i = 0; i < src.cols(); ++i) {
      dst(i) = sigmoid(src(i)) * (1-sigmoid(src(i)));
    }
    break;
  }
  }
  return dst;
}

Layer::Layer(int insz, int outsz, Act af)
  : X(insz), A_grad(outsz),
    W(insz, outsz), W_grad(insz, outsz),
    B(outsz), B_grad(outsz),
    Z(outsz), Z_grad(outsz),
    act(af) {}

void Layer::forward(Row& output) {
  matrix::dot(Z, X, W);
  matrix::add(Z, Z, B);
  activate(act, output, Z);
}

void Layer::backward(Row& output) {
  matrix::assign(output, 0);
  derivative(act, Z_grad, Z);
  for (int j = 0; j < W_grad.cols(); ++j) {
    B_grad(j) += A_grad(j) * Z_grad(j);
    for (int i = 0; i < W_grad.rows(); ++i) {
      W_grad(i, j) += A_grad(j) * Z_grad(j) * X(i);
      output(i) += A_grad(j) * Z_grad(j) * W(i, j);
    }
  }
}

void Layer::reset_grads() {
  matrix::assign(A_grad, 0);
  matrix::assign(W_grad, 0);
  matrix::assign(B_grad, 0);
  matrix::assign(Z_grad, 0);
}

void NN::start_learn() {
  run_ = 0;
  for (size_t i = 0; i < ls_.size(); ++i) {
    ls_[i].reset_grads();
  }
}

void NN::add_layer(int insz, int outsz, Act act) {
  ls_.push_back(Layer(insz, outsz, act));
}

void NN::forward(const Row& input_row, Row& output_row) {
  // setup first layer's input
  matrix::assign(ls_.front().X, input_row);

  // forward until last layer
  for (size_t i = 0; i < ls_.size()-1; ++i) {
    Layer& curr = ls_[i];
    Layer& next = ls_[i+1];
    curr.forward(next.X);
  }

  ls_.back().forward(output_row);
}

void NN::backward(const Row& pred, const Row& want) {
  matrix::sub(ls_.back().A_grad, pred, want);
  for (int i = ls_.size()-1; i > 0; --i) {
    Layer& curr = ls_[i];
    Layer& prev = ls_[i-1];
    curr.backward(prev.A_grad);
  }
  Row dummy(ls_.front().X.cols());
  ls_.front().backward(dummy);
  run_++;
}

std::ostream& operator<<(std::ostream& os, NN& nn) {
  os << "{" << std::endl;
  for (int i = 0; i < nn.nlayers(); ++i) {
    os << "W_" << i << " = ";
    os << nn.ls_[i].W;
    os << "B_" << i << " = ";
  }
  os << "}" << std::endl;
  return os;
}

void NN::end_learn() {
  for (size_t i = 0; i < ls_.size(); ++i) {
    Layer& l = ls_[i];
    number scale = learning_rate / static_cast<number>(run_);
    matrix::mul(l.W_grad, l.W_grad, scale);
    matrix::mul(l.B_grad, l.B_grad, scale);
    matrix::sub(l.W, l.W, l.W_grad);
    matrix::sub(l.B, l.B, l.B_grad);
  }
}

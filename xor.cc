#include <cassert>
#include <cmath>
#include <vector>
#include <map>
#include <random>
#include <string>
#include <functional>
#include <memory>
#include <iostream>

#include "jpnn.h"

int main() {
  Mat input(4, 2);
  matrix::assign(input,
    { 0, 0,
      0, 1,
      1, 0,
      1, 1,
    }
  );

  Mat output(4, 1);
  matrix::assign(output,
    { 0,
      1,
      1,
      1,
    }
  );

  NN nn;
  nn.add_layer(2, 2);
  nn.add_layer(2, 1);

  Row input_row(input.cols());
  Row output_row(output.cols());
  for (int i = 0; i < input.rows(); ++i) {
    input_row.from_mat(input, i);
    nn.forward(input_row, output_row);
    std::cout << output_row;
  }

  Row pred(output.cols()), want(output.cols());
  int epoch = 10000;
  for (int i = 0; i < epoch; ++i) {
    nn.start_learn();
    for (int j = 0; j < input.rows(); ++j) {
      input_row.from_mat(input, j);
      nn.forward(input_row, pred);
      want.from_mat(output, j);
      nn.backward(pred, want);
    }
    nn.end_learn(1);
  }

  std::cout << nn;

  for (int i = 0; i < input.rows(); ++i) {
    input_row.from_mat(input, i);
    nn.forward(input_row, output_row);
    std::cout << output_row;
  }
}

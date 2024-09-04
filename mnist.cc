#include "jpnn.h"

#include <iostream>
#include <fstream>

using namespace std;

Mat read_mnist(const string file, int ndata, int datasize, int hdsize) {
  Mat mat(ndata, datasize);
  string buf(datasize, '\0');
  ifstream is(file, ios::binary);
  is.read(&buf[0], hdsize);
  for (int i = 0; i < ndata; ++i) {
    is.read(&buf[0], datasize);
    for (int j = 0; j < datasize; ++j) {
      mat(i, j) = static_cast<number>(static_cast<uint8_t>(buf[j]));
    }
  }
  is.close();
  return mat;
}

#define image_hd_size (4 * sizeof(int))
#define label_hd_size (2 * sizeof(int))

#define image_size 28*28
#define label_size 10

#define train_image_file "data/train-images.idx3-ubyte"
#define train_label_file "data/train-labels.idx1-ubyte"
#define train_data_size 60000

Mat train_images(train_data_size, image_size);
Mat train_labels(train_data_size, label_size);

#define test_image_file "data/t10k-images.idx3-ubyte"
#define test_label_file "data/t10k-labels.idx1-ubyte"
#define test_data_size 10000

Mat test_images(test_data_size, image_size);
Mat test_labels(test_data_size, label_size);

void load_dataset() {
  train_images = read_mnist(train_image_file, train_data_size, image_size, image_hd_size);
  for (int i = 0; i < train_data_size; ++i) {
    for (int j = 0; j < image_size; ++j) {
      train_images(i, j) /= 255.0f;
    }
  }
  test_images = read_mnist(test_image_file, test_data_size, image_size, image_hd_size);
  for (int i = 0; i < test_data_size; ++i) {
    for (int j = 0; j < image_size; ++j) {
      test_images(i, j) /= 255.0f;
    }
  }

  auto train_labels_raw = read_mnist(train_label_file, train_data_size, 1, label_hd_size);
  for (int i = 0; i < train_data_size; ++i) {
    int x = static_cast<int>(train_labels_raw(i, 0));
    for (int j = 0; j < label_size; ++j) {
      train_labels(i, j) = (j == x) ? 1 : 0;
    }
  }
  auto test_labels_raw = read_mnist(test_label_file, test_data_size, 1, label_hd_size);
  for (int i = 0; i < test_data_size; ++i) {
    int x = static_cast<int>(test_labels_raw(i, 0));
    for (int j = 0; j < label_size; ++j) {
      test_labels(i, j) = (j == x) ? 1 : 0;
    }
  }
}

number loss_one(const Row& pred, const Row& want) {
  assert(pred.cols() == want.cols());
  number e = 0;
  for (int i = 0; i < pred.cols(); ++i) {
    e += (want(i)-pred(i))*(want(i)-pred(i));
  }
  return e / static_cast<number>(pred.cols());
}

number loss(NN& nn, const Mat& inputs, const Mat& outputs) {
  Row input_row(inputs.cols());
  Row output_row(outputs.cols()), pred_row(outputs.cols());
  number e = 0;
  for (int i = 0; i < inputs.rows(); ++i) {
    matrix::assign(input_row, inputs, i);
    matrix::assign(output_row, outputs, i);
    nn.forward(input_row, pred_row);
    e += loss_one(pred_row, output_row);
  }
  return e / static_cast<number>(inputs.rows());
}

void softmax(Row& dst, const Row& src) {
  assert(dst.cols() == src.cols());
  number max = matrix::max(src);
  for (int i = 0; i < dst.cols(); ++i) {
    dst(i) = exp(src(i)-max);
  }
  number sum = matrix::sum(dst);
  for (int i = 0; i < dst.cols(); ++i) {
    dst(i) /= sum;
  }
}

void cross_entropy_loss(Row& dst, const Row& pred, const Row& want) {
  softmax(dst, pred);
  matrix::sub(dst, dst, want);
}

number cross_entropy_one(const Row& pred, const Row& want) {
  assert(pred.cols() == want.cols());
  for (int i = 0; i < want.cols(); ++i) {
    if (want(i) == 1.0f) {
      return 0.0f - log(pred(i));
    }
  }
  assert(false);
  return 0;
}

number cross_entropy(NN& nn, const Mat& inputs, const Mat& outputs) {
  Row input_row(inputs.cols());
  Row output_row(outputs.cols()), pred_row(outputs.cols());
  number e = 0;
  for (int i = 0; i < inputs.rows(); ++i) {
    matrix::assign(input_row, inputs, i);
    matrix::assign(output_row, outputs, i);
    nn.forward(input_row, pred_row);
    softmax(pred_row, pred_row);
    e += cross_entropy_one(pred_row, output_row);
  }
  return e / static_cast<number>(inputs.rows());
}

int argsmax(const Row& r) {
  int x = 0;
  for (int i = 1; i < r.cols(); ++i) {
    x = (r(i) > r(x)) ?  i : x;
  }
  return x;
}

number correct_rate(NN& nn, const Mat& inputs, const Mat& outputs) {
  number ok = 0;
  Row input_row(inputs.cols());
  Row output_row(outputs.cols()), pred_row(outputs.cols());
  for (int i = 0; i < inputs.rows(); ++i) {
    matrix::assign(input_row, inputs, i);
    matrix::assign(output_row, outputs, i);
    nn.forward(input_row, pred_row);
    softmax(pred_row, pred_row);
    if (argsmax(pred_row) == argsmax(output_row)) {
      ok++;
    }
  }
  return ok / static_cast<number>(inputs.rows());
}

int main() {
  load_dataset();

  random_device rd;
  matrix::rand_seed(rd());

  NN nn;
  nn.add_layer(28*28, 10, Act::NONE);
  // nn.add_layer(16, 10, Act::NONE);

  // nn.random();

  // dint used for SGD
  mt19937 gen(rd());
  uniform_int_distribution<int> dint(0, train_images.rows()-1);

  Row input_row(train_images.cols());
  Row output_row(train_labels.cols());
  Row pred_row(train_labels.cols());
  Row loss_row(train_labels.cols());
  int batch_sz = 100;
  int epoch = 10000;

  cout << "Epoch " << 0 << " loss:"
       << cross_entropy(nn, test_images, test_labels) << endl;

  for (int i = 0; i < epoch; ++i) {
    nn.start_learn();
    for (int j = 0; j < batch_sz; ++j) {
      int k = dint(gen) % train_images.rows();
      // int k = (i * epoch + j) % train_images.rows();
      matrix::assign(input_row, train_images, k);
      nn.forward(input_row, pred_row);
      matrix::assign(output_row, train_labels, k);
      cross_entropy_loss(loss_row, pred_row, output_row);
      nn.backward(loss_row);
    }
    nn.end_learn(1);

    cout << "Epoch " << i
         << " loss: " << cross_entropy(nn, test_images, test_labels)
         << " correct rate: " << correct_rate(nn, test_images, test_labels) << endl;
  }

  return 0;
}

#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/util/command_line_flags.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace tensorflow;
using namespace tensorflow::ops;

//function to read csv

std::vector<std::vector<float>> read_csv(const std::string &input_data(Ag).csv){
    std::ifstream file(input_data(Ag).csv);
    std::vector<std::vector<float>> data;
    std::string line;

     while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> parsedRow;

        while (std::getline(lineStream,cell,',')){
             parsedRow.push_back(std::stof(cell));
        }
        data.push_back(parsedRow);
     }
     return data;
}

//splitting into train and test data
void train_test_split(const std::vector<std::vector<float>> &data,
                      std::vector<std::vector<float>> &X_train, std::vector<std::vector<float>> &X_test,
                      std::vector<float> &y_train, std::vector<float> &y_test, float test_size) {
    int test_size_int = static_cast<int>(data.size() * test_size);

    for (size_t i = 0; i < data.size(); ++i) {
        if (i < test_size_int) {
            X_test.push_back(std::vector<float>(data[i].begin(), data[i].end() - 1));
            y_test.push_back(data[i].back());
        } else {
            X_train.push_back(std::vector<float>(data[i].begin(), data[i].end() - 1));
            y_train.push_back(data[i].back());
        }
    }
}

//main
int main() {
    // Load data
    std::vector<std::vector<float>> data = read_csv("input_data(Ag).csv");
    data.erase(data.begin()); // Remove the first row as header

    std::vector<std::vector<float>> X_train, X_test;
    std::vector<float> y_train, y_test;

    // Split the data into training and test sets
    train_test_split(data, X_train, X_test, y_train, y_test, 0.30);

    // Build the model
    Scope root = Scope::NewRootScope();
    auto input = Placeholder(root.WithOpName("input"), DT_FLOAT);
    auto label = Placeholder(root.WithOpName("label"), DT_FLOAT);

    auto dense1 = Dense(root.WithOpName("dense1"), input, 64, Dense::Relu());
    auto dense2 = Dense(root.WithOpName("dense2"), dense1, 64, Dense::Relu());
    auto output = Dense(root.WithOpName("output"), dense2, 1);

    auto loss = MeanSquaredError(root.WithOpName("loss"), output, label);
    auto optimizer = GradientDescentOptimizer(root.WithOpName("optimizer"), 0.01);
    auto train_op = optimizer.Minimize(loss);

    // Create a session
    ClientSession session(root);

    // Training
    for (int epoch = 0; epoch < 20; ++epoch) {
        for (size_t i = 0; i < X_train.size(); ++i) {
            Tensor x(DT_FLOAT, TensorShape({1, static_cast<int>(X_train[i].size())}));
            std::copy_n(X_train[i].begin(), X_train[i].size(), x.flat<float>().data());

            Tensor y(DT_FLOAT, TensorShape({1, 1}));
            y.scalar<float>()() = y_train[i];

            std::vector<Tensor> outputs;
            TF_CHECK_OK(session.Run({{input, x}, {label, y}}, {train_op}, &outputs));
        }
    }

    double mse = 0.0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        Tensor x(DT_FLOAT, TensorShape({1, static_cast<int>(X_test[i].size())}));
        std::copy_n(X_test[i].begin(), X_test[i].size(), x.flat<float>().data());

        Tensor y(DT_FLOAT, TensorShape({1, 1}));
        y.scalar<float>()() = y_test[i];

        std::vector<Tensor> outputs;
        TF_CHECK_OK(session.Run({{input, x}}, {output}, &outputs));

        auto prediction = outputs[0].scalar<float>()();
        mse += (prediction - y_test[i]) * (prediction - y_test[i]);
    }
    mse /= X_test.size();
    std::cout << "Test MSE: " << mse << std::endl;

    return 0;
}
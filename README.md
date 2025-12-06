What is the best way of learning how a Deep Learning framework works?
Write (a basic) one yourself!

With this project you can:
* Build a neural network with linear layers and three possible
  activations (RELU, Sigmoid, Softmax).
* Optimize it using backpropagation and SGD (Adam on its way).
* Serialize and deserialize your models.

A training program for the MNIST datasets is available as a binary crate
as well as a testing program that runs inference on one random test
image, also printing it to the terminal.

To run the MNIST training program, download the MNIST dataset to
`datastes/mnist/`, then:
```bash
cargo run --release mnist_train
```
The model will be saved to `mnist_model.bin`.
And to test (after training):
```bash
cargo run --release mnist_test
```

The final goal of this project would be to implement and train a
transformer, but let's see...



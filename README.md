What is the best way of learning how a Deep Learning framework works?
Write (a basic) one yourself!

This creates relies on [Faer](https://docs.rs/faer) for matrix
operations, but aside from that, it implements all other other
abstractions with the intention of improving my understanding of how
every piece works.

With this project you can:
* Build a neural network with linear layers and three possible
  activations (RELU, Sigmoid, Softmax).
* Use a loss function on your predictions, currently only Cross Entropy.
* Optimize it using backpropagation and SGD (Adam on its way).
* Serialize and deserialize your models.

A training program for the MNIST dataset is available as a binary crate
as well as a testing program that runs inference on a random test
image, also printing it (and the results) to the terminal.

To run the MNIST training program, download the MNIST dataset to
`datastes/mnist/`, then:
```bash
cargo run --release --bin mnist_train
```
The model will be saved to `mnist_model.bin`.
And to test (after training):
```bash
cargo run --release --bin mnist_test
```

The final goal of this project would be to implement and train a
transformer, but let's see...



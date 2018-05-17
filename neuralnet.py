import numpy as np
from dataloading import loading_data, save_state, load_state, state_saved, loading_data_xor


def initialize_net(input_length, output_length, net_length, layers_length):
    if net_length == len(layers_length):
        layers_length += [output_length]
        net_layers = [np.zeros(input_length)]
        i = 0
        weight_net_layers = []
        for hidden_layer_lenght in layers_length:
            net_layers += [np.zeros(hidden_layer_lenght)]
            columns = len(net_layers[i])
            weight_net_layers += [(np.random.randn(hidden_layer_lenght, columns) * 2.5)/columns]
            i += 1
        return net_layers, weight_net_layers
    else:
        print("net_lenght debe ser igual a la cantidad de elementos de layers_lenght")


def perceptron(w, layer):
    return w.dot(layer)


# parametric relu using 0.01 as betta
def relu(pre_activation_result):
    return np.maximum(0, pre_activation_result)


def sigmoid(pre_activation_result):
    return 1 / (1 + np.exp(-pre_activation_result))


def softmax(pre_activation_result):
    regularized = pre_activation_result / np.max(pre_activation_result)
    classification_num = np.exp(regularized)
    return classification_num/np.sum(classification_num)


def forward_layer(pre_activation_function, activation_function, layer, w):
    return activation_function(pre_activation_function(w, layer))


def forward(net_layers, weights_net, pre_activation_function, activation_function):
    i = 0
    max = len(net_layers)-1
    softmax_activation = []
    while i < max:
        if i == max-1:
            net_layers[i + 1] = softmax(pre_activation_function(weights_net[i], apply_l2_reg(net_layers[i], weights_net[i])))
            softmax_activation = net_layers[i + 1]
        else:
            net_layers[i + 1] = forward_layer(pre_activation_function, activation_function[1], apply_l2_reg(net_layers[i], weights_net[i]),
                                              weights_net[i])
        i += 1
    return net_layers, softmax_activation


def cross_entropy(softmax_activation):
    return (- np.sum(np.log(softmax_activation)))/len(softmax_activation)


def relu_prime(activivation_result):
    return np.array(0 < activivation_result, dtype=int)


def sigmoid_prime(activation_result):
    return activation_result * (1 - activation_result)


def layer_grad(activation_function_prime, layer):
    return activation_function_prime(layer)


def apply_l2_reg(layer, weights):
    return layer - 0.0005 * np.sum(np.power(weights, 2))


def apply_derivative_l2_reg(derivative_layer, weights):
    return derivative_layer - 2 * 0.0005 * weights


def backward(activation_function_prime, weights_net, net_layers, derivative_log_softmax, learning_rate):
    i = len(weights_net) - 1
    delta = apply_derivative_l2_reg(derivative_log_softmax, weights_net[i].T).T
    while i > 0:
        weights_net[i] -= learning_rate * delta
        layer = net_layers[i]
        delta = delta * layer_grad(activation_function_prime[1], layer).T
        delta = apply_derivative_l2_reg(delta, weights_net[i]).T
        i -= 1
    return weights_net


def classify(net_layers, W_net, data):
    net_layers[0] = np.array(data)
    return forward(net_layers, W_net, perceptron, [relu, sigmoid])


def mse(softmax_activation, ground_truth):
    theoretical = softmax_activation[ground_truth]
    no_predicted = np.append(softmax_activation[:ground_truth], softmax_activation[ground_truth+1:])
    return (np.power(theoretical - 1, 2) + np.sum(np.power(no_predicted, 2)))/len(softmax_activation)


def output_loss_derivative(softmax_activation, ground_truth):
    ground_truth_hle = np.zeros(len(softmax_activation))
    ground_truth_hle[ground_truth] = 1
    _, predicted = get_hit(softmax_activation, ground_truth)
    return cross_entropy(softmax_activation), softmax_activation - ground_truth_hle, mse(softmax_activation, ground_truth)


def get_hit(softmax_activation, ground_truth):
    prediction = np.argmax(softmax_activation)
    return int(ground_truth == prediction), prediction


def get_batches(dataset, batch_size, batch_limit):
    batched_train_data = dataset[0][batch_size * batch_limit:batch_size * (batch_limit + 1)]
    batched_train_labels = dataset[1][batch_size * batch_limit:batch_size * (batch_limit + 1)]
    return batched_train_data, batched_train_labels


def accuracy_estimation(validation_data_batch, validation_labels_batch, batch_size, net_layers, W_net):
    epoch_hits = np.zeros(batch_size)
    for i in range(batch_size):
        _, softmax_activation = classify(net_layers, W_net, validation_data_batch[i])
        hit, prediction = get_hit(softmax_activation, validation_labels_batch[i])
        epoch_hits[i] = hit
    epoch_accuracy = (np.sum(epoch_hits, dtype="float") / float(batch_size)) * 100.0
    return epoch_accuracy


def update_learning_rate(learning_rate, epoch):
    if learning_rate >= 0.0085 and epoch % 10 == 0:
        learning_rate -= 0.001
    if learning_rate < 0.0085:
        learning_rate = 0.05
    return learning_rate


def train(net_layers, W_net, train_set, validation_set):
    epoch = 0
    batch_size = 50
    max_batchs = len(train_set[0]) / batch_size
    global_loss = []
    global_accuracy = []
    learning_rate = 0.05
    while epoch < (max_batchs - 1):
        batched_train_data, batched_train_labels = get_batches(train_set, batch_size, epoch)
        epoch_loss = 0
        epoch_error = 0
        epoch_delta = np.zeros(len(net_layers[len(net_layers)-1]))
        for i in range(batch_size):
            net_layers, softmax_activation = classify(net_layers, W_net, batched_train_data[i])
            ground_truth = batched_train_labels[i]
            output_loss, delta, error = output_loss_derivative(softmax_activation, ground_truth)
            epoch_delta += delta
            epoch_error += error
            epoch_loss += output_loss
        W_net = backward([relu_prime, sigmoid_prime], W_net, net_layers, (epoch_error * epoch_delta) / batch_size,
                         learning_rate)
        validation_batch_size = int(batch_size/5) if (batch_size/5) > 1 else 1
        validation_data_batch, validation_labels_batch = get_batches(validation_set, validation_batch_size, epoch)
        epoch_accuracy = accuracy_estimation(validation_data_batch, validation_labels_batch, validation_batch_size,
                                             net_layers, W_net)
        learning_rate = update_learning_rate(learning_rate, epoch)
        print("\nEpoch: ", epoch, ", Error validation_data_batch: ", epoch_error / batch_size, ", Cost function: ",
              epoch_loss / batch_size, ", Accuracy: ", epoch_accuracy)
        global_loss += [epoch_loss/batch_size]
        global_accuracy += [epoch_accuracy]
        save_state(W_net, 1, 128)
        epoch += 1
    return W_net, np.sum(global_accuracy)/len(global_accuracy)


train_set, valid_set, test_set = loading_data()
net_layers, W_net = initialize_net(784, 10, 1, [128])
if (state_saved(1, 128)):
    W_net = load_state(1, 128)


max = 10000
i = 0
treshold = 98
accuracy = 0.0
while i < max and accuracy < treshold:
    print("iter", i)
    W_net, accuracy = train(net_layers, W_net, train_set, valid_set)
    i += 1



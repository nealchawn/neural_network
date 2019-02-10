from neural_network2 import node as Node
import numpy as np

class Network(object):
    def __init__(self, sizes):
        self.layers  = sizes
        self.node_network = []
        self.neg_gradient = []
        #nodes.append([])
        #nodes.append([])

        #np.zeros(5)
        #a1 = np.array([1, 2, 3])
        #a2 = np.array([3, 4])
        #a3 = np.array([a1, a2])
        #a3

        #[2,3,1]
        for i, item in enumerate(self.layers):
            #a = np.zeros(self.layers[i])
            node_layer = []

            #nodes = numpy.zeros((5, 5))

            #each node takes in the amount of inputs it will have
            #which is the count from the previous layer
            j=0
            #3 nodes at middle layer with 2 inputs for each

            #import code; code.interact(local=dict(globals(), **locals()))

            while (j < self.layers[i]):
                if (i -1) <0:
                    #input layer
                    node_layer.append(Node.Node(0))
                else:
                    node_layer.append(Node.Node(self.layers[i - 1]))
                j +=1
            self.node_network.append(node_layer)

    def feed_network(self, inputs = []):
        #[2, 3]

        #input data needs to be same size as first/input layer
        if (len(inputs) != self.layers[0]):
            exit(0)

        # do first layer outside loop
        for index, item in enumerate(inputs):
            #for each mode of input_layer
            self.node_network[0][index].activation_sum([item])
        for index, item in enumerate(self.layers[:-1]):
            temp_input = []

            for idx, node in enumerate(self.node_network[index]):
                temp_input.append(node.output)

            for idx, node in enumerate(self.node_network[index+1]):
                node.activation_sum(temp_input)

    def SGD(self, training_data, mini_batch_size = 10):
        #training data is tuple of many (x,y) x = input, y = answer
        #chop training_data into many tuples the size of mini_batch_size
        #perform back_prop on each then average the -cost_gradient
        #,then apply update.
        n = len(training_data)
        mini_batches = [
            training_data[k:k + mini_batch_size]
            for k in range(0, n, mini_batch_size)
        ]
        for mini_batch in mini_batches:
            # perform back_prop on each then average the -cost_gradient
            # ,then apply update.
            self.update_mini_batch(mini_batch)

    def update_mini_batch(self, mini_batch):
        # perform back_prop on each then average the -cost_gradient
        temp_gradient = []
        for training_input in mini_batch:
            x, y = training_input
            temp_gradient.append(self.back_prop_dev(x, y))
        gradient = np.average(temp_gradient, axis=0)
        self.neg_gradient = gradient
        # ,then apply update. in the correct order
        # order: from output layer to input
        # from first to last node
        #from first to last input
        # each weight then bias
        gradient_index = 0

        for layer in (self.node_network[::-1])[0:(len(self.layers)-1)]:
            for node in layer:# for each node
                nodes_weights = []
                for weight in node.weights:
                    print("weight before: ",weight)
                    weight += gradient[gradient_index]
                    nodes_weights.append(weight)
                    gradient_index += 1
                bias = node.bias + gradient[gradient_index]
                node.set_weights_and_bias(nodes_weights, bias)
                gradient_index += 1

        for layer in (self.node_network[::-1])[0:(len(self.layers)-1)]:
            for node in layer:# for each node
                for weight in node.weights:
                    print("weight after: ",weight)

    #works for 1-dm netowrk
    def back_prop_dev(self, x, y):
        self.feed_network(x)
        last_layer = self.node_network[::-1][0]
        c_over_a = []
        init_cost = self.cost_derivative(last_layer, y)
        for input in last_layer[0].inputs:
            c_over_a.append(init_cost)
        neg_gradient = []
        for layer in (self.node_network[::-1])[0:(len(self.layers) - 1)]:
            #input_count = layer[0].number_of_inputs
            net_a_over_z = []
            for index, input in enumerate(layer[0].inputs):
                net_a_over_z.append(0)
            for node_index, node in enumerate(layer):
                a_over_z = self.sigmoid_prime(node.z_output)
                for input_index, input in enumerate(node.inputs):
                    z_over_w = input
                    neg_gradient.append(c_over_a[node_index]*a_over_z*z_over_w)
                    z_over_a = node.weights[input_index]
                    net_a_over_z[input_index] += c_over_a[node_index]*a_over_z * z_over_a
                neg_gradient.append(c_over_a[node_index] * a_over_z * 1)
            c_over_a = []
            for index, layer_ins in enumerate(net_a_over_z):
                c_over_a.append(layer_ins)
        return neg_gradient

    def back_prop(self, x, y):
        #the order of the gradient is now 1 for every input to a node then its bias
        self.feed_network(x)
        last_layer = self.node_network[::-1][0]
        c_over_a = self.cost_derivative(last_layer, y)
        neg_gradient = []
        net_a_over_z = 0
        for layer in (self.node_network[::-1])[0:(len(self.layers)-1)]:
            for node in layer:# for each node
                a_over_z = self.sigmoid_prime(node.z_output)
                for input in node.inputs: # every dz/dw = a^L-1 = input
                    z_over_w = input #first weight  z_over_w = activation z/w, z =aw +b = a
                    adjustment = 0
                    adjustment = c_over_a*a_over_z*z_over_w
                    neg_gradient.append(adjustment)
                z_over_b = 1 #node.bias remeber partail of z/b z =aw + b = 1
                adjustment = c_over_a * a_over_z * z_over_b
                neg_gradient.append(adjustment)
                #for next layer chain rule
                net_a_over_z += a_over_z
            c_over_a *= net_a_over_z #after each layer multiply the net cost of the previous layer chain rule
            net_a_over_z = 0
        return neg_gradient


    def cost_derivative(self, output_layer, y):
        #oC/oA
        #sum the squares of the difference between the output and actual
        cost =0
        for index, node in enumerate(output_layer):
            cost += 2*(node.output-y[index])
        return cost#/len(output_layer)

    #### Miscellaneous functions
    def sigmoid(self, z):
        """"The sigmoid function."""
        return (1.0 / (1.0 + np.exp(-z)))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function"""
        return (self.sigmoid(z) * (1 - self.sigmoid(z)))
"""
    def compute_network_cost(self, test_data, labels ):
        for layer in enumerate(self.layers[::-1]):
            for index, node in enumerate(layer):
                cost=(node.output - labels[index])**2
"""
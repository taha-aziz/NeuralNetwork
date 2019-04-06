""" Neural Network

Author: Taha Aziz
Date: 03/24/2019

Enhancements in this release:

- Making the Neural Network Interactive!

"""


from abc import *
import collections
from enum import Enum
import random
import math
import json


class DataMismatchError(Exception):
    pass


class NNData:
    class Order(Enum):
        RANDOM = 1
        SEQUENTIAL = 2

    class Set(Enum):
        TRAIN = 1
        TEST = 2

    @staticmethod
    def percentage_limiter(percentage):
        if percentage < 0:
            return 0
        elif percentage > 100:
            return 100
        elif 0 <= percentage <= 100:
            return percentage

    def __init__(self,
                 x=None,
                 y=None,
                 percentage=100):
        if x is None:
            x = []
        if y is None:
            y = []

        self.x = None
        self.y = None
        self.train_percentage = NNData.percentage_limiter(percentage)
        self.train_indices = None
        self.train_pool = None
        self.test_indices = None
        self.test_pool = None

        self.load_data(x, y)

    def load_data(self, x, y):
        size_x = len(x)
        size_y = len(y)
        if size_x != size_y:
            raise DataMismatchError
        self.x = x
        self.y = y
        self.split_set(self.train_percentage)

    def split_set(self,
                  new_train_percentage=None):
        if new_train_percentage is not None:
            self.train_percentage = NNData.percentage_limiter(new_train_percentage)
        loaded_data_size = len(self.x)
        training_set_size = (self.train_percentage * loaded_data_size) // 100
        self.train_indices = random.sample(range(0, loaded_data_size), training_set_size)
        self.test_indices = list(set(range(0, loaded_data_size)) - set(self.train_indices))
        self.prime_data()

    def prime_data(self, my_set=None, order=None):
        if order is None:
            order = NNData.Order.SEQUENTIAL
        test_indices_temp = list(self.test_indices)
        train_indices_temp = list(self.train_indices)
        if order == NNData.Order.RANDOM:
            random.shuffle(test_indices_temp)
            random.shuffle(train_indices_temp)
        if my_set == NNData.Set.TEST:
            self.test_pool = collections.deque(test_indices_temp)
        elif my_set == NNData.Set.TRAIN:
            self.train_pool = collections.deque(train_indices_temp)
        elif my_set is None:
            self.test_pool = collections.deque(test_indices_temp)
            self.train_pool = collections.deque(train_indices_temp)

    def empty_pool(self, my_set=None):
        if my_set is None:
            my_set = NNData.Set.TRAIN
        if my_set is NNData.Set.TEST and len(self.test_pool) == 0:
            return True
        elif my_set is NNData.Set.TRAIN and len(self.train_pool) == 0:
            return True
        return False

    def get_number_samples(self, my_set=None):
        if my_set is NNData.Set.TEST:
            return len(self.test_indices)
        elif my_set is NNData.Set.TRAIN:
            return len(self.train_indices)
        else:
            return len(self.x)

    def get_one_item(self, my_set=None):
        if my_set is None:
            my_set = NNData.Set.TRAIN
        if not self.empty_pool(my_set) and my_set == NNData.Set.TRAIN:
            popped_item = self.train_pool.popleft()
            popped_example = self.x[popped_item]
            popped_label = self.y[popped_item]
            return [popped_example, popped_label]
        elif not self.empty_pool(my_set) and my_set == NNData.Set.TEST:
            popped_item = self.test_pool.popleft()
            popped_example = self.x[popped_item]
            popped_label = self.y[popped_item]
            return [popped_example, popped_label]
        else:
            return None


class MultiLinkNode(ABC):

    def __init__(self):
        self.num_inputs = 0
        self.num_outputs = 0
        self.reporting_inputs = 0
        self.reporting_outputs = 0
        self.compare_inputs_full = 0
        self.compare_outputs_full = 0
        self.input_nodes = collections.OrderedDict([])
        self.output_nodes = collections.OrderedDict([])

    @abstractmethod
    def process_new_input_node(self, node):
        pass

    @abstractmethod
    def process_new_output_node(self, node):
        pass

    def clear_inputs(self):
        self.input_nodes = collections.OrderedDict([])
        self.num_inputs = 0
        self.compare_inputs_full = 0

    def add_input_nodes(self, node):
        self.input_nodes.update({node: None})
        self.num_inputs += 1
        self.compare_inputs_full = 2 ** self.num_inputs - 1

    def clear_outputs(self):
        self.output_nodes = collections.OrderedDict([])
        self.num_outputs = 0
        self.compare_outputs_full = 0

    def add_output_nodes(self, node):
        self.output_nodes.update({node: None})
        self.num_outputs += 1
        self.compare_outputs_full = 2 ** self.num_outputs - 1

    def clear_and_add_input_nodes(self, nodes):
        self.clear_inputs()
        for node in nodes:
            self.add_input_nodes(node)
            self.process_new_input_node(node)

    def clear_and_add_output_nodes(self, nodes):
        self.clear_outputs()
        for node in nodes:
            self.add_output_nodes(node)
            self.process_new_output_node(node)


class LayerType(Enum):
    INPUT = 1
    HIDDEN = 2
    OUTPUT = 3


class Neurode(MultiLinkNode):

    def __init__(self, my_type):
        super().__init__()
        self.value = 0
        if isinstance(my_type, LayerType):
            self.my_type = my_type

    def get_value(self):
        return self.value

    def get_type(self):
        return self.my_type

    def process_new_input_node(self, node):
        weight = random.random()
        self.input_nodes[node] = weight

    def process_new_output_node(self, node):
        pass


class FFNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)

    @staticmethod
    def activate_sigmoid(value):
        return 1 / (1 + math.exp(-value))

    def receive_input(self, from_node=None, input_value=0):
        if self.my_type is LayerType.INPUT:
            self.value = input_value
            for node in self.output_nodes:
                node.receive_input(self)
        else:
            if self.register_input(from_node) is True:
                self.fire()

    def register_input(self, from_node):
        index = list(self.input_nodes.keys()).index(from_node)
        self.reporting_inputs = self.reporting_inputs | (2 ** index)
        if self.reporting_inputs == self.compare_inputs_full:
            self.reporting_inputs = 0
            return True
        return False

    def fire(self):
        weighted_sum = 0
        for node in self.input_nodes:
            each_node_value = self.input_nodes[node] * node.get_value()
            weighted_sum += each_node_value
        self.value = FFNeurode.activate_sigmoid(weighted_sum)
        for node in self.output_nodes:
            node.receive_input(self)


class BPNeurode(Neurode):

    def __init__(self, my_type):
        super().__init__(my_type)
        self.delta = 0
        self.learning_rate = .05

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def receive_back_input(self, from_node, expected=None):
        if self.register_back_input(from_node):
            self.calculate_delta(expected)
            self.back_fire()
            if self.my_type is not LayerType.OUTPUT:
                self.update_weights()

    def register_back_input(self, from_node):
        if self.my_type is LayerType.OUTPUT:
            return True
        index = list(self.output_nodes.keys()).index(from_node)
        self.reporting_outputs = self.reporting_outputs | (2 ** index)
        if self.reporting_outputs == self.compare_outputs_full:
            self.reporting_outputs = 0
            return True
        return False

    def calculate_delta(self, expected=None):
        if self.my_type is LayerType.OUTPUT:
            self.delta = (expected - self.value) * self.value * (1 - self.value)
        elif self.my_type is LayerType.HIDDEN:
            weighted_sum = 0
            for node in self.output_nodes:
                weighted_sum += node.delta * node.get_weight_for_input_node(self)
            self.delta = weighted_sum * BPNeurode.sigmoid_derivative(self.value)

    def update_weights(self):
        for key, node_data in self.output_nodes.items():
            adjustment = key.get_learning_rate() * key.get_delta() * self.value
            key.adjust_input_node(self, adjustment)

    def back_fire(self):
        for node in self.input_nodes:
            node.receive_back_input(self)

    def get_learning_rate(self):
        return self.learning_rate

    def get_delta(self):
        return self.delta

    def get_weight_for_input_node(self, from_node):
        return self.input_nodes[from_node]

    def adjust_input_node(self, node, value):
        self.input_nodes[node] += value


class FFBPNeurode(FFNeurode, BPNeurode):
    pass


class DLLNode:
    """ Node class for a DoublyLinkedList - not designed for
        general clients, so no accessors or exception raising """

    def __init__(self):
        self.prev = None
        self.next = None

    def set_next(self, next_node):
        self.next = next_node

    def get_next(self):
        return self.next

    def set_prev(self, prev_node):
        self.prev = prev_node

    def get_prev(self):
        return self.prev


class DoublyLinkedList:

    def __init__(self):
        self.head = None
        self.tail = None
        self.current = None

    def reset_cur(self):
        self.current = self.head
        return self.current

    def iterate(self):
        if self.current is not None:
            self.current = self.current.get_next()
        return self.current

    def rev_iterate(self):
        if self.current is not None:
            self.current = self.current.get_prev()
        return self.current

    def add_to_head(self, new_node):
        if isinstance(new_node, DLLNode):
            new_node.set_next(self.head)
            if self.head:
                self.head.set_prev(new_node)
            self.head = new_node
            if self.tail is None:
                self.tail = new_node

    def remove_from_head(self):
        if not self.head:
            return None
        ret_node = self.head
        self.head = ret_node.get_next()  # unlink
        if self.head:
            self.head.set_prev(None)
        ret_node.set_next(None)  # don't give client way in
        if self.head is None:
            self.tail = None
        return ret_node

    def insert_after_cur(self, new_node):
        if isinstance(new_node, DLLNode) and self.current:
            new_node.set_next(self.current.get_next())
            new_node.set_prev(self.current)
            if self.current.get_next():
                self.current.get_next().set_prev(new_node)
            self.current.set_next(new_node)
            if self.tail == self.current:
                self.tail = new_node
            return True
        else:
            return False

    def remove_after_cur(self):
        if not self.current or not self.current.get_next():
            return False
        else:
            if self.tail == self.current.get_next():
                self.tail = self.current
            self.current.set_next(self.current.get_next().get_next())
            if self.current.get_next():
                self.current.get_next().set_prev(self.current)


class NodePositionError(Exception):
    pass


class LayerList(DoublyLinkedList):

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        input_layer = Layer(num_inputs, LayerType.INPUT)
        self.add_to_head(input_layer)
        self.reset_cur()
        output_layer = Layer(num_outputs, LayerType.OUTPUT)
        self.insert_after_cur(output_layer)

    def insert_after_cur(self, new_layer):

        if self.current.get_my_type() is LayerType.OUTPUT:
            raise NodePositionError

        # Add input nodes to our new layer
        for neurode in new_layer.get_my_neurodes():
            neurode.clear_and_add_input_nodes(self.current.get_my_neurodes())

        # Add our layer to the previous one
        for neurode in self.current.get_my_neurodes():
            neurode.clear_and_add_output_nodes(new_layer.get_my_neurodes())

        if new_layer.get_my_type() is LayerType.HIDDEN:

            # Add output nodes to our new layer
            for neurode in new_layer.get_my_neurodes():
                neurode.clear_and_add_output_nodes(self.current.get_next().get_my_neurodes())

            # Add our layer to the next one
            for neurode in self.current.next.get_my_neurodes():
                neurode.clear_and_add_input_nodes(new_layer.get_my_neurodes())

        super().insert_after_cur(new_layer)

    def remove_after_cur(self):
        self.iterate()  # move on top of node to remove, to avoid ugly next.next references
        for neurode in self.current.get_next().get_my_neurodes():
            neurode.clear_and_add_input_nodes(self.current.get_prev().get_my_neurodes())

        for neurode in self.current.get_prev().get_my_neurodes():
            neurode.clear_and_add_output_nodes(self.current.get_next().get_my_neurodes())
        self.rev_iterate()  # move back so we can use the parent method
        super().remove_after_cur()

    def insert_hidden_layer(self, num_neurodes=5):
        if self.current == self.tail:
            raise NodePositionError
        hidden_layer = Layer(num_neurodes, LayerType.HIDDEN)
        self.insert_after_cur(hidden_layer)

    def remove_hidden_layer(self):
        if self.current.get_next().get_my_type() is not LayerType.HIDDEN:
            raise NodePositionError
        self.remove_after_cur()

    def get_input_nodes(self):
        return self.head.get_my_neurodes()
        # need to raise error if not set up

    def get_output_nodes(self):
        return self.tail.get_my_neurodes()
        # need to raise error if not set up


class Layer(DLLNode):

    def __init__(self, num_neurodes=5, my_type=LayerType.HIDDEN):
        super().__init__()
        self.my_type = my_type
        self.neurodes = []
        for i in range(num_neurodes):
            self.add_neurode()

    def get_my_neurodes(self):
        return self.neurodes

    def add_neurode(self):
        new_neurode = FFBPNeurode(self.my_type)
        self.neurodes.append(new_neurode)

    def get_layer_info(self):
        return self.my_type, len(self.neurodes)

    def get_my_type(self):
        return self.my_type


class FFBPNetwork:

    class EmptyLayerException(Exception):
        pass

    class EmptySetException(Exception):
        pass

    def __init__(self, num_inputs, num_outputs):
        self.layers = LayerList(num_inputs, num_outputs)
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

    def add_hidden_layer(self, num_neurodes=5):
        if num_neurodes < 1:
            raise FFBPNetwork.EmptyLayerException
        self.layers.insert_hidden_layer(num_neurodes)

    def remove_hidden_layer(self):
        return self.layers.remove_hidden_layer()

    def iterate(self):
        return self.layers.iterate()

    def rev_iterate(self):
        return self.layers.rev_iterate()

    def reset_cur(self):
        return self.layers.reset_cur()

    def get_layer_info(self):
        return self.layers.current.get_layer_info()

    def train(self, data_set: NNData, epochs=1000, verbosity=2, order=NNData.Order.SEQUENTIAL):
        if data_set.get_number_samples(NNData.Set.TRAIN) == 0:
            raise FFBPNetwork.EmptySetException
        for epoch in range(0, epochs):
            data_set.prime_data(order=order)
            sum_error = 0
            while not data_set.empty_pool(NNData.Set.TRAIN):
                x, y = data_set.get_one_item(NNData.Set.TRAIN)
                for j, node in enumerate(self.layers.get_input_nodes()):
                    node.receive_input(None, x[j])
                produced = []
                for j, node in enumerate(self.layers.get_output_nodes()):
                    node.receive_back_input(None, y[j])
                    sum_error += (node.get_value() - y[j]) ** 2 / self.num_outputs
                    produced.append(node.get_value())

                if epoch % 1000 == 0 and verbosity > 1:
                    print("Sample", x, "expected", y, "produced", produced)
            if epoch % 100 == 0 and verbosity > 0:
                print("Epoch", epoch, "RMSE = ", math.sqrt(sum_error / data_set.get_number_samples(NNData.Set.TRAIN)))
        print("Final Epoch RMSE = ", math.sqrt(sum_error / data_set.get_number_samples(NNData.Set.TRAIN)))

    def test(self, data_set: NNData, order=NNData.Order.SEQUENTIAL):
        if data_set.get_number_samples(NNData.Set.TEST) == 0:
            raise FFBPNetwork.EmptySetException
        data_set.prime_data(order=order)
        sum_error = 0
        while not data_set.empty_pool(NNData.Set.TEST):
            x, y = data_set.get_one_item(NNData.Set.TEST)
            for j, node in enumerate(self.layers.get_input_nodes()):
                node.receive_input(None, x[j])
            produced = []
            for j, node in enumerate(self.layers.get_output_nodes()):
                sum_error += (node.get_value() - y[j]) ** 2 / self.num_outputs
                produced.append(node.get_value())

            print(x, ",", y, ",", produced)
        print("RMSE = ", math.sqrt(sum_error / data_set.get_number_samples(NNData.Set.TEST)))


class MultiTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, NNData):
            return {'__NNData__': obj.__dict__}
        if isinstance(obj, collections.deque):
            obj = list(obj)
            return {'__deque__': obj}
        return {type(obj): obj.__dict__}


def multi_type_decoder(obj):
    if "__NNData__" in obj:
        item = obj["__NNData__"]
        return_obj = NNData(item["x"], item["y"], item["train_percentage"])
        return_obj.train_pool = item["train_pool"]
        return_obj.train_indices = item["train_indices"]
        return_obj.test_pool = item["test_pool"]
        return_obj.test_indices = item["test_indices"]
        return return_obj
    if "__deque__" in obj:
        item = obj["__deque__"]
        converted_item = collections.deque(item)
        return converted_item
    return obj


def menu():
    print("\nMain Menu\n" 
          "---------\n" 
          "1 - Load/Reload a JSON object\n" 
          "2 - Browse network layers\n" 
          "3 - Run network\n" 
          "4 - Quit")


def load_object_menu():
    print("\nChoose which object to load\n"
          "---------\n"
          "1 - XOR\n"
          "2 - Sin\n"
          "3 - NAND\n"
          "4 - Quit")


def browse_network_menu():
    print("\nNetwork Menu\n"
          "---------\n"
          "1 - Next layer\n"
          "2 - Previous Layer\n"
          "3 - Add layer\n"
          "4 - Remove layer\n"
          "5 - Quit")


def main():

    network = None
    while True:
        menu()
        try:
            choice = int(input("\nWhat is your choice? "))
            if choice < 1 or choice > 4:
                print("Invalid choice")
                continue
        except ValueError:
            print("*** Please enter an integer only ***")
            continue
        if choice == 1:
            while True:
                load_object_menu()
                try:
                    choice = int(input("\nWhat is your choice? "))
                    if choice < 1 or choice > 4:
                        print("Invalid choice")
                        continue
                except ValueError:
                    print("*** Please enter an integer only ***")
                    continue
                if choice == 1:
                    data = input("\nPlease enter the JSON XOR object data enclosed"
                                 " in curly braces: \n")
                    data_decoded = json.loads(data, object_hook=multi_type_decoder)
                    network = FFBPNetwork(2, 1)
                    network.add_hidden_layer(3)
                elif choice == 2:
                    data = input("\nPlease enter the JSON Sin object data enclosed"
                                 " in curly braces: \n")
                    data_decoded = json.loads(data, object_hook=multi_type_decoder)
                    network = FFBPNetwork(1, 1)
                    network.add_hidden_layer(3)
                elif choice == 3:
                    data = input("\nPlease enter the JSON NAND object data enclosed"
                                 " in curly braces: \n")
                    data_decoded = json.loads(data, object_hook=multi_type_decoder)
                    network = FFBPNetwork(2, 1)
                    network.add_hidden_layer(3)
                elif choice == 4:
                    print("Returning to main menu...")
                    break
        elif choice == 2:
            if network is None:
                print("Please load a JSON object first!")
                continue
            print("Current layer is: " + str(network.get_layer_info()))
            while True:
                browse_network_menu()
                try:
                    choice = int(input("\nWhat is your choice? "))
                    if choice < 1 or choice > 5:
                        print("Invalid choice")
                        continue
                except ValueError:
                    print("*** Please enter an integer only ***")
                    continue
                if choice == 1:
                    if network.layers.current.get_next():
                        network.iterate()
                    else:
                        print("You are at the the network's last layer!")
                    print(network.get_layer_info())
                elif choice == 2:
                    if network.layers.current.get_prev():
                        network.rev_iterate()
                    else:
                        print("You are at the the network's first layer!")
                    print(network.get_layer_info())
                elif choice == 3:
                    if network.layers.current.my_type is LayerType.INPUT or LayerType.HIDDEN:
                        try:
                            num_neurodes = int(input("\nHow many neurodes would you like to add? "))
                            network.add_hidden_layer(num_neurodes)
                        except ValueError:
                            print("*** Please enter an integer only ***")
                            continue
                    elif network.layers.current.my_type is LayerType.OUTPUT:
                        print("Cannot add after output layer!")
                        break
                elif choice == 4:
                    if network.layers.current.my_type is LayerType.HIDDEN:
                        network.remove_hidden_layer()
                    else:
                        print("Cannot remove a non-hidden layer!")
                        break
                elif choice == 5:
                    print("Returning to main menu...")
                    break
        elif choice == 3:
            if network is None:
                print("Please load a JSON object first!")
                continue
            try:
                epochs = int(input("\nPlease enter the number of epochs desired: "))
                network.train(data_decoded, epochs, order=NNData.Order.RANDOM)
            except ValueError:
                print("*** Please enter an integer only ***")
        elif choice == 4:
            print("Quitting Neural Network Runner...")
            break


if __name__ == "__main__":
    main()

""" -------------------- EXAMPLE CONSOLE OUTPUT --------------------
Main Menu
---------
1 - Load/Reload a JSON object
2 - Browse network layers
3 - Run network
4 - Quit

What is your choice? 1

Choose which object to load
---------
1 - XOR
2 - Sin
3 - NAND
4 - Quit

What is your choice? 2

Please enter the JSON Sin object data enclosed in curly braces: 
{"__NNData__": {"train_percentage": 10, "x": [[0.0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08], [0.09], [0.1], [0.11], [0.12], [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23], [0.24], [0.25], [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38], [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47], [0.48], [0.49], [0.5], [0.51], [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63], [0.64], [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72], [0.73], [0.74], [0.75], [0.76], [0.77], [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87], [0.88], [0.89], [0.9], [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98], [0.99], [1.0], [1.01], [1.02], [1.03], [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11], [1.12], [1.13], [1.14], [1.15], [1.16], [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27], [1.28], [1.29], [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42], [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51], [1.52], [1.53], [1.54], [1.55], [1.56], [1.57], [1.58], [1.59], [1.6], [1.61], [1.62], [1.63], [1.64], [1.65], [1.66], [1.67], [1.68], [1.69], [1.7], [1.71], [1.72], [1.73], [1.74], [1.75], [1.76], [1.77], [1.78], [1.79], [1.8], [1.81], [1.82], [1.83], [1.84], [1.85], [1.86], [1.87], [1.88], [1.89], [1.9], [1.91], [1.92], [1.93], [1.94], [1.95], [1.96], [1.97], [1.98], [1.99], [2.0], [2.01], [2.02], [2.03], [2.04], [2.05], [2.06], [2.07], [2.08], [2.09], [2.1], [2.11], [2.12], [2.13], [2.14], [2.15], [2.16], [2.17], [2.18], [2.19], [2.2], [2.21], [2.22], [2.23], [2.24], [2.25], [2.26], [2.27], [2.28], [2.29], [2.3], [2.31], [2.32], [2.33], [2.34], [2.35], [2.36], [2.37], [2.38], [2.39], [2.4], [2.41], [2.42], [2.43], [2.44], [2.45], [2.46], [2.47], [2.48], [2.49], [2.5], [2.51], [2.52], [2.53], [2.54], [2.55], [2.56], [2.57], [2.58], [2.59], [2.6], [2.61], [2.62], [2.63], [2.64], [2.65], [2.66], [2.67], [2.68], [2.69], [2.7], [2.71], [2.72], [2.73], [2.74], [2.75], [2.76], [2.77], [2.78], [2.79], [2.8], [2.81], [2.82], [2.83], [2.84], [2.85], [2.86], [2.87], [2.88], [2.89], [2.9], [2.91], [2.92], [2.93], [2.94], [2.95], [2.96], [2.97], [2.98], [2.99], [3.0], [3.01], [3.02], [3.03], [3.04], [3.05], [3.06], [3.07], [3.08], [3.09], [3.1], [3.11], [3.12], [3.13], [3.14]], "y": [[0.0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957], [0.0399893341866342], [0.0499791692706783], [0.0599640064794446], [0.0699428473375328], [0.0799146939691727], [0.089878549198011], [0.0998334166468282], [0.109778300837175], [0.119712207288919], [0.129634142619695], [0.139543114644236], [0.149438132473599], [0.159318206614246], [0.169182349066996], [0.179029573425824], [0.188858894976501], [0.198669330795061], [0.2084598998461], [0.218229623080869], [0.227977523535188], [0.237702626427135], [0.247403959254523], [0.257080551892155], [0.266731436688831], [0.276355648564114], [0.285952225104836], [0.29552020666134], [0.305058636443443], [0.314566560616118], [0.324043028394868], [0.333487092140814], [0.342897807455451], [0.35227423327509], [0.361615431964962], [0.370920469412983], [0.380188415123161], [0.389418342308651], [0.398609327984423], [0.40776045305957], [0.416870802429211], [0.425939465066], [0.43496553411123], [0.44394810696552], [0.452886285379068], [0.461779175541483], [0.470625888171158], [0.479425538604203], [0.488177246882907], [0.496880137843737], [0.505533341204847], [0.514135991653113], [0.522687228930659], [0.531186197920883], [0.539632048733969], [0.548023936791874], [0.556361022912784], [0.564642473395035], [0.572867460100481], [0.581035160537305], [0.58914475794227], [0.597195441362392], [0.60518640573604], [0.613116851973434], [0.62098598703656], [0.628793024018469], [0.636537182221968], [0.644217687237691], [0.651833771021537], [0.659384671971473], [0.666869635003698], [0.674287911628145], [0.681638760023334], [0.688921445110551], [0.696135238627357], [0.70327941920041], [0.710353272417608], [0.717356090899523], [0.724287174370143], [0.731145829726896], [0.737931371109963], [0.744643119970859], [0.751280405140293], [0.757842562895277], [0.764328937025505], [0.770738878898969], [0.777071747526824], [0.783326909627483], [0.78950373968995], [0.795601620036366], [0.801619940883777], [0.807558100405114], [0.813415504789374], [0.819191568300998], [0.82488571333845], [0.83049737049197], [0.836025978600521], [0.841470984807897], [0.846831844618015], [0.852108021949363], [0.857298989188603], [0.862404227243338], [0.867423225594017], [0.872355482344986], [0.877200504274682], [0.881957806884948], [0.886626914449487], [0.891207360061435], [0.895698685680048], [0.900100442176505], [0.904412189378826], [0.908633496115883], [0.912763940260521], [0.916803108771767], [0.920750597736136], [0.92460601240802], [0.928368967249167], [0.932039085967226], [0.935616001553386], [0.939099356319068], [0.942488801931697], [0.945783999449539], [0.948984619355586], [0.952090341590516], [0.955100855584692], [0.958015860289225], [0.960835064206073], [0.963558185417193], [0.966184951612734], [0.968715100118265], [0.971148377921045], [0.973484541695319], [0.975723357826659], [0.977864602435316], [0.979908061398614], [0.98185353037236], [0.983700814811277], [0.98544972998846], [0.98710010101385], [0.98865176285172], [0.990104560337178], [0.991458348191686], [0.992712991037588], [0.993868363411645], [0.994924349777581], [0.99588084453764], [0.996737752043143], [0.997494986604054], [0.998152472497548], [0.998710143975583], [0.999167945271476], [0.999525830605479], [0.999783764189357], [0.999941720229966], [0.999999682931835], [0.99995764649874], [0.999815615134291], [0.999573603041505], [0.999231634421391], [0.998789743470524], [0.998247974377632], [0.997606381319174], [0.996865028453919], [0.996023989916537], [0.99508334981018], [0.994043202198076], [0.992903651094118], [0.991664810452469], [0.990326804156158], [0.988889766004701], [0.987353839700716], [0.985719178835553], [0.983985946873937], [0.982154317137618], [0.980224472788045], [0.978196606808045], [0.976070921982524], [0.973847630878195], [0.971526955822315], [0.969109128880456], [0.966594391833298], [0.963982996152448], [0.9612752029753], [0.958471283078914], [0.955571516852944], [0.952576194271595], [0.94948561486463], [0.946300087687414], [0.943019931290011], [0.939645473685325], [0.936177052316306], [0.9326150140222], [0.928959715003869], [0.925211520788168], [0.921370806191395], [0.91743795528181], [0.913413361341225], [0.909297426825682], [0.905090563325201], [0.900793191522627], [0.89640574115156], [0.89192865095338], [0.887362368633375], [0.882707350815974], [0.877964062999078], [0.873132979507516], [0.868214583445613], [0.863209366648874], [0.858117829634809], [0.852940481552876], [0.84767784013357], [0.842330431636646], [0.836898790798498], [0.831383460778683], [0.825784993105608], [0.820103947621374], [0.814340892425796], [0.80849640381959], [0.802571066246747], [0.796565472236087], [0.790480222342005], [0.78431592508442], [0.778073196887921], [0.771752662020126], [0.765354952529254], [0.758880708180922], [0.752330576394171], [0.74570521217672], [0.739005278059471], [0.732231444030251], [0.72538438746682], [0.718464793069126], [0.711473352790844], [0.704410765770176], [0.697277738259938], [0.690074983556936], [0.68280322193064], [0.675463180551151], [0.668055593416491], [0.660581201279201], [0.653040751572265], [0.645434998334371], [0.637764702134504], [0.630030629995892], [0.622233555319305], [0.614374257805712], [0.606453523378315], [0.598472144103957], [0.590430918113913], [0.582330649524082], [0.574172148354573], [0.565956230448703], [0.557683717391417], [0.549355436427127], [0.540972220376989], [0.532534907555621], [0.524044341687276], [0.515501371821464], [0.506906852248053], [0.498261642411839], [0.4895666068266], [0.480822614988648], [0.472030541289883], [0.463191264930345], [0.454305669830306], [0.445374644541871], [0.436399082160126], [0.42737988023383], [0.418317940675659], [0.409214169672017], [0.40006947759242], [0.390884778898452], [0.381660992052332], [0.372399039425056], [0.363099847204168], [0.353764345301143], [0.34439346725839], [0.334988150155905], [0.32554933451756], [0.316077964217054], [0.306574986383523], [0.297041351306832], [0.287478012342544], [0.277885925816587], [0.268266050929618], [0.258619349661111], [0.248946786673153], [0.239249329213982], [0.229527947021264], [0.219783612225117], [0.210017299250899], [0.200229984721771], [0.190422647361027], [0.180596267894233], [0.170751828951145], [0.160890314967456], [0.151012712086344], [0.141120008059867], [0.131213192150184], [0.12129325503063], [0.11136118868665], [0.101417986316602], [0.0914646422324372], [0.0815021517602691], [0.0715315111408437], [0.0615537174299131], [0.0515697683985346], [0.0415806624332905], [0.0315873984364539], [0.021590975726096], [0.0115923939361583], [0.00159265291648683]], "train_indices": [8, 13, 44, 48, 58, 67, 69, 70, 71, 75, 77, 83, 102, 112, 127, 130, 143, 164, 166, 188, 214, 219, 223, 228, 240, 243, 257, 260, 286, 301, 308], "test_indices": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 68, 72, 73, 74, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 216, 217, 218, 220, 221, 222, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 258, 259, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 309, 310, 311, 312, 313, 314], "train_pool": {"__deque__": [8, 13, 44, 48, 58, 67, 69, 70, 71, 75, 77, 83, 102, 112, 127, 130, 143, 164, 166, 188, 214, 219, 223, 228, 240, 243, 257, 260, 286, 301, 308]}, "test_pool": {"__deque__": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 68, 72, 73, 74, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 216, 217, 218, 220, 221, 222, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 258, 259, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 309, 310, 311, 312, 313, 314]}}}

Choose which object to load
---------
1 - XOR
2 - Sin
3 - NAND
4 - Quit

What is your choice? 4
Returning to main menu...

Main Menu
---------
1 - Load/Reload a JSON object
2 - Browse network layers
3 - Run network
4 - Quit

What is your choice? 2
Current layer is: (<LayerType.INPUT: 1>, 1)

Network Menu
---------
1 - Next layer
2 - Previous Layer
3 - Add layer
4 - Remove layer
5 - Quit

What is your choice? 1
(<LayerType.HIDDEN: 2>, 3)

Network Menu
---------
1 - Next layer
2 - Previous Layer
3 - Add layer
4 - Remove layer
5 - Quit

What is your choice? 3

How many neurodes would you like to add? 5

Network Menu
---------
1 - Next layer
2 - Previous Layer
3 - Add layer
4 - Remove layer
5 - Quit

What is your choice? 5
Returning to main menu...

Main Menu
---------
1 - Load/Reload a JSON object
2 - Browse network layers
3 - Run network
4 - Quit

What is your choice? 3

Please enter the number of epochs desired: 100
Sample [0.75] expected [0.681638760023334] produced [0.8631719475934436]
Sample [1.88] expected [0.952576194271595] produced [0.8695301225854254]
Sample [1.66] expected [0.996023989916537] produced [0.868509305573032]
Sample [1.64] expected [0.997606381319174] produced [0.8686253083463215]
Sample [0.08] expected [0.0799146939691727] produced [0.8587786926917359]
Sample [0.77] expected [0.696135238627357] produced [0.8621868625094902]
Sample [0.58] expected [0.548023936791874] produced [0.8606082636224981]
Sample [2.57] expected [0.540972220376989] produced [0.871144067715777]
Sample [2.6] expected [0.515501371821464] produced [0.8706772268261482]
Sample [3.08] expected [0.0615537174299131] produced [0.8719284797965563]
Sample [0.48] expected [0.461779175541483] produced [0.8566273335238325]
Sample [2.4] expected [0.675463180551151] produced [0.8669397588995068]
Sample [2.43] expected [0.653040751572265] produced [0.8667143828923716]
Sample [1.02] expected [0.852108021949363] produced [0.8587029015358233]
Sample [0.83] expected [0.737931371109963] produced [0.8574662548536729]
Sample [0.44] expected [0.425939465066] produced [0.8545959795350373]
Sample [1.43] expected [0.990104560337178] produced [0.8601258128423436]
Sample [2.28] expected [0.758880708180922] produced [0.8648202878058233]
Sample [0.7] expected [0.644217687237691] produced [0.85559622455385]
Sample [1.3] expected [0.963558185417193] produced [0.8589987890303327]
Sample [0.13] expected [0.129634142619695] produced [0.8514010621389542]
Sample [1.27] expected [0.955100855584692] produced [0.8576165211431044]
Sample [2.19] expected [0.814340892425796] produced [0.862784315254281]
Sample [2.23] expected [0.790480222342005] produced [0.8628803370727381]
Sample [0.71] expected [0.651833771021537] produced [0.8539999318141798]
Sample [2.14] expected [0.842330431636646] produced [0.8619106372184523]
Sample [2.86] expected [0.277885925816587] produced [0.8650376596458781]
Sample [3.01] expected [0.131213192150184] produced [0.8644620350201457]
Sample [1.12] expected [0.900100442176505] produced [0.8535969525261914]
Sample [0.67] expected [0.62098598703656] produced [0.8507718216563702]
Sample [0.69] expected [0.636537182221968] produced [0.8504373952809173]
Epoch 0 RMSE =  0.34914529546479156
Final Epoch RMSE =  0.27464744909761213

Main Menu
---------
1 - Load/Reload a JSON object
2 - Browse network layers
3 - Run network
4 - Quit

What is your choice? 4
Quitting Neural Network Runner...
"""
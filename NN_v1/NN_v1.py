""" Neural Network

Author: Taha Aziz
Date: 03/18/2019

Enhancements in this release:

- Deploy the JSON encoder/decoder

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
        self.iterate() # move on top of node to remove, to avoid ugly next.next references
        for neurode in self.current.get_next().get_my_neurodes():
            neurode.clear_and_add_input_nodes(self.current.get_prev().get_my_neurodes())

        for neurode in self.current.get_prev().get_my_neurodes():
            neurode.clear_and_add_output_nodes(self.current.get_next().get_my_neurodes())
        self.rev_iterate() # move back so we can use the parent method
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

def main():
    network = FFBPNetwork(2, 1)
    network.add_hidden_layer(3)
    xor_X = [[0, 0], [1, 0], [0, 1], [0, 0]]
    xor_y = [[0], [1], [1], [0]]
    xor_data = NNData(xor_X, xor_y, 50)
    xor_data_encoded = json.dumps(xor_data, cls=MultiTypeEncoder)
    xor_data_decoded = json.loads(xor_data_encoded, object_hook=multi_type_decoder)
    network.train(xor_data_decoded, 10001, order=NNData.Order.RANDOM)
    sin_data = '{"__NNData__": {"train_percentage": 10, "x": [[0.0], [0.01], [0.02], [0.03], [0.04], [0.05], [0.06], [0.07], [0.08], [0.09], [0.1], [0.11], [0.12], [0.13], [0.14], [0.15], [0.16], [0.17], [0.18], [0.19], [0.2], [0.21], [0.22], [0.23], [0.24], [0.25], [0.26], [0.27], [0.28], [0.29], [0.3], [0.31], [0.32], [0.33], [0.34], [0.35], [0.36], [0.37], [0.38], [0.39], [0.4], [0.41], [0.42], [0.43], [0.44], [0.45], [0.46], [0.47], [0.48], [0.49], [0.5], [0.51], [0.52], [0.53], [0.54], [0.55], [0.56], [0.57], [0.58], [0.59], [0.6], [0.61], [0.62], [0.63], [0.64], [0.65], [0.66], [0.67], [0.68], [0.69], [0.7], [0.71], [0.72], [0.73], [0.74], [0.75], [0.76], [0.77], [0.78], [0.79], [0.8], [0.81], [0.82], [0.83], [0.84], [0.85], [0.86], [0.87], [0.88], [0.89], [0.9], [0.91], [0.92], [0.93], [0.94], [0.95], [0.96], [0.97], [0.98], [0.99], [1.0], [1.01], [1.02], [1.03], [1.04], [1.05], [1.06], [1.07], [1.08], [1.09], [1.1], [1.11], [1.12], [1.13], [1.14], [1.15], [1.16], [1.17], [1.18], [1.19], [1.2], [1.21], [1.22], [1.23], [1.24], [1.25], [1.26], [1.27], [1.28], [1.29], [1.3], [1.31], [1.32], [1.33], [1.34], [1.35], [1.36], [1.37], [1.38], [1.39], [1.4], [1.41], [1.42], [1.43], [1.44], [1.45], [1.46], [1.47], [1.48], [1.49], [1.5], [1.51], [1.52], [1.53], [1.54], [1.55], [1.56], [1.57], [1.58], [1.59], [1.6], [1.61], [1.62], [1.63], [1.64], [1.65], [1.66], [1.67], [1.68], [1.69], [1.7], [1.71], [1.72], [1.73], [1.74], [1.75], [1.76], [1.77], [1.78], [1.79], [1.8], [1.81], [1.82], [1.83], [1.84], [1.85], [1.86], [1.87], [1.88], [1.89], [1.9], [1.91], [1.92], [1.93], [1.94], [1.95], [1.96], [1.97], [1.98], [1.99], [2.0], [2.01], [2.02], [2.03], [2.04], [2.05], [2.06], [2.07], [2.08], [2.09], [2.1], [2.11], [2.12], [2.13], [2.14], [2.15], [2.16], [2.17], [2.18], [2.19], [2.2], [2.21], [2.22], [2.23], [2.24], [2.25], [2.26], [2.27], [2.28], [2.29], [2.3], [2.31], [2.32], [2.33], [2.34], [2.35], [2.36], [2.37], [2.38], [2.39], [2.4], [2.41], [2.42], [2.43], [2.44], [2.45], [2.46], [2.47], [2.48], [2.49], [2.5], [2.51], [2.52], [2.53], [2.54], [2.55], [2.56], [2.57], [2.58], [2.59], [2.6], [2.61], [2.62], [2.63], [2.64], [2.65], [2.66], [2.67], [2.68], [2.69], [2.7], [2.71], [2.72], [2.73], [2.74], [2.75], [2.76], [2.77], [2.78], [2.79], [2.8], [2.81], [2.82], [2.83], [2.84], [2.85], [2.86], [2.87], [2.88], [2.89], [2.9], [2.91], [2.92], [2.93], [2.94], [2.95], [2.96], [2.97], [2.98], [2.99], [3.0], [3.01], [3.02], [3.03], [3.04], [3.05], [3.06], [3.07], [3.08], [3.09], [3.1], [3.11], [3.12], [3.13], [3.14]], "y": [[0.0], [0.00999983333416666], [0.0199986666933331], [0.0299955002024957], [0.0399893341866342], [0.0499791692706783], [0.0599640064794446], [0.0699428473375328], [0.0799146939691727], [0.089878549198011], [0.0998334166468282], [0.109778300837175], [0.119712207288919], [0.129634142619695], [0.139543114644236], [0.149438132473599], [0.159318206614246], [0.169182349066996], [0.179029573425824], [0.188858894976501], [0.198669330795061], [0.2084598998461], [0.218229623080869], [0.227977523535188], [0.237702626427135], [0.247403959254523], [0.257080551892155], [0.266731436688831], [0.276355648564114], [0.285952225104836], [0.29552020666134], [0.305058636443443], [0.314566560616118], [0.324043028394868], [0.333487092140814], [0.342897807455451], [0.35227423327509], [0.361615431964962], [0.370920469412983], [0.380188415123161], [0.389418342308651], [0.398609327984423], [0.40776045305957], [0.416870802429211], [0.425939465066], [0.43496553411123], [0.44394810696552], [0.452886285379068], [0.461779175541483], [0.470625888171158], [0.479425538604203], [0.488177246882907], [0.496880137843737], [0.505533341204847], [0.514135991653113], [0.522687228930659], [0.531186197920883], [0.539632048733969], [0.548023936791874], [0.556361022912784], [0.564642473395035], [0.572867460100481], [0.581035160537305], [0.58914475794227], [0.597195441362392], [0.60518640573604], [0.613116851973434], [0.62098598703656], [0.628793024018469], [0.636537182221968], [0.644217687237691], [0.651833771021537], [0.659384671971473], [0.666869635003698], [0.674287911628145], [0.681638760023334], [0.688921445110551], [0.696135238627357], [0.70327941920041], [0.710353272417608], [0.717356090899523], [0.724287174370143], [0.731145829726896], [0.737931371109963], [0.744643119970859], [0.751280405140293], [0.757842562895277], [0.764328937025505], [0.770738878898969], [0.777071747526824], [0.783326909627483], [0.78950373968995], [0.795601620036366], [0.801619940883777], [0.807558100405114], [0.813415504789374], [0.819191568300998], [0.82488571333845], [0.83049737049197], [0.836025978600521], [0.841470984807897], [0.846831844618015], [0.852108021949363], [0.857298989188603], [0.862404227243338], [0.867423225594017], [0.872355482344986], [0.877200504274682], [0.881957806884948], [0.886626914449487], [0.891207360061435], [0.895698685680048], [0.900100442176505], [0.904412189378826], [0.908633496115883], [0.912763940260521], [0.916803108771767], [0.920750597736136], [0.92460601240802], [0.928368967249167], [0.932039085967226], [0.935616001553386], [0.939099356319068], [0.942488801931697], [0.945783999449539], [0.948984619355586], [0.952090341590516], [0.955100855584692], [0.958015860289225], [0.960835064206073], [0.963558185417193], [0.966184951612734], [0.968715100118265], [0.971148377921045], [0.973484541695319], [0.975723357826659], [0.977864602435316], [0.979908061398614], [0.98185353037236], [0.983700814811277], [0.98544972998846], [0.98710010101385], [0.98865176285172], [0.990104560337178], [0.991458348191686], [0.992712991037588], [0.993868363411645], [0.994924349777581], [0.99588084453764], [0.996737752043143], [0.997494986604054], [0.998152472497548], [0.998710143975583], [0.999167945271476], [0.999525830605479], [0.999783764189357], [0.999941720229966], [0.999999682931835], [0.99995764649874], [0.999815615134291], [0.999573603041505], [0.999231634421391], [0.998789743470524], [0.998247974377632], [0.997606381319174], [0.996865028453919], [0.996023989916537], [0.99508334981018], [0.994043202198076], [0.992903651094118], [0.991664810452469], [0.990326804156158], [0.988889766004701], [0.987353839700716], [0.985719178835553], [0.983985946873937], [0.982154317137618], [0.980224472788045], [0.978196606808045], [0.976070921982524], [0.973847630878195], [0.971526955822315], [0.969109128880456], [0.966594391833298], [0.963982996152448], [0.9612752029753], [0.958471283078914], [0.955571516852944], [0.952576194271595], [0.94948561486463], [0.946300087687414], [0.943019931290011], [0.939645473685325], [0.936177052316306], [0.9326150140222], [0.928959715003869], [0.925211520788168], [0.921370806191395], [0.91743795528181], [0.913413361341225], [0.909297426825682], [0.905090563325201], [0.900793191522627], [0.89640574115156], [0.89192865095338], [0.887362368633375], [0.882707350815974], [0.877964062999078], [0.873132979507516], [0.868214583445613], [0.863209366648874], [0.858117829634809], [0.852940481552876], [0.84767784013357], [0.842330431636646], [0.836898790798498], [0.831383460778683], [0.825784993105608], [0.820103947621374], [0.814340892425796], [0.80849640381959], [0.802571066246747], [0.796565472236087], [0.790480222342005], [0.78431592508442], [0.778073196887921], [0.771752662020126], [0.765354952529254], [0.758880708180922], [0.752330576394171], [0.74570521217672], [0.739005278059471], [0.732231444030251], [0.72538438746682], [0.718464793069126], [0.711473352790844], [0.704410765770176], [0.697277738259938], [0.690074983556936], [0.68280322193064], [0.675463180551151], [0.668055593416491], [0.660581201279201], [0.653040751572265], [0.645434998334371], [0.637764702134504], [0.630030629995892], [0.622233555319305], [0.614374257805712], [0.606453523378315], [0.598472144103957], [0.590430918113913], [0.582330649524082], [0.574172148354573], [0.565956230448703], [0.557683717391417], [0.549355436427127], [0.540972220376989], [0.532534907555621], [0.524044341687276], [0.515501371821464], [0.506906852248053], [0.498261642411839], [0.4895666068266], [0.480822614988648], [0.472030541289883], [0.463191264930345], [0.454305669830306], [0.445374644541871], [0.436399082160126], [0.42737988023383], [0.418317940675659], [0.409214169672017], [0.40006947759242], [0.390884778898452], [0.381660992052332], [0.372399039425056], [0.363099847204168], [0.353764345301143], [0.34439346725839], [0.334988150155905], [0.32554933451756], [0.316077964217054], [0.306574986383523], [0.297041351306832], [0.287478012342544], [0.277885925816587], [0.268266050929618], [0.258619349661111], [0.248946786673153], [0.239249329213982], [0.229527947021264], [0.219783612225117], [0.210017299250899], [0.200229984721771], [0.190422647361027], [0.180596267894233], [0.170751828951145], [0.160890314967456], [0.151012712086344], [0.141120008059867], [0.131213192150184], [0.12129325503063], [0.11136118868665], [0.101417986316602], [0.0914646422324372], [0.0815021517602691], [0.0715315111408437], [0.0615537174299131], [0.0515697683985346], [0.0415806624332905], [0.0315873984364539], [0.021590975726096], [0.0115923939361583], [0.00159265291648683]], "train_indices": [8, 13, 44, 48, 58, 67, 69, 70, 71, 75, 77, 83, 102, 112, 127, 130, 143, 164, 166, 188, 214, 219, 223, 228, 240, 243, 257, 260, 286, 301, 308], "test_indices": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 68, 72, 73, 74, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 216, 217, 218, 220, 221, 222, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 258, 259, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 309, 310, 311, 312, 313, 314], "train_pool": {"__deque__": [8, 13, 44, 48, 58, 67, 69, 70, 71, 75, 77, 83, 102, 112, 127, 130, 143, 164, 166, 188, 214, 219, 223, 228, 240, 243, 257, 260, 286, 301, 308]}, "test_pool": {"__deque__": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 68, 72, 73, 74, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 215, 216, 217, 218, 220, 221, 222, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 241, 242, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 258, 259, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 302, 303, 304, 305, 306, 307, 309, 310, 311, 312, 313, 314]}}}'
    sin_decoded = json.loads(sin_data, object_hook=multi_type_decoder)
    network = FFBPNetwork(1, 1)
    network.add_hidden_layer(3)
    network.train(sin_decoded, 10001, order=NNData.Order.RANDOM)


if __name__ == "__main__":
    main()

""" -------------------- CONSOLE OUTPUT --------------------
Sample [0, 0] expected [0] produced [0.6891272181639452]
Sample [1, 0] expected [1] produced [0.7373215158211553]
Epoch 0 RMSE =  0.5214864853785155
Epoch 100 RMSE =  0.5045526750883595
Epoch 200 RMSE =  0.49791522295884255
Epoch 300 RMSE =  0.4954603294062128
Epoch 400 RMSE =  0.49429144828177535
Epoch 500 RMSE =  0.4934525716241461
Epoch 600 RMSE =  0.49266236629134813
Epoch 700 RMSE =  0.49183766750499547
Epoch 800 RMSE =  0.4909402762950994
Epoch 900 RMSE =  0.4899485042015089
Sample [0, 0] expected [0] produced [0.5583604172391011]
Sample [1, 0] expected [1] produced [0.5923865894754478]
Epoch 1000 RMSE =  0.48883281803643275
Epoch 1100 RMSE =  0.48756718868991056
Epoch 1200 RMSE =  0.4861182980666928
Epoch 1300 RMSE =  0.4844526075319164
Epoch 1400 RMSE =  0.48252031578454363
Epoch 1500 RMSE =  0.48026697087444825
Epoch 1600 RMSE =  0.4776287075722097
Epoch 1700 RMSE =  0.47452401631221225
Epoch 1800 RMSE =  0.47085267820332355
Epoch 1900 RMSE =  0.4664999290233055
Sample [0, 0] expected [0] produced [0.527334134141206]
Sample [1, 0] expected [1] produced [0.6158288075861561]
Epoch 2000 RMSE =  0.4613397848176169
Epoch 2100 RMSE =  0.455227797906475
Epoch 2200 RMSE =  0.44803022024863554
Epoch 2300 RMSE =  0.43963685466045826
Epoch 2400 RMSE =  0.4299867988592123
Epoch 2500 RMSE =  0.4190989845344733
Epoch 2600 RMSE =  0.4070883682293506
Epoch 2700 RMSE =  0.39416543928677245
Epoch 2800 RMSE =  0.3806110417945559
Epoch 2900 RMSE =  0.3667320237775139
Sample [0, 0] expected [0] produced [0.3967806845444116]
Sample [1, 0] expected [1] produced [0.6974487085319268]
Epoch 3000 RMSE =  0.35282587461560333
Epoch 3100 RMSE =  0.33914773472301024
Epoch 3200 RMSE =  0.3258934846227873
Epoch 3300 RMSE =  0.3131993016813519
Epoch 3400 RMSE =  0.30114732067240035
Epoch 3500 RMSE =  0.28977675446056844
Epoch 3600 RMSE =  0.2790960664108969
Epoch 3700 RMSE =  0.2690909535663088
Epoch 3800 RMSE =  0.25973365548312655
Epoch 3900 RMSE =  0.2509880504999249
Sample [0, 0] expected [0] produced [0.2679003101562566]
Sample [1, 0] expected [1] produced [0.7851803048643076]
Epoch 4000 RMSE =  0.24281482409442215
Epoch 4100 RMSE =  0.23517270918087674
Epoch 4200 RMSE =  0.22802165244114597
Epoch 4300 RMSE =  0.22132303924413965
Epoch 4400 RMSE =  0.21504038929265837
Epoch 4500 RMSE =  0.2091400898334812
Epoch 4600 RMSE =  0.20359106526225207
Epoch 4700 RMSE =  0.19836484453697664
Epoch 4800 RMSE =  0.1934355352172761
Epoch 4900 RMSE =  0.1887795153204771
Sample [0, 0] expected [0] produced [0.20134218532596854]
Sample [1, 0] expected [1] produced [0.8343200076612459]
Epoch 5000 RMSE =  0.1843753446819914
Epoch 5100 RMSE =  0.18020348111655812
Epoch 5200 RMSE =  0.17624634699946534
Epoch 5300 RMSE =  0.1724878205892916
Epoch 5400 RMSE =  0.16891334288421012
Epoch 5500 RMSE =  0.16550963874914984
Epoch 5600 RMSE =  0.16226467060039157
Epoch 5700 RMSE =  0.15916742491652172
Epoch 5800 RMSE =  0.15620783905933638
Epoch 5900 RMSE =  0.153376777841081
Sample [0, 0] expected [0] produced [0.16359068205867439]
Sample [1, 0] expected [1] produced [0.8634772695947704]
Epoch 6000 RMSE =  0.15066580098635735
Epoch 6100 RMSE =  0.1480672321514416
Epoch 6200 RMSE =  0.1455740352810136
Epoch 6300 RMSE =  0.14317970192612775
Epoch 6400 RMSE =  0.1408782711212808
Epoch 6500 RMSE =  0.1386642489692442
Epoch 6600 RMSE =  0.1365325562532941
Epoch 6700 RMSE =  0.13447850490650015
Epoch 6800 RMSE =  0.1324977536482667
Epoch 6900 RMSE =  0.13058628374765885
Sample [0, 0] expected [0] produced [0.13926960321816328]
Sample [1, 0] expected [1] produced [0.8827304842718685]
Epoch 7000 RMSE =  0.12874036216287243
Epoch 7100 RMSE =  0.1269565286637365
Epoch 7200 RMSE =  0.12523157480577202
Epoch 7300 RMSE =  0.12356247945089273
Epoch 7400 RMSE =  0.12194645115447886
Epoch 7500 RMSE =  0.12038088775091457
Epoch 7600 RMSE =  0.11886334500698055
Epoch 7700 RMSE =  0.1173915408237633
Epoch 7800 RMSE =  0.11596333913712331
Epoch 7900 RMSE =  0.11457672842832185
Sample [1, 0] expected [1] produced [0.8965288604173525]
Sample [0, 0] expected [0] produced [0.12221176311160815]
Epoch 8000 RMSE =  0.11322983654801316
Epoch 8100 RMSE =  0.1119208967975444
Epoch 8200 RMSE =  0.11064825006989454
Epoch 8300 RMSE =  0.10941032511186381
Epoch 8400 RMSE =  0.10820565992739713
Epoch 8500 RMSE =  0.10703285993813426
Epoch 8600 RMSE =  0.1058906163605683
Epoch 8700 RMSE =  0.10477769112020743
Epoch 8800 RMSE =  0.10369291482495349
Epoch 8900 RMSE =  0.10263518046143048
Sample [1, 0] expected [1] produced [0.9069019902208768]
Sample [0, 0] expected [0] produced [0.10944989319757445]
Epoch 9000 RMSE =  0.101603441245349
Epoch 9100 RMSE =  0.10059670398127332
Epoch 9200 RMSE =  0.09961402740294104
Epoch 9300 RMSE =  0.09865451753226169
Epoch 9400 RMSE =  0.09771732340786535
Epoch 9500 RMSE =  0.09680163937281061
Epoch 9600 RMSE =  0.09590670005316332
Epoch 9700 RMSE =  0.09503177338028879
Epoch 9800 RMSE =  0.09417616629824216
Epoch 9900 RMSE =  0.0933392141128062
Sample [1, 0] expected [1] produced [0.9150532379359605]
Sample [0, 0] expected [0] produced [0.09951911593972504]
Epoch 10000 RMSE =  0.0925202864851514
Final Epoch RMSE =  0.0925202864851514
Sample [0.47] expected [0.452886285379068] produced [0.6545106983397069]
Sample [0.65] expected [0.60518640573604] produced [0.6621338524015732]
Sample [3.12] expected [0.021590975726096] produced [0.7246020090407885]
Sample [2.36] expected [0.704410765770176] produced [0.7114390111604233]
Sample [1.58] expected [0.99995764649874] produced [0.6935382812912294]
Sample [1.43] expected [0.990104560337178] produced [0.6900547685917503]
Sample [0.85] expected [0.751280405140293] produced [0.6699901776340729]
Sample [0.06] expected [0.0599640064794446] produced [0.633975759829749]
Sample [1.47] expected [0.994924349777581] produced [0.6910577689957261]
Sample [0.72] expected [0.659384671971473] produced [0.6643188119672311]
Sample [0.79] expected [0.710353272417608] produced [0.6672727268732342]
Sample [1.82] expected [0.969109128880456] produced [0.7019447221109761]
Sample [0.99] expected [0.836025978600521] produced [0.6763161216414422]
Sample [1.46] expected [0.993868363411645] produced [0.6932577675962943]
Sample [1.55] expected [0.999783764189357] produced [0.6969753524018837]
Sample [1.21] expected [0.935616001553386] produced [0.6868759068247023]
Sample [2.9] expected [0.239249329213982] produced [0.7259080079287749]
Sample [2.63] expected [0.4895666068266] produced [0.7204141859214863]
Sample [1.72] expected [0.988889766004701] produced [0.7010228309393528]
Sample [3.07] expected [0.0715315111408437] produced [0.7261542365541596]
Sample [1.8] expected [0.973847630878195] produced [0.701710124983081]
Sample [2.85] expected [0.287478012342544] produced [0.7220421380394041]
Sample [1.31] expected [0.966184951612734] produced [0.6866862418628499]
Sample [0.1] expected [0.0998334166468282] produced [0.6360310603479039]
Sample [2.37] expected [0.697277738259938] produced [0.7126961334236138]
Sample [0.8] expected [0.717356090899523] produced [0.6669821140749612]
Sample [0.25] expected [0.247403959254523] produced [0.642255643521897]
Sample [1.62] expected [0.998789743470524] produced [0.6947760270277699]
Sample [1.74] expected [0.985719178835553] produced [0.6990790918996651]
Sample [1.2] expected [0.932039085967226] produced [0.6834222959490804]
Sample [1.97] expected [0.921370806191395] produced [0.7065519484684593]
Epoch 0 RMSE =  0.32944260885989957
Epoch 100 RMSE =  0.32465579959029517
Epoch 200 RMSE =  0.31961748592914907
Epoch 300 RMSE =  0.31422007329409435
Epoch 400 RMSE =  0.3085733377554182
Epoch 500 RMSE =  0.30304503435815283
Epoch 600 RMSE =  0.2977750106310933
Epoch 700 RMSE =  0.29271784231785547
Epoch 800 RMSE =  0.2878197559684269
Epoch 900 RMSE =  0.2831443484732699
Sample [2.63] expected [0.4895666068266] produced [0.6476418411188164]
Sample [1.62] expected [0.998789743470524] produced [0.7139530257527169]
Sample [0.1] expected [0.0998334166468282] produced [0.49568390506291143]
Sample [1.72] expected [0.988889766004701] produced [0.7087281621543648]
Sample [1.31] expected [0.966184951612734] produced [0.7291913598315228]
Sample [0.85] expected [0.751280405140293] produced [0.7225816174867185]
Sample [0.06] expected [0.0599640064794446] produced [0.4749324514460281]
Sample [1.47] expected [0.994924349777581] produced [0.7236059708972705]
Sample [0.99] expected [0.836025978600521] produced [0.7306092133547083]
Sample [2.36] expected [0.704410765770176] produced [0.6709421067913424]
Sample [1.58] expected [0.99995764649874] produced [0.7207500394829123]
Sample [1.74] expected [0.985719178835553] produced [0.7137822377197197]
Sample [1.8] expected [0.973847630878195] produced [0.7120784636576805]
Sample [1.46] expected [0.993868363411645] produced [0.7310472121217076]
Sample [0.25] expected [0.247403959254523] produced [0.5763542924485204]
Sample [2.9] expected [0.239249329213982] produced [0.6414208858311712]
Sample [0.8] expected [0.717356090899523] produced [0.72026783757915]
Sample [0.47] expected [0.452886285379068] produced [0.6574880170879298]
Sample [0.65] expected [0.60518640573604] produced [0.6989990415239495]
Sample [1.21] expected [0.935616001553386] produced [0.7337437930422136]
Sample [1.2] expected [0.932039085967226] produced [0.7347767900134561]
Sample [3.07] expected [0.0715315111408437] produced [0.6262489772279841]
Sample [3.12] expected [0.021590975726096] produced [0.6156249285388046]
Sample [1.55] expected [0.999783764189357] produced [0.7148877314684354]
Sample [0.79] expected [0.710353272417608] produced [0.7136858619055778]
Sample [1.97] expected [0.921370806191395] produced [0.6911343262254886]
Sample [1.43] expected [0.990104560337178] produced [0.7236082654014759]
Sample [1.82] expected [0.969109128880456] produced [0.7043704302407165]
Sample [0.72] expected [0.659384671971473] produced [0.7089457032864303]
Sample [2.37] expected [0.697277738259938] produced [0.6689912725668716]
Sample [2.85] expected [0.287478012342544] produced [0.6357086195229146]
Epoch 1000 RMSE =  0.2786218955415162
Epoch 1100 RMSE =  0.274290136321669
Epoch 1200 RMSE =  0.2700909111936949
Epoch 1300 RMSE =  0.2660437172848553
Epoch 1400 RMSE =  0.2620832895353042
Epoch 1500 RMSE =  0.25837183348634873
Epoch 1600 RMSE =  0.2547663761494598
Epoch 1700 RMSE =  0.25109991657288205
Epoch 1800 RMSE =  0.24786825536501292
Epoch 1900 RMSE =  0.24451399104450167
Sample [1.58] expected [0.99995764649874] produced [0.7414501245390392]
Sample [1.31] expected [0.966184951612734] produced [0.7654651739865591]
Sample [0.8] expected [0.717356090899523] produced [0.7538734943223864]
Sample [0.72] expected [0.659384671971473] produced [0.739579501743994]
Sample [1.47] expected [0.994924349777581] produced [0.755222783603088]
Sample [1.74] expected [0.985719178835553] produced [0.7310300417636805]
Sample [1.43] expected [0.990104560337178] produced [0.7632993826404219]
Sample [0.06] expected [0.0599640064794446] produced [0.3565141518420661]
Sample [0.47] expected [0.452886285379068] produced [0.6578625624178863]
Sample [1.8] expected [0.973847630878195] produced [0.7276981417245987]
Sample [0.85] expected [0.751280405140293] produced [0.7649616834395403]
Sample [0.25] expected [0.247403959254523] produced [0.5151852832047958]
Sample [0.65] expected [0.60518640573604] produced [0.7258977810091831]
Sample [2.63] expected [0.4895666068266] produced [0.6189197886813528]
Sample [1.46] expected [0.993868363411645] produced [0.7596003718519166]
Sample [2.36] expected [0.704410765770176] produced [0.6566689574953047]
Sample [1.62] expected [0.998789743470524] produced [0.7485057796278042]
Sample [2.37] expected [0.697277738259938] produced [0.6604918708492368]
Sample [1.72] expected [0.988889766004701] produced [0.7418740002088511]
Sample [2.9] expected [0.239249329213982] produced [0.5931780708017855]
Sample [3.07] expected [0.0715315111408437] produced [0.5577556088107801]
Sample [0.99] expected [0.836025978600521] produced [0.7681218523029967]
Sample [3.12] expected [0.021590975726096] produced [0.5343153836803938]
Sample [1.82] expected [0.969109128880456] produced [0.7044990174551942]
Sample [2.85] expected [0.287478012342544] produced [0.5607376463328475]
Sample [1.21] expected [0.935616001553386] produced [0.7594595365211992]
Sample [0.79] expected [0.710353272417608] produced [0.7439193326062918]
Sample [1.2] expected [0.932039085967226] produced [0.7609146006921739]
Sample [1.55] expected [0.999783764189357] produced [0.7369779128113811]
Sample [0.1] expected [0.0998334166468282] produced [0.3854970158128997]
Sample [1.97] expected [0.921370806191395] produced [0.6883464243532885]
Epoch 2000 RMSE =  0.2412894683767259
Epoch 2100 RMSE =  0.23844637597790488
Epoch 2200 RMSE =  0.23543349898305602
Epoch 2300 RMSE =  0.23259342780644712
Epoch 2400 RMSE =  0.2300714966631919
Epoch 2500 RMSE =  0.227376857026682
Epoch 2600 RMSE =  0.2248607895840595
Epoch 2700 RMSE =  0.2225404760819724
Epoch 2800 RMSE =  0.21990618774340775
Epoch 2900 RMSE =  0.2177387873913301
Sample [0.06] expected [0.0599640064794446] produced [0.2765216917596174]
Sample [1.31] expected [0.966184951612734] produced [0.788414113671822]
Sample [1.97] expected [0.921370806191395] produced [0.7040688086624057]
Sample [3.07] expected [0.0715315111408437] produced [0.4943666587773082]
Sample [0.99] expected [0.836025978600521] produced [0.7886697795318447]
Sample [2.36] expected [0.704410765770176] produced [0.6162681958322879]
Sample [1.72] expected [0.988889766004701] produced [0.7385964999998638]
Sample [1.62] expected [0.998789743470524] produced [0.7576139491077934]
Sample [2.85] expected [0.287478012342544] produced [0.5340060546656831]
Sample [1.55] expected [0.999783764189357] produced [0.7631941564284893]
Sample [0.72] expected [0.659384671971473] produced [0.7538670943291536]
Sample [1.58] expected [0.99995764649874] produced [0.7621865022056455]
Sample [0.85] expected [0.751280405140293] produced [0.7817852882939554]
Sample [2.63] expected [0.4895666068266] produced [0.5766905373355482]
Sample [2.37] expected [0.697277738259938] produced [0.6256711988097432]
Sample [1.46] expected [0.993868363411645] produced [0.7785777756585376]
Sample [0.47] expected [0.452886285379068] produced [0.6470737123183269]
Sample [0.8] expected [0.717356090899523] produced [0.7735797488396802]
Sample [1.43] expected [0.990104560337178] produced [0.7822291375778327]
Sample [0.25] expected [0.247403959254523] produced [0.4639318125843509]
Sample [0.1] expected [0.0998334166468282] produced [0.3141780270526721]
Sample [2.9] expected [0.239249329213982] produced [0.5244149899167999]
Sample [1.21] expected [0.935616001553386] produced [0.7908983741107549]
Sample [1.82] expected [0.969109128880456] produced [0.7245166406770792]
Sample [1.74] expected [0.985719178835553] produced [0.7424563070323578]
Sample [1.8] expected [0.973847630878195] produced [0.738238745520376]
Sample [1.47] expected [0.994924349777581] produced [0.7847468595124382]
Sample [1.2] expected [0.932039085967226] produced [0.8044726479648714]
Sample [0.79] expected [0.710353272417608] produced [0.7780353174849088]
Sample [3.12] expected [0.021590975726096] produced [0.5011184090050451]
Sample [0.65] expected [0.60518640573604] produced [0.7314864549652993]
Epoch 3000 RMSE =  0.2156878284482206
Epoch 3100 RMSE =  0.2127193834045277
Epoch 3200 RMSE =  0.21152627738521107
Epoch 3300 RMSE =  0.2094750240488035
Epoch 3400 RMSE =  0.20716961618446753
Epoch 3500 RMSE =  0.2054489762911241
Epoch 3600 RMSE =  0.2035798265771217
Epoch 3700 RMSE =  0.2016239317639259
Epoch 3800 RMSE =  0.19984688270898995
Epoch 3900 RMSE =  0.19903134883376164
Sample [0.8] expected [0.717356090899523] produced [0.779634388698238]
Sample [1.72] expected [0.988889766004701] produced [0.7479359057485787]
Sample [1.47] expected [0.994924349777581] produced [0.7907288844260483]
Sample [2.37] expected [0.697277738259938] produced [0.6168692902555626]
Sample [1.43] expected [0.990104560337178] produced [0.8007614211417319]
Sample [3.07] expected [0.0715315111408437] produced [0.45480561091996896]
Sample [1.8] expected [0.973847630878195] produced [0.7327214551322534]
Sample [1.58] expected [0.99995764649874] produced [0.7765727539490148]
Sample [1.62] expected [0.998789743470524] produced [0.7751871834481086]
Sample [1.46] expected [0.993868363411645] produced [0.8000283660417739]
Sample [1.97] expected [0.921370806191395] produced [0.722081850988875]
Sample [2.63] expected [0.4895666068266] produced [0.5776685986377075]
Sample [0.25] expected [0.247403959254523] produced [0.42682943157052616]
Sample [1.31] expected [0.966184951612734] produced [0.816074234323911]
Sample [0.99] expected [0.836025978600521] produced [0.8183016444002795]
Sample [1.74] expected [0.985719178835553] produced [0.7681703655871037]
Sample [2.9] expected [0.239249329213982] produced [0.5168492471757763]
Sample [0.85] expected [0.751280405140293] produced [0.7980601980282648]
Sample [0.79] expected [0.710353272417608] produced [0.7856450433171909]
Sample [1.21] expected [0.935616001553386] produced [0.8169679110637269]
Sample [2.85] expected [0.287478012342544] produced [0.5069179932882559]
Sample [3.12] expected [0.021590975726096] produced [0.42498411232686906]
Sample [1.55] expected [0.999783764189357] produced [0.7658594321938533]
Sample [2.36] expected [0.704410765770176] produced [0.5946738744467612]
Sample [0.72] expected [0.659384671971473] produced [0.758376679233033]
Sample [0.47] expected [0.452886285379068] produced [0.6289365120976371]
Sample [1.2] expected [0.932039085967226] produced [0.8047652916805779]
Sample [0.65] expected [0.60518640573604] produced [0.7320198402743]
Sample [0.1] expected [0.0998334166468282] produced [0.2606598414704656]
Sample [0.06] expected [0.0599640064794446] produced [0.22371843835813732]
Sample [1.82] expected [0.969109128880456] produced [0.7232870566750109]
Epoch 4000 RMSE =  0.19684087194720976
Epoch 4100 RMSE =  0.19534556130007016
Epoch 4200 RMSE =  0.19397786464315311
Epoch 4300 RMSE =  0.1914155286713641
Epoch 4400 RMSE =  0.19103149318293208
Epoch 4500 RMSE =  0.18921005826505635
Epoch 4600 RMSE =  0.18789974698866424
Epoch 4700 RMSE =  0.18374787897278386
Epoch 4800 RMSE =  0.1846815130924966
Epoch 4900 RMSE =  0.1843830054605374
Sample [0.85] expected [0.751280405140293] produced [0.8015191801228418]
Sample [1.31] expected [0.966184951612734] produced [0.8189086076599917]
Sample [1.55] expected [0.999783764189357] produced [0.7951647032214596]
Sample [1.8] expected [0.973847630878195] produced [0.7568370760317081]
Sample [0.47] expected [0.452886285379068] produced [0.6279243874665068]
Sample [1.47] expected [0.994924349777581] produced [0.8130186264901765]
Sample [1.97] expected [0.921370806191395] produced [0.7301666209554905]
Sample [1.72] expected [0.988889766004701] produced [0.786993182751084]
Sample [2.36] expected [0.704410765770176] produced [0.6504336779778859]
Sample [3.07] expected [0.0715315111408437] produced [0.4589635638557545]
Sample [1.21] expected [0.935616001553386] produced [0.8302628452062529]
Sample [2.9] expected [0.239249329213982] produced [0.4674835968712122]
Sample [0.99] expected [0.836025978600521] produced [0.81966833037288]
Sample [2.37] expected [0.697277738259938] produced [0.6001507443388446]
Sample [0.25] expected [0.247403959254523] produced [0.38972063674015517]
Sample [1.46] expected [0.993868363411645] produced [0.806827433689439]
Sample [1.82] expected [0.969109128880456] produced [0.7517561852479107]
Sample [2.85] expected [0.287478012342544] produced [0.4862221965056348]
Sample [3.12] expected [0.021590975726096] produced [0.3932434993126136]
Sample [0.72] expected [0.659384671971473] produced [0.7584170273734623]
Sample [1.62] expected [0.998789743470524] produced [0.7652205827385898]
Sample [0.79] expected [0.710353272417608] produced [0.7821921559336578]
Sample [0.8] expected [0.717356090899523] produced [0.7839997288170504]
Sample [0.06] expected [0.0599640064794446] produced [0.18769149688076886]
Sample [1.43] expected [0.990104560337178] produced [0.7975141274310085]
Sample [1.58] expected [0.99995764649874] produced [0.7800966021681144]
Sample [1.2] expected [0.932039085967226] produced [0.8226482576308313]
Sample [0.65] expected [0.60518640573604] produced [0.7376281471450528]
Sample [2.63] expected [0.4895666068266] produced [0.5187597584245215]
Sample [0.1] expected [0.0998334166468282] produced [0.22413326797162278]
Sample [1.74] expected [0.985719178835553] produced [0.7548798122209301]
Epoch 5000 RMSE =  0.18272535444177254
Epoch 5100 RMSE =  0.18213182036319683
Epoch 5200 RMSE =  0.1800848802699047
Epoch 5300 RMSE =  0.17902192895187533
Epoch 5400 RMSE =  0.17853208039494503
Epoch 5500 RMSE =  0.17670077479498927
Epoch 5600 RMSE =  0.17559188026414987
Epoch 5700 RMSE =  0.1741984365521572
Epoch 5800 RMSE =  0.17380878228229354
Epoch 5900 RMSE =  0.17177466357041954
Sample [1.58] expected [0.99995764649874] produced [0.7949499973901678]
Sample [0.99] expected [0.836025978600521] produced [0.8293729174324604]
Sample [1.47] expected [0.994924349777581] produced [0.8157949348682108]
Sample [2.85] expected [0.287478012342544] produced [0.453092051612391]
Sample [1.2] expected [0.932039085967226] produced [0.8326204999448649]
Sample [1.21] expected [0.935616001553386] produced [0.8336762481085085]
Sample [3.07] expected [0.0715315111408437] produced [0.37602344932121834]
Sample [2.63] expected [0.4895666068266] produced [0.4794647893222048]
Sample [1.72] expected [0.988889766004701] produced [0.7560805240201981]
Sample [1.82] expected [0.969109128880456] produced [0.7423955045360026]
Sample [0.72] expected [0.659384671971473] produced [0.769781426396512]
Sample [0.65] expected [0.60518640573604] produced [0.7371881294574804]
Sample [0.1] expected [0.0998334166468282] produced [0.19589103483398834]
Sample [1.8] expected [0.973847630878195] produced [0.7515183311411356]
Sample [2.37] expected [0.697277738259938] produced [0.6022207093451991]
Sample [2.9] expected [0.239249329213982] produced [0.44160193389180014]
Sample [0.47] expected [0.452886285379068] produced [0.6074516813695441]
Sample [2.36] expected [0.704410765770176] produced [0.5890314024476848]
Sample [3.12] expected [0.021590975726096] produced [0.3614804402497955]
Sample [1.43] expected [0.990104560337178] produced [0.8035229347811914]
Sample [1.74] expected [0.985719178835553] produced [0.7538118219036374]
Sample [0.06] expected [0.0599640064794446] produced [0.16158398231488552]
Sample [0.25] expected [0.247403959254523] produced [0.35826319228381975]
Sample [0.85] expected [0.751280405140293] produced [0.8031227024208885]
Sample [1.31] expected [0.966184951612734] produced [0.8235785162466079]
Sample [0.8] expected [0.717356090899523] produced [0.7922505809738972]
Sample [1.46] expected [0.993868363411645] produced [0.8099836793000105]
Sample [0.79] expected [0.710353272417608] produced [0.7912941923093647]
Sample [1.62] expected [0.998789743470524] produced [0.7888511546577497]
Sample [1.55] expected [0.999783764189357] produced [0.8057941096231787]
Sample [1.97] expected [0.921370806191395] produced [0.726231174258576]
Epoch 6000 RMSE =  0.17145052218992834
Epoch 6100 RMSE =  0.171324804654833
Epoch 6200 RMSE =  0.17050707608507037
Epoch 6300 RMSE =  0.16897177413631073
Epoch 6400 RMSE =  0.16852939593774632
Epoch 6500 RMSE =  0.16619304177573502
Epoch 6600 RMSE =  0.16683396342029339
Epoch 6700 RMSE =  0.16587333975737414
Epoch 6800 RMSE =  0.1644173013411932
Epoch 6900 RMSE =  0.16343125225538105
Sample [0.79] expected [0.710353272417608] produced [0.8027698772662419]
Sample [3.07] expected [0.0715315111408437] produced [0.3737008095274866]
Sample [1.62] expected [0.998789743470524] produced [0.7916633917941147]
Sample [0.85] expected [0.751280405140293] produced [0.8111445981048618]
Sample [1.2] expected [0.932039085967226] produced [0.8407285607953346]
Sample [0.72] expected [0.659384671971473] produced [0.7702804839251394]
Sample [1.46] expected [0.993868363411645] produced [0.8212719671453221]
Sample [1.47] expected [0.994924349777581] produced [0.8238691019261833]
Sample [2.37] expected [0.697277738259938] produced [0.6024544670721353]
Sample [0.65] expected [0.60518640573604] produced [0.7435932799555697]
Sample [0.1] expected [0.0998334166468282] produced [0.1751405796071271]
Sample [1.43] expected [0.990104560337178] produced [0.8341502496144036]
Sample [0.8] expected [0.717356090899523] produced [0.8057921044189968]
Sample [2.85] expected [0.287478012342544] produced [0.44562598378875273]
Sample [1.74] expected [0.985719178835553] produced [0.7775014710704574]
Sample [2.63] expected [0.4895666068266] produced [0.5154053932020648]
Sample [2.36] expected [0.704410765770176] produced [0.6068428394718755]
Sample [1.55] expected [0.999783764189357] produced [0.8222661895040735]
Sample [0.06] expected [0.0599640064794446] produced [0.14285811395031714]
Sample [1.72] expected [0.988889766004701] produced [0.7987804295100487]
Sample [1.82] expected [0.969109128880456] produced [0.7856150569998321]
Sample [1.97] expected [0.921370806191395] produced [0.7596921372339361]
Sample [0.47] expected [0.452886285379068] produced [0.6177124642004398]
Sample [1.58] expected [0.99995764649874] produced [0.8363228383755321]
Sample [1.8] expected [0.973847630878195] produced [0.8055395703713991]
Sample [0.99] expected [0.836025978600521] produced [0.8582146236155661]
Sample [1.21] expected [0.935616001553386] produced [0.8684781386554772]
Sample [3.12] expected [0.021590975726096] produced [0.42886001459175527]
Sample [1.31] expected [0.966184951612734] produced [0.8490035685979912]
Sample [0.25] expected [0.247403959254523] produced [0.34244990100293043]
Sample [2.9] expected [0.239249329213982] produced [0.4422892854001029]
Epoch 7000 RMSE =  0.16283742803624232
Epoch 7100 RMSE =  0.1623343961170039
Epoch 7200 RMSE =  0.16212390011112052
Epoch 7300 RMSE =  0.16129098031693745
Epoch 7400 RMSE =  0.15968427052220036
Epoch 7500 RMSE =  0.15897960475076608
Epoch 7600 RMSE =  0.15543546227853017
Epoch 7700 RMSE =  0.1557782019135528
Epoch 7800 RMSE =  0.15734633499722625
Epoch 7900 RMSE =  0.15683344253999196
Sample [3.12] expected [0.021590975726096] produced [0.3244138182013608]
Sample [1.2] expected [0.932039085967226] produced [0.8374160743704635]
Sample [1.82] expected [0.969109128880456] produced [0.7457528518511052]
Sample [0.79] expected [0.710353272417608] produced [0.7949446165222378]
Sample [2.85] expected [0.287478012342544] produced [0.3947103126354194]
Sample [1.55] expected [0.999783764189357] produced [0.805357108365716]
Sample [1.97] expected [0.921370806191395] produced [0.7125529975894154]
Sample [2.36] expected [0.704410765770176] produced [0.5962405652888271]
Sample [2.9] expected [0.239249329213982] produced [0.4101993795407296]
Sample [0.72] expected [0.659384671971473] produced [0.7696893332452696]
Sample [1.43] expected [0.990104560337178] produced [0.8303504585358628]
Sample [0.47] expected [0.452886285379068] produced [0.5873445443927051]
Sample [1.46] expected [0.993868363411645] produced [0.8281814185791411]
Sample [0.25] expected [0.247403959254523] produced [0.3164943624572746]
Sample [2.63] expected [0.4895666068266] produced [0.4915053068655094]
Sample [0.99] expected [0.836025978600521] produced [0.8397220816384741]
Sample [2.37] expected [0.697277738259938] produced [0.5895097540663103]
Sample [0.65] expected [0.60518640573604] produced [0.7387613756363701]
Sample [1.31] expected [0.966184951612734] produced [0.847727601394392]
Sample [3.07] expected [0.0715315111408437] produced [0.3508212958228521]
Sample [0.8] expected [0.717356090899523] produced [0.7941863518503524]
Sample [1.74] expected [0.985719178835553] produced [0.7687537105771544]
Sample [0.85] expected [0.751280405140293] produced [0.8122891922812779]
Sample [1.72] expected [0.988889766004701] produced [0.7816927011791799]
Sample [1.8] expected [0.973847630878195] produced [0.7725567711807635]
Sample [1.21] expected [0.935616001553386] produced [0.8553564279874496]
Sample [1.58] expected [0.99995764649874] produced [0.825202018841528]
Sample [1.62] expected [0.998789743470524] produced [0.8238628099572027]
Sample [1.47] expected [0.994924349777581] produced [0.8481225145385137]
Sample [0.06] expected [0.0599640064794446] produced [0.12739070972350347]
Sample [0.1] expected [0.0998334166468282] produced [0.1590452647615037]
Epoch 8000 RMSE =  0.1543312381226111
Epoch 8100 RMSE =  0.15440868265270893
Epoch 8200 RMSE =  0.15039802646986108
Epoch 8300 RMSE =  0.15443301260837977
Epoch 8400 RMSE =  0.1537535897310094
Epoch 8500 RMSE =  0.15405528560991272
Epoch 8600 RMSE =  0.15203545508804311
Epoch 8700 RMSE =  0.15073414886630052
Epoch 8800 RMSE =  0.14638487616193663
Epoch 8900 RMSE =  0.15040653794958156
Sample [2.36] expected [0.704410765770176] produced [0.613695304257609]
Sample [1.55] expected [0.999783764189357] produced [0.8409018327855542]
Sample [0.72] expected [0.659384671971473] produced [0.7845896097339488]
Sample [0.06] expected [0.0599640064794446] produced [0.11412486908806879]
Sample [2.9] expected [0.239249329213982] produced [0.4224848791450952]
Sample [2.63] expected [0.4895666068266] produced [0.4956244426109538]
Sample [1.62] expected [0.998789743470524] produced [0.818888946417836]
Sample [1.31] expected [0.966184951612734] produced [0.8586913489636443]
Sample [0.25] expected [0.247403959254523] produced [0.30285686495251757]
Sample [2.37] expected [0.697277738259938] produced [0.613529932487172]
Sample [1.47] expected [0.994924349777581] produced [0.8516074164757346]
Sample [1.46] expected [0.993868363411645] produced [0.8556247312733587]
Sample [1.97] expected [0.921370806191395] produced [0.766553255163248]
Sample [0.99] expected [0.836025978600521] produced [0.8630162375547551]
Sample [0.1] expected [0.0998334166468282] produced [0.14519069431681025]
Sample [1.58] expected [0.99995764649874] produced [0.8500422165695019]
Sample [0.85] expected [0.751280405140293] produced [0.8390915458179061]
Sample [0.65] expected [0.60518640573604] produced [0.7545942993383873]
Sample [1.82] expected [0.969109128880456] produced [0.8093772693064816]
Sample [3.12] expected [0.021590975726096] produced [0.38721290215081555]
Sample [1.8] expected [0.973847630878195] produced [0.7883307329332155]
Sample [0.47] expected [0.452886285379068] produced [0.588244917602893]
Sample [1.74] expected [0.985719178835553] produced [0.8066977113878244]
Sample [0.79] expected [0.710353272417608] produced [0.8120820150394947]
Sample [1.43] expected [0.990104560337178] produced [0.8561190760208256]
Sample [2.85] expected [0.287478012342544] produced [0.44972916281871245]
Sample [3.07] expected [0.0715315111408437] produced [0.3441792251403469]
Sample [1.21] expected [0.935616001553386] produced [0.848821845711188]
Sample [1.2] expected [0.932039085967226] produced [0.8503962773696564]
Sample [1.72] expected [0.988889766004701] produced [0.7891487504498801]
Sample [0.8] expected [0.717356090899523] produced [0.8052912447985434]
Epoch 9000 RMSE =  0.14955713578424684
Epoch 9100 RMSE =  0.1468101577413104
Epoch 9200 RMSE =  0.14792308564886514
Epoch 9300 RMSE =  0.14857962883112946
Epoch 9400 RMSE =  0.14772498530908504
Epoch 9500 RMSE =  0.1450711107982856
Epoch 9600 RMSE =  0.14570847044014523
Epoch 9700 RMSE =  0.146979623938977
Epoch 9800 RMSE =  0.14576020580547167
Epoch 9900 RMSE =  0.14361762999506855
Sample [1.58] expected [0.99995764649874] produced [0.8452971100481181]
Sample [1.74] expected [0.985719178835553] produced [0.8234663191405192]
Sample [0.72] expected [0.659384671971473] produced [0.7902742332399681]
Sample [1.8] expected [0.973847630878195] produced [0.8150326436119598]
Sample [0.99] expected [0.836025978600521] produced [0.8671409049493737]
Sample [3.12] expected [0.021590975726096] produced [0.3662509363221781]
Sample [1.47] expected [0.994924349777581] produced [0.8477461282892526]
Sample [0.8] expected [0.717356090899523] produced [0.8114220644130056]
Sample [1.2] expected [0.932039085967226] produced [0.8652143410859456]
Sample [1.21] expected [0.935616001553386] produced [0.8661903093916714]
Sample [2.85] expected [0.287478012342544] produced [0.41634926826702656]
Sample [1.97] expected [0.921370806191395] produced [0.7369206947442999]
Sample [0.06] expected [0.0599640064794446] produced [0.10334194073728437]
Sample [1.46] expected [0.993868363411645] produced [0.8527911984489618]
Sample [1.31] expected [0.966184951612734] produced [0.867074660415046]
Sample [2.36] expected [0.704410765770176] produced [0.627193211252031]
Sample [0.79] expected [0.710353272417608] produced [0.8154494957764123]
Sample [0.47] expected [0.452886285379068] produced [0.5815962393760439]
Sample [1.55] expected [0.999783764189357] produced [0.8475997550172236]
Sample [2.63] expected [0.4895666068266] produced [0.5300476085639595]
Sample [0.65] expected [0.60518640573604] produced [0.7433992264535616]
Sample [3.07] expected [0.0715315111408437] produced [0.3457747662311048]
Sample [1.62] expected [0.998789743470524] produced [0.8157503004830435]
Sample [2.9] expected [0.239249329213982] produced [0.3768047876975706]
Sample [0.25] expected [0.247403959254523] produced [0.2808293414978632]
Sample [1.82] expected [0.969109128880456] produced [0.7660274948103124]
Sample [2.37] expected [0.697277738259938] produced [0.5896641169348769]
Sample [0.85] expected [0.751280405140293] produced [0.8253503236684865]
Sample [1.43] expected [0.990104560337178] produced [0.8524613372251022]
Sample [0.1] expected [0.0998334166468282] produced [0.13119900182190525]
Sample [1.72] expected [0.988889766004701] produced [0.8124967184695995]
Epoch 10000 RMSE =  0.1447992156311248
Final Epoch RMSE =  0.1447992156311248
"""
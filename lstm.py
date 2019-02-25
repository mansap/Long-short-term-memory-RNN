"""
There are 3 supported model configurations: "small", "medium", and "large",
each representing a setting of the hyperparameters
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
To run:
$ python lstm_learn.py --data_path=/path-to-data/data/
"""

import random
import sys
import os
from lstm_utils import num_features_head, num_element_types, num_op_types, max_elements, feature_dict
from lstm_utils import num_features
import reader
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


def parse_input():
    # code to parse command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Possible options are: small, medium, large.",
                        default="small")
    parser.add_argument("--data_path", help="data path")
    parser.add_argument("--training_data", help="training_data", default=None)
    parser.add_argument("--testing_data", help="testing_data", default=None)
    parser.add_argument("--use_fp16", help="Train using 16-bit floats instead of 32bit floats", default=False,
                        type=bool)
    parser.add_argument("--summaries_dir", help="Summaries directory", default='/tmp/mnist_log')
    parser.add_argument("--outfile", type=argparse.FileType('w'), default=sys.stdout)
    args = parser.parse_args()
    return args

class Features:
    # geometry = tf.slice(matrix, [0,0], [-1, num_features_head])
    # element_type = tf.slice (matrix, [0, num_features_head], [-1, num_element_types])
    # operation_type = tf.slice(matrix, [0, num_features_head + num_element_types], [-1, num_op_types])
    # element = tf.slice(matrix, [0, num_features_head + num_element_types + num_op_types], [-1, max_elements])

    def __init__(self, matrix):
        num_features_idx = num_features_head #6
        element_type_idx = num_features_head +num_element_types #6+11=17
        operation_type_idx = num_features_head + num_element_types+num_op_types # 6+11+8=25
        max_elements_idx = num_features_head + num_element_types+num_op_types+max_elements # 6+11+8+20=45

        self.geometry = matrix[:,-1,:num_features_idx]
        self.element_type = matrix[:,-1,num_features_idx:element_type_idx]
        self.operation_type = matrix[:,-1,element_type_idx: operation_type_idx]
        self.max_elements = matrix[:,-1,operation_type_idx:max_elements_idx]


class ApparitionModel(nn.Module):
    """The Apparition model."""

    def init_hidden(self, config):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)

        return (Variable(torch.zeros(config.num_layers, 1, config.hidden_size)),
                Variable(torch.zeros((config.num_layers, 1, config.hidden_size))))

    def __init__(self, config, out_size):
        # call init of super class, nn.Module
        super(ApparitionModel, self).__init__()
        self.out_size = num_features #num_features:45
        self.batch_size = config.batch_size

        """
        parameters: 
            1. input_size:  number of expected features in input x
            2. hidden_size: number of features in hidden state h
            3. num_layers: number of recurrent layers
        """
        # Description: creates multilayer LSTM cell
        self.lstm = torch.nn.LSTM(num_features, config.hidden_size, config.num_layers)

        # depicts hidden state
        self.hidden_state = self.init_hidden(config)

    def forward(self, inputs):
        (cell_output, self.hidden) = self.lstm(inputs, self.hidden)
        return cell_output


class SmallConfig(object):
    """Small config."""
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 3
    hidden_size = 45  # num_features #// 2
    max_epoch = 20
    max_max_epoch = 50
    keep_prob = 1
    lr_decay = .995
    batch_size = 1


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = num_features
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1


def prepare_sequence(lst):
    tensor = torch.FloatTensor(lst)
    return Variable(tensor)

def prepare_target(lst):
    tensor = torch.LongTensor(lst)
    return Variable(tensor)


def get_config(args):
    if args.model == "small":
        return SmallConfig()
    elif args.model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", args.model)


def check_arguments(args):
    # report errors for invalid command line inputs
    if not args.data_path:
        raise ValueError("Must set --data_path to apparition animation data directory")
    if not args.training_data:
        raise ValueError("Must set --data_path to apparition animation data directory")
    if not args.testing_data:
        raise ValueError("Must set --data_path to apparition animation data directory")

    if os.path.exists(args.summaries_dir):
        os.removedirs(args.summaries_dir)
    os.mkdir(args.summaries_dir)

def test_functions(m, train_set, config, plot_flag, area_dict):
    count = 0
    for j in range(len(train_set)):

        raw_data = train_set[j]
        # variables used to plot loss Vs num_epochs and training data trace
        train_x_values, train_y_values = create_dict(raw_data[:-1])
        test_data = train_set[j][1:32]
        m.hidden = m.init_hidden(config)

        # use of generated outputs is to plot trace of output
        generated_outputs = []

        for i in range(len(test_data)):
            input = test_data[i]
            input = prepare_sequence(input)
            output = m(input)

            # split outputs to hard_max non-geometry attributes to 1.0
            split_outputs = Features(output)
            temp = []

            # add geometry attributes as it is
            for x in split_outputs.geometry[0].data:
                temp.append(x)

            lst = hard_max(split_outputs.element_type)
            temp.extend(lst)

            lst = hard_max(split_outputs.operation_type)
            temp.extend(lst)

            lst = hard_max(split_outputs.max_elements)
            temp.extend(lst)

            generated_outputs.append([temp])

        # take last output as input, and repeat this in a loop
        input = output[0]

        for i in range(100):
            output = m(input)

            # split outputs to hard_max non-geometry attributes to 1.0
            split_outputs = Features(output)
            temp = []
            # add geometry attributes as it is
            for x in split_outputs.geometry[0].data:
                temp.append(x)

            lst = hard_max(split_outputs.element_type)
            temp.extend(lst)

            lst = hard_max(split_outputs.operation_type)
            temp.extend(lst)

            lst = hard_max(split_outputs.max_elements)
            temp.extend(lst)

            generated_outputs.append([temp])
            input = output[0]

        output_x_values, output_y_values = create_dict(generated_outputs[1:])
        count+=1
        # plot trace of input and output animation
        plot_trace(train_x_values, train_y_values, output_x_values, output_y_values, plot_flag, area_dict, count)


"""
This function runs the test data on the model, writes the output to output file and 
plots the graphs
"""


def test_model(args, m, test_data, config):
    op_file = args.outfile
    m.hidden = m.init_hidden(config)

    input = test_data[0]
    input = prepare_sequence(input)

    temp = []
    for x in input[0].data:
        temp.append(x)
    op_file.write("%s\n" % temp)

    print("running on test data...")
    for i in range(len(test_data)):
        input = test_data[i]
        input = prepare_sequence(input)
        output = m(input)

        # split outputs to hard_max non-geometry attributes to 1.0
        split_outputs = Features(output)
        temp = []

        # add geometry attributes as it is
        for x in split_outputs.geometry[0].data:
            temp.append(x)

        lst = hard_max(split_outputs.element_type)
        temp.extend(lst)

        lst = hard_max(split_outputs.operation_type)
        temp.extend(lst)

        lst = hard_max(split_outputs.max_elements)
        temp.extend(lst)

        op_file.write("%s\n" % temp)

    # take last output as input, and repeat this in a loop
    input = output[0]

    for i in range(1000):
        output = m(input)

        # split outputs to hard_max non-geometry attributes to 1.0
        split_outputs = Features(output)
        temp = []
        # add geometry attributes as it is
        for x in split_outputs.geometry[0].data:
            temp.append(x)

        lst = hard_max(split_outputs.element_type)
        temp.extend(lst)

        lst = hard_max(split_outputs.operation_type)
        temp.extend(lst)

        lst = hard_max(split_outputs.max_elements)
        temp.extend(lst)

        op_file.write("%s\n" % temp)
        input = output[0]

    # adding end instance
    print("adding last line...")
    last = [0.0] * num_features
    last[feature_dict["optype_end"]] = 1.0
    op_file.write("%s\n" % last)


def calculate_loss(targets,outputs):
    class_values = buildClass(targets)
    class_values = prepare_target(class_values)
    loss_values = nn.CrossEntropyLoss()(outputs, class_values)
    return class_values,loss_values


def main():

    # parse command line inputs
    args = parse_input()
    # check arguments for errors
    check_arguments(args)

    #read the animation data into a 3D list raw_data
    raw_data = reader.apparition_raw_data(args.data_path, args.training_data)

    loss_dict={}
    training_loss={}
    gap = get_config(args).max_max_epoch//20
    if gap < 1:
        gap =1

    # code to split raw_data if multiple animations are present
    multi_data = split_animations(raw_data)

    # read the test animation into a 3d list test_data
    test_data = reader.apparition_raw_data_test(args.data_path, args.testing_data)

    # get current configuration for model
    config = get_config(args)

    # create apparition model which inherits nn.Module class
    m = ApparitionModel(config, num_features)

    # loss function and optimizer
    # function to automatically apply the gradients after backprop
    optimizer = optim.SGD(m.parameters(), lr=0.1)

    print ("training...")
    num_epochs = 0
    training_size = 0
    area_list = []

    # --------------train the model------------------------

    for epoch in range(config.max_max_epoch):
        training_size = 0
        train_set = []
        random_idx = random.randint(0, len(multi_data) - 1)
        set = multi_data[random_idx]
        train_set.append(set)
        for data in multi_data:
            training_size+=1

            #run the model on each animation in multi-data
            targets = data[1:]
            inputs = data[:-1]

            # Step 1.
            # Pytorch accumulates gradients,clear them out before each instance
            # Also, clear out the hidden state of the LSTM, to clear history of last instance
            m.hidden = m.init_hidden(config)

            # Step 2. Get our inputs ready for the network, that is, turn them into 3D tensors
            var_inputs = prepare_sequence(inputs)
            targets = prepare_sequence(targets)

            # Step 3. Run our forward pass.
            # m(inputs) works by defining __call__ in one of it's ancestor call.
            outputs = m(var_inputs)

            # break it up into 4 parts
            sliced_outputs = Features(outputs)
            sliced_targets = Features(targets)

            #initialze the gradients to zero after every epoch run
            m.zero_grad()

            # L1 loss for geometry attributes, cross_entropy for rest

            # calculate L1 loss for geometry features
            loss_geometry = nn.L1Loss()(sliced_outputs.geometry, sliced_targets.geometry)

            # calculate cross_entropy_loss loss for element_type features
            element_type_class, loss_element_type = \
                calculate_loss(sliced_targets.element_type,sliced_outputs.element_type)

            # calculate cross_entropy_loss loss for operation_type features
            operation_type_class, loss_operation_type = \
                calculate_loss(sliced_targets.operation_type,sliced_outputs.operation_type)

            # calculate cross_entropy_loss loss for max_elements features
            max_elements_class, loss_max_elements = \
                calculate_loss(sliced_targets.max_elements,sliced_outputs.max_elements)

            # create a list of all losses
            losses = [loss_geometry, loss_element_type, loss_operation_type, loss_max_elements]

            # sum total loss
            total_loss = sum(losses)

            #backpropagate
            total_loss.backward()
            optimizer.step()
            for x in total_loss.data:
                training_loss[training_size] = x

        # storing loss after "gap" number of epochs
        print ("epoch %d complete" % num_epochs)
        if len(total_loss.data) > 0:
            for x in total_loss.data:
                if num_epochs%gap == 0:
                    print("loss",x,"epoch",num_epochs)
                    loss_dict[num_epochs+gap] = x
        num_epochs += 1
        # keeps tab of area between input and output for each epoch for a randomly chose sample
        test_functions(m,train_set,config,False, area_list)

    print("\n")
    print("Choosing random testing samples chosen for plotting results")
    train_set =[]
    for i in range(3):
        random_idx = random.randint(0,len(multi_data)-1)
        print(multi_data[0])
        set = multi_data[random_idx]
        train_set.append(set)
    # plots area between input and output for final epoch for 3 randomly chosen samples
    test_functions(m, train_set, config,True, area_list)

    print("\n")
    # --------------test the model------------------------
    test_model(args,m,test_data,config)

    # plot loss vs num_epochs
    plot_loss(loss_dict)
    # plot graph between training size and loss
    plot_training_loss(training_loss,training_size, config.max_max_epoch)

    # plot area Vs num of epochs
    plot_area(area_list)

#input is a Variable type, split_outputs
def hard_max(input):
    temp = []
    for x in input[0].data:
        temp.append(x)
    max_val = max(temp)
    idx = temp.index(max_val)
    lst = [0.0] * len(temp)
    lst[idx] = 1.0
    #print(lst)
    return lst

def buildClass(input):
    result = []
    for instance in input.data:
        class_val =0
        flag = False
        temp=[]
        for idx in range(len(instance)):
            temp.append(instance[idx])
        if 1.0 in temp:
            idx = temp.index(1.0)
            class_val = idx
        result.append(class_val)
    return result

def plot_area(area_list):
    plt.clf()
    plt.cla()
    plt.close()
    gap = len(area_list) //20
    if gap == 0:
        gap =1
    x_list = range(0, len(area_list),gap)

    block_area = []
    for k in range(0, len(area_list), gap):
        block_area.append(area_list[k])

    plt.bar(x_list, block_area, color=['orange'])
    plt.xlabel('number of epochs')
    # naming the y-axis
    plt.ylabel('area between input and output trace')
    # plot title
    plt.title('Area Vs Number of Epochs ')
    plt.savefig("Plot_Area_Vs_Epochs.png")


def plot_training_loss(training_loss, training_size, epoch_num):
    plt.clf()
    plt.cla()
    plt.close()
    gap = training_size //20
    if gap == 0:
        gap =1
    y_values = []
    x_values = []
    for i in range(1,training_size+1,gap):
        y_values.append(training_loss[i])
        x_values.append(i)
    plt.bar(x_values, y_values, color=['green'])

    plt.xlabel('training_size')
    # naming the y-axis
    plt.ylabel('loss')
    # plot title
    plt.title('loss Vs Training size with ' + str(epoch_num) + " epochs")
    plt.savefig("Plot_Loss_Vs_TrainingSize.png")


def plot_loss(loss_dict):
    plt.clf()
    plt.cla()
    plt.close()
    x_values = loss_dict.keys()
    y_values = loss_dict.values()
    plt.bar(x_values,y_values,color=['red'])

    plt.xlabel('num_epochs')
    # naming the y-axis
    plt.ylabel('loss')
    # plot title
    plt.title('loss Vs num_epochs')

    # function to show the plot
    plt.savefig("Plot_Loss_Vs_Epochs.png")

def create_dict(data):
    x_values=[]
    y_values=[]
    for i in range(len(data)):
        # x coordinate is key and y coordinate is value
        x_values.append(data[i][0][0])
        y_values.append(data[i][0][1])
    return x_values,y_values

def plot_trace(input_x,input_y,output_x,output_y, plot_flag, area_list, count):
    plt.clf()
    plt.cla()
    plt.close()
    # else log area per epoch
    if plot_flag== False:
        poly = []
        x_idx = 0
        while x_idx < len(input_x):
            poly.append((input_x[x_idx], input_y[x_idx]))
            x_idx += 1
        x_idx = len(output_x) - 1
        while x_idx >= 0:
            poly.append((output_x[x_idx], output_y[x_idx]))
            x_idx -= 1
        poly.append((input_x[0], input_y[0]))
        axes = plt.gca()
        axes.add_patch(Polygon(poly, closed=True, facecolor='yellow'))
        x1, y1 = [output_x[0], input_x[0]], [output_y[0], input_y[0]]
        plt.plot(x1, y1)
        area = 0.0

        n = len(poly)
        for i in range(n):
            i1 = (i + 1) % n
            area += poly[i][0] * poly[i1][1] - poly[i1][0] * poly[i][1]
        area *= 0.5
        area = abs(area)
        area_list.append(area)
        return

    plt.plot(input_x,input_y, label = "input", color = 'red')
    plt.plot(output_x, output_y, label="output",color = 'blue')
    x1, y1 = [output_x[len(output_x)-1],input_x[len(input_x)-1]],[output_y[len(output_y)-1],input_y[len(input_y)-1]]
    plt.plot(x1,y1)
    x1, y1 = [output_x[0], input_x[0]], [output_y[0],input_y[0]]

    poly = []
    x_idx = 0
    while x_idx < len(input_x):
        poly.append((input_x[x_idx], input_y[x_idx]))
        x_idx+=1
    x_idx = len(output_x) - 1
    while x_idx >= 0:
        poly.append((output_x[x_idx], output_y[x_idx]))
        x_idx-=1
    poly.append((input_x[0], input_y[0]))
    axes = plt.gca()
    axes.add_patch(Polygon(poly, closed=True, facecolor='yellow'))
    plt.plot(x1, y1)

    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    # giving a title to my graph
    plt.title('Input Vs Output trace')

    # show a legend on the plot
    plt.legend()
    # plot when final epoch
    if plot_flag:
        # function to show the plot
        plt.savefig("Plot_trace_output" + str(count) + ".png")
        plt.clf()
        plt.cla()
        plt.close()

def split_animations(raw_data):

    multi_data =[]
    temp_data = []

    for i in range(len(raw_data)):
        temp_data.append([raw_data[i][0]])
        if raw_data[i][0][feature_dict["optype_end"]] == 1.0:
            multi_data.append(temp_data)
            temp_data = []
    return multi_data


main()
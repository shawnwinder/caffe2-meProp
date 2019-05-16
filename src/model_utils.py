from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import os
import sys
import time

from caffe2.python import (
    workspace,
    model_helper,
    core,
    brew,
    utils,
    optimizer,
)
from caffe2.proto import caffe2_pb2

from caffe2.python.optimizer import Optimizer
from LeNet import (
    add_lenet,
    add_lenet_fc_topk,
    add_lenet_conv_topk,
    add_lenet_full_topk,

)



##############################################################################
# model maintaining utils
##############################################################################
def load_model(model, init_net_pb, predict_net_pb):
    ''' load init and predict net from .pb file for model validation/testing
        model: current model
        init_net: the .pb file of the init_net
        predict_net: the .pb file of the predict_net
    '''
    # Make sure both nets exists
    if (not os.path.exists(init_net_pb)) or (not os.path.exists(predict_net_pb)):
            print("ERROR: input net.pb not found!")

    # Append net
    init_net_proto = caffe2_pb2.NetDef()
    with open(init_net_pb, 'r') as f:
        init_net_proto.ParseFromString(f.read())
    model.param_init_net = model.param_init_net.AppendNet(core.Net(init_net_proto))

    predict_net_proto = caffe2_pb2.NetDef()
    with open(predict_net_pb, 'r') as f:
        predict_net_proto.ParseFromString(f.read())
    model.net = model.net.AppendNet(core.Net(predict_net_proto))


def load_init_net(init_net_pb, device_opt):
    ''' load params of pretrained init_net on given device '''
    init_net_proto = caffe2_pb2.NetDef()
    with open(init_net_pb, 'rb') as f:
        init_net_proto.ParseFromString(f.read())
        for op in init_net_proto.op:
            op.device_option.CopyFrom(device_opt)
    workspace.RunNetOnce(core.Net(init_net_proto))


def snapshot_init_net(params, workspace, snapshot_prefix, snapshot_name,
                      postfix, epoch):
    ''' save the model init_net as .pb file periodically '''
    timestamp = time.time()
    timestamp_s = time.strftime('%m%d-%H:%M', time.localtime(timestamp))
    init_net_snapshot = os.path.join(
        snapshot_prefix,
        '{}_init_net_{}_epoch-{}_{}.pb'.format(
            snapshot_name, postfix, epoch, timestamp_s),
    )

    init_net_proto = caffe2_pb2.NetDef()
    for param in params:
        blob = workspace.FetchBlob(param)
        shape = blob.shape
        op = core.CreateOperator(
            'GivenTensorFill',
            [],
            [param],
            arg=[
                utils.MakeArgument('shape', shape),
                utils.MakeArgument('values', blob)
            ]
        )
        init_net_proto.op.extend([op])
    with open(init_net_snapshot, 'wb') as f:
        f.write(init_net_proto.SerializeToString())


##############################################################################
# model construction utils
##############################################################################
def add_input(model, config, is_test=False):
    """
    Add an database input data
    """
    if is_test:
        db_reader = model.CreateDB(
            "val_db_reader",
            db=config['evaluate_data']['data_path'],
            db_type=config['evaluate_data']['data_format'],
        )
        data, label = brew.image_input(
            model,
            db_reader,
            ['data', 'label'],
            batch_size=config['evaluate_data']['input_transform']['batch_size'],
            use_gpu_transform=config['evaluate_data']['input_transform']['use_gpu_transform'],
            scale=config['evaluate_data']['input_transform']['scale'],
            crop=config['evaluate_data']['input_transform']['crop_size'],
            mean_per_channel=config['evaluate_data']['input_transform']['mean_per_channel'],
            std_per_channel=config['evaluate_data']['input_transform']['std_per_channel'],
            mirror=True,
            is_test=True,
        )
    else:
        db_reader = model.CreateDB(
            "train_db_reader",
            db=config['training_data']['data_path'],
            db_type=config['training_data']['data_format'],
        )
        data, label = brew.image_input(
            model,
            db_reader,
            ['data', 'label'],
            batch_size=config['training_data']['input_transform']['batch_size'],
            use_gpu_transform=config['training_data']['input_transform']['use_gpu_transform'],
            scale=config['training_data']['input_transform']['scale'],
            crop=config['training_data']['input_transform']['crop_size'],
            mean_per_channel=config['training_data']['input_transform']['mean_per_channel'],
            std_per_channel=config['training_data']['input_transform']['std_per_channel'],
            mirror=True,
            is_test=False,
        )

    # stop bp
    model.StopGradient('data', 'data')
    model.StopGradient('label', 'label')

    return data, label


def add_model(model, config, data):
    if config['model_arch']['topk'] is not None:
        softmax = add_lenet_conv_topk(model, data, config['model_arch']['topk'])
    else:
        softmax = add_lenet(model, data)
    return softmax


def add_softmax_loss(model, softmax, label):
    # compute softmax cross-entropy loss
    xent = model.LabelCrossEntropy([softmax, label], ['xent'])
    softmax_loss = model.AveragedLoss(xent, 'softmax_loss')
    return softmax_loss


def add_optimizer(model, config):
    # add L2 norm for every weights
    optimizer.add_weight_decay(model, config['solver']['weight_decay'])
    optimizer.build_multi_precision_sgd(
        model,
        base_learning_rate = config['solver']['base_learning_rate'],
        momentum = config['solver']['momentum'],
        nesterov = config['solver']['nesterov'],
        policy = config['solver']['lr_policy'],
        gamma = config['solver']['gamma'],
        stepsize = config['solver']['stepsize'],
        # power = config['solver']['power'],
        # max_iter = config['solver']['max_iter'],
    )


def add_training_operators(model, config, loss):
    """
    compute model loss and add backword propagation with optimization method
    """
    model.AddGradientOperators([loss])
    add_optimizer(model, config)


def add_accuracy(model, softmax, label):
    """ compute model classification accuracy """
    accuracy = brew.accuracy(
        model,
        [softmax, label],
        "accuracy"
    )
    accuracy_5 = model.net.Accuracy(
        [softmax, label],
        "accuracy_5",
        top_k=5,
    )
    return (accuracy, accuracy_5)



if __name__ == '__main__':
    pass






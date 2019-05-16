from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import sys

from caffe2.python import workspace, model_helper, core, brew
from caffe2.proto import caffe2_pb2



def add_lenet(model, data):
    '''
    This part is the standard LeNet model: from data to the softmax prediction.
    "data -> conv -> max_pool -> conv -> max_pool -> fc -> relu ->fc"
    '''
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=3, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    # Image size: 8 x 8  -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    relu = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, relu, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')

    return softmax


def add_lenet_conv_topk(model, data, k=10):
    '''
    Based on original LeNet structure, according to meProp, we add topk
    gradient selecting layer for conv layers
    '''
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=3, dim_out=20, kernel=5)
    conv1_hook = model.net.TopKGradHook(conv1, 'conv1_hook', k=k)

    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1_hook, 'pool1', kernel=2, stride=2)

    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    conv2_hook = model.net.TopKGradHook(conv2, 'conv2_hook', k=k)

    # Image size: 8 x 8  -> 4 x 4
    pool2 = brew.max_pool(model, conv2_hook, 'pool2', kernel=2, stride=2)

    # 50 * 4 * 4 stands for dim_out from previous layer multiplied by the image size
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    relu = brew.relu(model, fc3, fc3)
    pred = brew.fc(model, relu, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')

    return softmax


def add_lenet_fc_topk(model, data, k=10):
    '''
    Based on original LeNet structure, according to meProp, we add topk
    gradient selecting layer for fc layers
    '''
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=3, dim_out=20, kernel=5)
    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    # Image size: 8 x 8  -> 4 x 4
    pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)

    # DEBUG PRINT
    model.net.Print(pool2, [])

    # set topkgradhook for the hidden fc layers
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    fc3_hook = model.net.TopKGradHook(fc3, 'fc3_hook', k=k)
    relu = brew.relu(model, fc3_hook, fc3_hook)

    # don't set topkgradhook for the output fc layer
    pred = brew.fc(model, relu, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')

    return softmax


def add_lenet_full_topk(model, data, k=10):
    '''
    Based on original LeNet structure, according to meProp, we add topk
    gradient selecting layer for all conv & fc layers
    '''
    # Image size: 28 x 28 -> 24 x 24
    conv1 = brew.conv(model, data, 'conv1', dim_in=3, dim_out=20, kernel=5)
    conv1_hook = model.net.TopKGradHook(conv1, 'conv1_hook', k=k)

    # Image size: 24 x 24 -> 12 x 12
    pool1 = brew.max_pool(model, conv1_hook, 'pool1', kernel=2, stride=2)

    # Image size: 12 x 12 -> 8 x 8
    conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50, kernel=5)
    conv2_hook = model.net.TopKGradHook(conv2, 'conv2_hook', k=k)

    # Image size: 8 x 8  -> 4 x 4
    pool2 = brew.max_pool(model, conv2_hook, 'pool2', kernel=2, stride=2)

    # set topkgradhook for the hidden fc layers
    fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
    fc3_hook = model.net.TopKGradHook(fc3, 'fc3_hook', k=k)
    relu = brew.relu(model, fc3_hook, fc3_hook)

    # don't set topkgradhook for the output fc layer
    pred = brew.fc(model, relu, 'pred', 500, 10)
    softmax = brew.softmax(model, pred, 'softmax')

    return softmax

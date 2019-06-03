from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import sys

from caffe2.python import workspace, model_helper, core, brew
from caffe2.proto import caffe2_pb2



''' ResNet20 for cifar10 '''
###############################################################################
#                            original ResNet20                                #
###############################################################################
def input_block(
    model,
    inputs,
    dim_in,
    dim_out,
    kernel,
    pad,
    stride,
    no_bias=False,
    is_test=False,
):
    '''
    add input conv module (separated out due to the name of predict.pbtxt)
    '''
    # convolution layer
    conv = brew.conv(
        model,
        inputs,
        'first_conv',
        dim_in=dim_in,
        dim_out=dim_out,
        kernel=kernel,
        pad=pad,
        stride=stride,
        no_bias=no_bias
    )

    # spaial batchnormalization layer
    bn = brew.spatial_bn(
        model,
        conv,
        'first_conv_bn',
        # conv,    # in-place
        dim_in = dim_out,
        epsilon = 1e-05,
        is_test = is_test,
    )

    # ReLU layer
    relu = brew.relu(
        model,
        bn,
        bn # in-place
    )

    return relu


def meta_conv(
    model,
    inputs,
    dim_in,
    dim_out,
    kernel,
    pad,
    stride,
    no_bias=False,
    is_test=False,
    has_relu=False,
    module_seq=None, # group seq
    sub_seq=None, # block seq
    branch_seq=None,
    conv_seq=None
):
    '''
    add basic conv module of resnet 50
    '''
    # set projection branch
    if branch_seq == "2": # branch of 2 normal part
        conv_name = "conv{}".format(conv_seq)
    elif branch_seq == "1" : # branch of 1 projection part
        conv_name = "proj"

    # convolution layer
    conv = brew.conv(
        model,
        inputs,
        'group{}_block{}_{}'.format(module_seq, sub_seq, conv_name),
        dim_in=dim_in,
        dim_out=dim_out,
        kernel=kernel,
        stride=stride,
        pad=pad,
        no_bias=no_bias,
    )

    # spaial batchnormalization layer
    bn = brew.spatial_bn(
        model,
        conv,
        'group{}_block{}_{}_bn'.format(module_seq, sub_seq, conv_name),
        # conv,    # in-place
        dim_in = dim_out,
        epsilon = 1e-05,
        is_test = is_test,
    )

    # ReLU layer
    if has_relu:
        relu = brew.relu(model, bn, bn)
        return relu
    else:
        return bn


def res_block_1(
    model,
    inputs,
    dim_in,
    dim_out,
    no_bias=False,
    is_test=False,
    module_seq=None,
    sub_seq=None,
):
    # branch of 1 (projection)
    branch1_conv = meta_conv(
        model,
        inputs,
        dim_in,
        dim_out,
        kernel=1,
        pad=0,
        stride=2,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=False,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='1',
        conv_seq='',
    )

    # branch of 2 (normal)
    branch2_conv1 = meta_conv(
        model,
        inputs,
        dim_in,
        dim_out,
        kernel=3,
        pad=1,
        stride=2,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=True,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='0',
    )

    branch2_conv2 = meta_conv(
        model,
        branch2_conv1,
        dim_out,
        dim_out,
        kernel=3,
        pad=1,
        stride=1,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=False,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='1',
    )

    # sum
    branch_sum = brew.sum(
        model,
        [branch2_conv2, branch1_conv],
        # branch2_conv2 # in-place
        ["group{}_conv{}_sum".format(module_seq, sub_seq)]
    )
    branch_relu = brew.relu(
        model,
        branch_sum,
        branch_sum
    )
    return branch_relu


def res_block_2(
    model,
    inputs,
    dim_in,
    dim_out,
    no_bias=False,
    is_test=False,
    module_seq=None,
    sub_seq=None,
):
    # input & output channel check
    assert(dim_in == dim_out)

    # branch of 2 (normal)
    branch2_conv1 = meta_conv(
        model,
        inputs,
        dim_in,
        dim_out,
        kernel=3,
        pad=1,
        stride=1,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=True,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='0',
    )

    branch2_conv2 = meta_conv(
        model,
        branch2_conv1,
        dim_in,
        dim_out,
        kernel=3,
        pad=1,
        stride=1,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=False,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='1',
    )

    # sum
    branch_sum = brew.sum(
        model,
        [branch2_conv2, inputs],
        # branch2_conv2 # in-place
        ["group{}_conv{}_sum".format(module_seq, sub_seq)]
    )
    branch_relu = brew.relu(
        model,
        branch_sum,
        branch_sum
    )
    return branch_relu


def res_module(
    model,
    inputs,
    dim_in,
    dim_out,
    no_bias=False,
    is_test=False,
    module_seq=None,
    block_list=None,
):
    # check block list
    assert(block_list is not None)
    length = len(block_list)
    blocks = [None for i in range(length + 1)]
    blocks[0] = inputs

    # iter blocks
    for i in range(len(block_list)):
        if block_list[i] == 1:
            blocks[i + 1] = res_block_1(
                model,
                blocks[i],
                dim_in=dim_in,
                dim_out=dim_out,
                no_bias=no_bias,
                is_test=is_test,
                module_seq=module_seq,
                sub_seq=str(i),
            )
        else:
            blocks[i + 1] = res_block_2(
                model,
                blocks[i],
                dim_in=dim_out,
                dim_out=dim_out,
                no_bias=no_bias,
                is_test=is_test,
                module_seq=module_seq,
                sub_seq=str(i)
            )

    return blocks[-1]


def add_resnet20(model, data, **kwargs):
    '''
    This part is the standard ResNet20 model: from data to the softmax prediction.
    '''
    # resolving arguments
    config = kwargs.get('config', {})
    is_test = kwargs.get('is_test', False)

    # construct models
    # Image size: 32 x 32 -> 32 x 32
    first_conv_relu = input_block(model, data, 3, 16, 3, 1, 1, is_test=is_test)
    # Image size: 32 x 32 -> 32 x 32
    group_0 = res_module(model, first_conv_relu, 16, 16,
                         module_seq="0", block_list=[2, 2, 2])
    # Image size: 32 x 32 -> 16 x 16
    group_1 = res_module(model, group_0, 16, 32,
                         module_seq="1", block_list=[1, 2, 2])
    # Image size: 16 x 16 -> 8 x 8
    group_2 = res_module(model, group_1, 32, 64,
                         module_seq="2", block_list=[1, 2, 2])
    # predicting
    pool = brew.average_pool(
        model,
        group_2,
        'pool',
        kernel=config['model_arch']['last_conv_size'],
        stride=1,
    )
    pred = brew.fc(
        model,
        pool,
        'pred',
        dim_in=64,
        dim_out=config['model_arch']['num_classes'],
    )
    softmax = brew.softmax(model, pred, 'softmax')

    return softmax


###############################################################################
#                            topk conv ResNet20                               #
###############################################################################
def input_block_conv_topk(
    model,
    inputs,
    dim_in,
    dim_out,
    kernel,
    pad,
    stride,
    topk=5,
    no_bias=False,
    is_test=False,
):
    '''
    add input conv module (separated out due to the name of predict.pbtxt)
    '''
    # convolution layer
    conv = brew.conv(
        model,
        inputs,
        'first_conv',
        dim_in=dim_in,
        dim_out=dim_out,
        kernel=kernel,
        pad=pad,
        stride=stride,
        no_bias=no_bias
    )

    conv_hook = model.net.TopKGradHook(
        conv,
        'first_conv_hook',
        k=topk,
    )

    # spaial batchnormalization layer
    bn = brew.spatial_bn(
        model,
        conv_hook,
        'first_conv_bn',
        # conv,    # in-place
        dim_in = dim_out,
        epsilon = 1e-05,
        is_test = is_test,
    )

    # ReLU layer
    relu = brew.relu(
        model,
        bn,
        bn # in-place
    )

    return relu


def meta_conv_conv_topk(
    model,
    inputs,
    dim_in,
    dim_out,
    kernel,
    pad,
    stride,
    topk=5,
    no_bias=False,
    is_test=False,
    has_relu=False,
    module_seq=None, # group seq
    sub_seq=None, # block seq
    branch_seq=None,
    conv_seq=None
):
    '''
    add basic conv module of resnet 50
    '''
    # set projection branch
    if branch_seq == "2": # branch of 2 normal part
        conv_name = "conv{}".format(conv_seq)
    elif branch_seq == "1" : # branch of 1 projection part
        conv_name = "proj"

    # convolution layer
    conv = brew.conv(
        model,
        inputs,
        'group{}_block{}_{}'.format(module_seq, sub_seq, conv_name),
        dim_in=dim_in,
        dim_out=dim_out,
        kernel=kernel,
        stride=stride,
        pad=pad,
        no_bias=no_bias,
    )

    conv_hook = model.net.TopKGradHook(
        conv,
        'group{}_block{}_{}_hook'.format(module_seq, sub_seq, conv_name),
        k=topk,
    )

    # spaial batchnormalization layer
    bn = brew.spatial_bn(
        model,
        conv_hook,
        'group{}_block{}_{}_bn'.format(module_seq, sub_seq, conv_name),
        # conv,    # in-place
        dim_in = dim_out,
        epsilon = 1e-05,
        is_test = is_test,
    )

    # ReLU layer
    if has_relu:
        relu = brew.relu(model, bn, bn)
        return relu
    else:
        return bn


def res_block_1_conv_topk(
    model,
    inputs,
    dim_in,
    dim_out,
    topk=5,
    no_bias=False,
    is_test=False,
    module_seq=None,
    sub_seq=None,
):
    # branch of 1 (projection)
    branch1_conv = meta_conv_conv_topk(
        model,
        inputs,
        dim_in,
        dim_out,
        kernel=1,
        pad=0,
        stride=2,
        topk=topk,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=False,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='1',
        conv_seq='',
    )

    # branch of 2 (normal)
    branch2_conv1 = meta_conv_conv_topk(
        model,
        inputs,
        dim_in,
        dim_out,
        kernel=3,
        pad=1,
        stride=2,
        topk=topk,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=True,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='0',
    )

    branch2_conv2 = meta_conv_conv_topk(
        model,
        branch2_conv1,
        dim_out,
        dim_out,
        kernel=3,
        pad=1,
        stride=1,
        topk=topk,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=False,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='1',
    )

    # sum
    branch_sum = brew.sum(
        model,
        [branch2_conv2, branch1_conv],
        # branch2_conv2 # in-place
        ["group{}_conv{}_sum".format(module_seq, sub_seq)]
    )
    branch_relu = brew.relu(
        model,
        branch_sum,
        branch_sum
    )
    return branch_relu


def res_block_2_conv_topk(
    model,
    inputs,
    dim_in,
    dim_out,
    topk=5,
    no_bias=False,
    is_test=False,
    module_seq=None,
    sub_seq=None,
):
    # input & output channel check
    assert(dim_in == dim_out)

    # branch of 2 (normal)
    branch2_conv1 = meta_conv_conv_topk(
        model,
        inputs,
        dim_in,
        dim_out,
        kernel=3,
        pad=1,
        stride=1,
        topk=topk,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=True,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='0',
    )

    branch2_conv2 = meta_conv_conv_topk(
        model,
        branch2_conv1,
        dim_in,
        dim_out,
        kernel=3,
        pad=1,
        stride=1,
        topk=topk,
        no_bias=no_bias,
        is_test=is_test,
        has_relu=False,
        module_seq=module_seq,
        sub_seq=sub_seq,
        branch_seq='2',
        conv_seq='1',
    )

    # sum
    branch_sum = brew.sum(
        model,
        [branch2_conv2, inputs],
        # branch2_conv2 # in-place
        ["group{}_conv{}_sum".format(module_seq, sub_seq)]
    )
    branch_relu = brew.relu(
        model,
        branch_sum,
        branch_sum
    )
    return branch_relu


def res_module_conv_topk(
    model,
    inputs,
    dim_in,
    dim_out,
    topk=5,
    no_bias=False,
    is_test=False,
    module_seq=None,
    block_list=None,
):
    # check block list
    assert(block_list is not None)
    length = len(block_list)
    blocks = [None for i in range(length + 1)]
    blocks[0] = inputs

    # iter blocks
    for i in range(len(block_list)):
        if block_list[i] == 1:
            blocks[i + 1] = res_block_1_conv_topk(
                model,
                blocks[i],
                dim_in=dim_in,
                dim_out=dim_out,
                topk=topk,
                no_bias=no_bias,
                is_test=is_test,
                module_seq=module_seq,
                sub_seq=str(i),
            )
        else:
            blocks[i + 1] = res_block_2_conv_topk(
                model,
                blocks[i],
                dim_in=dim_out,
                dim_out=dim_out,
                topk=topk,
                no_bias=no_bias,
                is_test=is_test,
                module_seq=module_seq,
                sub_seq=str(i)
            )

    return blocks[-1]


def add_resnet20_conv_topk(model, data, **kwargs):
    '''
    Based on original ResNet20 network  structure, according to meProp, we add
    topk gradient selecting layer for conv layers
    '''
    # resolving arguments
    config = kwargs.get('config', {})
    k = kwargs.get('k', 10)
    is_test = kwargs.get('is_test', False)

    # construct models
    # Image size: 32 x 32 -> 32 x 32
    first_conv_relu = input_block_conv_topk(model, data, 3, 16, 3, 1, 1,
                                            topk=k,
                                            is_test=is_test)
    # Image size: 32 x 32 -> 32 x 32
    group_0 = res_module_conv_topk(model, first_conv_relu, 16, 16,
                                   topk=k, module_seq="0", block_list=[2, 2, 2])
    # Image size: 32 x 32 -> 16 x 16
    group_1 = res_module_conv_topk(model, group_0, 16, 32,
                                   topk=k, module_seq="1", block_list=[1, 2, 2])
    # Image size: 16 x 16 -> 8 x 8
    group_2 = res_module_conv_topk(model, group_1, 32, 64,
                                   topk=k, module_seq="2", block_list=[1, 2, 2])
    # predicting
    pool = brew.average_pool(
        model,
        group_2,
        'pool',
        kernel=config['model_arch']['last_conv_size'],
        stride=1,
    )
    pred = brew.fc(
        model,
        pool,
        'pred',
        dim_in=64,
        dim_out=config['model_arch']['num_classes'],
    )
    softmax = brew.softmax(model, pred, 'softmax')

    return softmax


def add_resnet20_conv_topk_list(model, data, **kwargs):
    '''
    Based on original ResNet20 network  structure, according to meProp, we add
    topk gradient selecting layer for conv layers
    '''
    # resolving arguments
    config = kwargs.get('config', {})
    ks = kwargs.get('k', [10, 10, 10, 10]) # hard-coded topk for resnet20 groups
    assert(type(ks) is list)
    is_test = kwargs.get('is_test', False)

    # construct models
    # Image size: 32 x 32 -> 32 x 32
    first_conv_relu = input_block_conv_topk(model, data, 3, 16, 3, 1, 1,
                                            topk=ks[0],
                                            is_test=is_test)
    # Image size: 32 x 32 -> 32 x 32
    group_0 = res_module_conv_topk(model, first_conv_relu, 16, 16,
                                   topk=ks[1],
                                   module_seq="0", block_list=[2, 2, 2])
    # Image size: 32 x 32 -> 16 x 16
    group_1 = res_module_conv_topk(model, group_0, 16, 32,
                                   topk=ks[2],
                                   module_seq="1", block_list=[1, 2, 2])
    # Image size: 16 x 16 -> 8 x 8
    group_2 = res_module_conv_topk(model, group_1, 32, 64,
                                   topk=ks[3],
                                   module_seq="2", block_list=[1, 2, 2])
    # predicting
    pool = brew.average_pool(
        model,
        group_2,
        'pool',
        kernel=config['model_arch']['last_conv_size'],
        stride=1,
    )
    pred = brew.fc(
        model,
        pool,
        'pred',
        dim_in=64,
        dim_out=config['model_arch']['num_classes'],
    )
    softmax = brew.softmax(model, pred, 'softmax')

    return softmax

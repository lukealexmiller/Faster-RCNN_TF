#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import time, os, sys
import tensorflow as tf
from tensorflow.python.framework import graph_util

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Convert Checkpoint to ProtoBuf for Fast R-CNN network')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--model_path', dest='model_path',
                        help='checkpoint file directory',
                        default=None, type=str)
    parser.add_argument('--model_file', dest='model_file',
                        help='model to convert',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=False, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='voc_2007_test', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def freeze_graph(network_name,model_path,model_file):
    # We retrieve our checkpoint fullpath
    #checkpoint = tf.train.get_checkpoint_state(model_path,model_file)
    #input_checkpoint = checkpoint.model_checkpoint_path
    input_checkpoint = model_path+model_file
    print(input_checkpoint)
    # We precise the file fullname of our freezed graph
    #absolute_model_path = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = model_path + os.path.splitext(model_file)[0] + ".pb"
    print(output_graph)
    # Before exporting our graph, we need to specify our output nodes
    # This is how TF decides what part of the Graph he has to keep and what part it can dump
    # NOTE: this variable is plural, because you can have multiple output nodes
    output_node_names = "cls_prob"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    network = get_network(network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    #init = tf.global_variables_initializer()
    # We import the meta graph and retrieve a Saver
    
    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    saver = tf.train.Saver()
    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        #sess.run(init)
        saver.restore(sess, input_checkpoint)
        print ('Loading model weights from {:s}').format(input_checkpoint)
        #[n.name for n in tf.get_default_graph().as_graph_def().node]
        print(output_node_names.split(","))
        # We use a built-in TF helper to export variables to constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the useful nodes
        ) 

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    while not os.path.exists(args.model_path) and args.wait:
        print('Waiting for {} to exist...'.format(args.model_path))
        time.sleep(10)

    freeze_graph(args.network_name,args.model_path,args.model_file)
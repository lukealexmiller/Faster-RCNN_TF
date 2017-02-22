#!/usr/bin/env python

"""Freeze a Fast R-CNN network and export to ProtoBuf file for use in deployment."""

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
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

def export_graph(network_name,model_path,model_file):
    # Define full path and filename of input checkpoint file
    input_checkpoint = model_path + model_file
    # Define full path and filename of output protobuf file
    output_graph = model_path + os.path.splitext(model_file)[0] + ".pb"

    # List of name strings for desired result nodes of graph
    output_node_names = ['cls_prob', 'bbox_pred/bbox_pred']

    network = get_network(network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    saver = tf.train.Saver()
    
    # We start a session and restore the graph weights
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver.restore(sess, input_checkpoint)
        print ('Loading model weights from {:s}').format(input_checkpoint)
        
        # We retrieve the protobuf graph definition
        input_graph_def = sess.graph.as_graph_def()

        # Display names of nodes in graph (useful for debugging)
        for v in sess.graph.get_operations():
            print(v.name)

        # Export variables as constants
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes
            output_node_names # The output node names are used to define the useful nodes
        )
        print('Saving output node names: {:s}').format(output_node_names)

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.FastGFile(output_graph, 'wb') as f:
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
        print('Waiting for {} to exist...'.format(args.model_path+args.model_file))
        time.sleep(10)

    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print device_name

    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    if args.device == 'gpu':
        cfg.USE_GPU_NMS = True
        cfg.GPU_ID = args.device_id
    else:
        cfg.USE_GPU_NMS = False

    export_graph(args.network_name,args.model_path,args.model_file)
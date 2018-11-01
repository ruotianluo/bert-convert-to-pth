from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from collections import OrderedDict
import re
import os
import torch
import numpy as np

from scipy.misc import imread, imresize
import modeling_pth as modeling

import argparse
parser = argparse.ArgumentParser(description='Convert tf-faster-rcnn model to pytorch-faster-rcnn model')
parser.add_argument('--folder',
                    help='the path of tensorflow_model',
                    default='uncased_L-12_H-768_A-12', type=str)

args = parser.parse_args()

def tf_to_pth(tensorflow_model):
    reader = pywrap_tensorflow.NewCheckpointReader(tensorflow_model)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_dict = {k:reader.get_tensor(k) for k in var_to_shape_map.keys()}

    # config = modeling.BertConfig.from_json_file(os.path.join(args.folder, 'bert_config.json'))
    # model = modeling.BertModel(config=config)

    # x = model.state_dict()

    for k in list(var_dict.keys()):
        if var_dict[k].ndim == 2 and 'kernel' in k:
            var_dict[k] = var_dict[k].transpose((1, 0)).copy(order='C')

    dummy_replace = OrderedDict([
                    ('layer_', 'layers.'),\
                    ('weights', 'weight'),\
                    ('/LayerNorm', '_layer_norm'),\
                    ('gamma', 'weight'),\
                    ('beta', 'bias'),\
                    ('kernel', 'weight'),\
                    ('biases', 'bias'),\
                    ('/dense', ''), \
                    ('embeddings_layer_norm', 'embedding_postprocessor.layer_norm'), \
                    ('word_embeddings', 'word_embeddings.weight'), \
                    ('embeddings/token_type_embeddings', 'embedding_postprocessor.token_type_embeddings.weight'),\
                    ('embeddings/position', 'embedding_postprocessor.position'),\
                    ('/', '.')])

    for a, b in dummy_replace.items():
        for k in list(var_dict.keys()):
            if a in k:
                var_dict[k.replace(a,b)] = var_dict[k]
                del var_dict[k]

    # print(set(var_dict.keys()) - set(x.keys()))
    # print(set(x.keys()) - set(var_dict.keys()))

    # assert len(set(x.keys()) - set(var_dict.keys())) == 0
    # for k in set(var_dict.keys()) - set(x.keys()):
    #     del var_dict[k]

    # for k in list(var_dict.keys()):
    #     if x[k].shape != var_dict[k].shape:
    #         var_dict[k] = var_dict[k].transpose(1,0)
    #         print(k, x[k].shape, var_dict[k].shape)
    #     assert x[k].shape == var_dict[k].shape, k

    for k in list(var_dict.keys()):
        var_dict[k] = torch.from_numpy(var_dict[k])
    
    return var_dict

converted_state_dict = tf_to_pth(os.path.join(args.folder, 'bert_model.ckpt'))
torch.save(converted_state_dict, os.path.join(args.folder, 'bert_model.pth'))

"""
Make sure the tensorflow and pytorch gives the same output (Haven't passed yet.)
"""

def test_tf(input_ids, input_mask, token_type_ids):
    import modeling as modeling_tf

    input_ids = tf.constant(value=input_ids, dtype=tf.int32, shape=input_ids.shape, name=None)
    input_mask = tf.constant(value=input_mask, dtype=tf.int32, shape=input_mask.shape, name=None)
    token_type_ids = tf.constant(value=token_type_ids, dtype=tf.int32, shape=token_type_ids.shape, name=None)

    config = modeling_tf.BertConfig.from_json_file(os.path.join(args.folder, 'bert_config.json'))

    model = modeling_tf.BertModel(
          config=config,
          is_training=False,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=token_type_ids)

    outputs = {
          "embedding_output": model.get_embedding_output(),
          "sequence_output": model.get_sequence_output(),
          "pooled_output": model.get_pooled_output(),
          "all_encoder_layers": model.get_all_encoder_layers(),
    }

    restorer = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        restorer.restore(sess, os.path.join(args.folder, 'bert_model.ckpt'))
        output_result = sess.run(outputs)

    return output_result

def test_pth(input_ids, input_mask, token_type_ids):
    config = modeling.BertConfig.from_json_file(os.path.join(args.folder, 'bert_config.json'))
    model = modeling.BertModel(config=config)
    model.load_state_dict(converted_state_dict, strict=False)
    model.eval()

    input_ids, input_mask, token_type_ids = \
        torch.from_numpy(input_ids).long(), \
        torch.from_numpy(input_mask).long(), \
        torch.from_numpy(token_type_ids).long()

    model(input_ids, input_mask, token_type_ids)
    outputs = {
          "embedding_output": model.get_embedding_output(),
          "sequence_output": model.get_sequence_output(),
          "pooled_output": model.get_pooled_output(),
          "all_encoder_layers": model.get_all_encoder_layers(),
    }
    return outputs

def assert_almost_equal(tf_tensor, th_tensor):
    t = th_tensor
    if t.dim() == 4:
        t = t.permute(0,2,3,1)
    t = t.data.numpy()
    f = tf_tensor

    #for i in range(0, t.shape[-1]):
    #    print("tf", i,  t[:,i])
    #    print("caffe", i,  c[:,i])

    if t.shape != f.shape:
        print("t.shape", t.shape)
        print("f.shape", f.shape)

    d = np.linalg.norm(t - f)
    print("d", d)
    # assert d < 500


def ids_tensor(shape, vocab_size):
    """Creates a random int32 tensor of the shape within the vocab size."""
    import random

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(random.randint(0, vocab_size - 1))

    return np.reshape(np.array(values).astype(np.int32), shape)

batch_size, seq_length, vocab_size = 13, 7, 99
sample_input = [ids_tensor([batch_size, seq_length],
                                           vocab_size),
                ids_tensor([batch_size, seq_length], vocab_size=2),
                ids_tensor([batch_size, seq_length], 1)]


print('forward tf')
tf_out = test_tf(*sample_input)
print('forward pth')
pth_out = test_pth(*sample_input)


import pdb;pdb.set_trace()
for k in tf_out.keys():
    if type(tf_out[k]) is list:
        for _ in range(len(tf_out[k])):
            print(k,_)
            assert_almost_equal(tf_out[k][_], pth_out[k][_])
    else:
        print(k)
        assert_almost_equal(tf_out[k], pth_out[k])


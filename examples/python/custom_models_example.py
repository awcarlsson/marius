import sys
sys.path.insert(0, 'build/') # need to add the build directory to the system path so python can find the bindings
import _pymarius as m
import torch
import os
import math
import torch.nn.functional as F

# initialize our custom model and train
def marius():

    # initialize Marius parameters to those specified in configuration file or default
    # values by calling parseConfig(config_path)
    config_path = "examples/training/configs/fb15k_cpu.ini"
    config = m.parseConfig(config_path)

    # Here we define a custom model to use during the training and evaluation process
    # A Model consists of both 1) Encoder and 2) Decoder
    # A Decoder consists of 1) Comparator, 2) Relation Operator, and 3) Loss Function

    # initialize the encoder to Empty
    encoder = m.EmptyEncoder()

    # initialize the loss function to built-in SoftMax
    loss_function = m.SoftMax()

    # initialize custom Relation Operator and Comparator implemented in Python
    rel_op = translation()
    comp = PyDotCompare()
    # comp = L2()

    # initialize the decoder with our Loss Function and custom Relation Operator and Comparator
    decoder = m.LinkPredictionDecoder(comp, rel_op, loss_function)

    # initialize the custom model with our Encoder and Decoder
    custom_model = transE(encoder, decoder)

    # model instantiation and train for one epoch
    train_set, eval_set = m.initializeDatasets(config)
    trainer = m.SynchronousTrainer(train_set, custom_model)
    evaluator = m.SynchronousEvaluator(eval_set, custom_model)
    trainer.train(1)
    evaluator.evaluate(True)

## CUSTOM MODEL IMPLEMENTATION EXAMPLES ##

# RELATION OPERATOR EXAMPLES

# Translation Relation Operator
class translation(m.RelationOperator):
    def __init__(self):
        m.RelationOperator.__init__(self)
    def __call__(self, node_embs, rel_embs):
        return node_embs + rel_embs

# COMPARATOR EXAMPLES

# L2 Comparator
class L2(m.Comparator):
    def __init__(self):
        m.Comparator.__init__(self)
    def __call__(self, src, dst, negs):

        num_chunks = negs.size(0)
        num_pos = src.size(0)
        num_per_chunk = math.ceil(num_pos/num_chunks)

        adjusted_src = src
        adjusted_dst = dst

        if (num_per_chunk != num_pos // num_chunks):
            new_size = num_per_chunk * num_chunks
            adjusted_src = F.pad(input=adjusted_src, pad=(0, 0, 0, new_size - num_pos), mode='constant', value=0)
            adjusted_dst = F.pad(input=adjusted_dst, pad=(0, 0, 0, new_size - num_pos), mode='constant', value=0)

        pos_scores = torch.cdist(src, dst, p=2)
        adjusted_src = adjusted_src.view(num_chunks, num_per_chunk, src.size(1))
        neg_scores = torch.cdist(adjusted_src, negs, p=2).flatten(0, 1)

        return pos_scores, neg_scores

# DotCompare Comparator
# Note: already built in as pymarius.DotCompare, Python example
# Can use to compare speed of operations when using C++ vs. Python bindings
class PyDotCompare(m.Comparator):
    def __init__(self):
        m.Comparator.__init__(self)
    def __call__(self, src, dst, negs):

        num_chunks = negs.size(0)
        num_pos = src.size(0)
        num_per_chunk = math.ceil(num_pos/num_chunks)

        adjusted_src = src
        adjusted_dst = dst

        if (num_per_chunk != num_pos // num_chunks):
            new_size = num_per_chunk * num_chunks
            adjusted_src = F.pad(input=adjusted_src, pad=(0, 0, 0, new_size - num_pos), mode='constant', value=0)
            adjusted_dst = F.pad(input=adjusted_dst, pad=(0, 0, 0, new_size - num_pos), mode='constant', value=0)

        pos_scores = (adjusted_src * adjusted_dst).sum(-1)
        adjusted_src = adjusted_src.view(num_chunks, num_per_chunk, src.size(1))
        neg_scores = adjusted_src.bmm(negs.transpose(-1, -2)).flatten(0, 1)

        return pos_scores, neg_scores

# CUSTOM MODEL EXAMPLES

# transE Model
class transE(m.Model):
    def __init__(self, encoder, decoder):
        m.Model.__init__(self, encoder, decoder)

if __name__ == "__main__":
    marius()
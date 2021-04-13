import sys
sys.path.insert(0, 'build/') # need to add the build directory to the system path so python can find the bindings
import pymarius as m
import torch
import os

def marius():

    config_path = "examples/training/configs/fb15k_cpu.ini"
    config = m.parseConfig(config_path)

    # allocating here is fine because objects will last through end of function stack
    encoder = m.EmptyEncoder() # initialize the encoder
    # decoder = m.LinkPredictionDecoder() # initialize the decoder
    # loss function affected when inititalizing with empty constructor and manually changing var?

    # prebuilt
    #rel_op = m.HadamardOperator() # initialize the relation operator
    #comp = m.CosineCompare() # comparator
    loss_function = m.SoftMax()

    # custom
    rel_op = translation()
    comp = L2()

    # decoder.relation_operator = rel_op
    # decoder.comparator = comp
    # decoder.loss_function = m.SoftMax()

    decoder = m.LinkPredictionDecoder(comp, rel_op, loss_function) # initialize the decoder
    custom_model = transE(encoder, decoder)


    #config.model.encoder_model = m.EncoderModelType.Custom # don't need this

    # model instantiation and train
    train_set, eval_set = m.initializeDatasets(config)
    #model = m.initializeModel(config.model.encoder_model, config.model.decoder_model) # don't need for custom
    trainer = m.SynchronousTrainer(train_set, custom_model)
    #trainer = m.SynchronousTrainer(train_set, model)
    evaluator = m.SynchronousEvaluator(eval_set, custom_model)
    #evaluator = m.SynchronousEvaluator(eval_set, model)
    trainer.train(1)
    evaluator.evaluate(True)

class translation(m.RelationOperator):
    def __init__(self):
        m.RelationOperator.__init__(self)
    def __call__(self, node_embs, rel_embs):
        return node_embs + rel_embs

class L2(m.Comparator):
    def __init__(self):
        m.Comparator.__init__(self)
    def __call__(self, src, dst, negs):

        # This is not L2 it's actually using a Dot product. Not sure how to do batched distance computation.
        pos_scores = (src * dst).sum(-1)
        neg_scores = src.view(negs.size(0), -1, src.size(1)).bmm(negs.transpose(-1, -2)).flatten(0, 1)
        return pos_scores, neg_scores

class transE(m.Model):
    def __init__(self, encoder, decoder):
        m.Model.__init__(self, encoder, decoder)

if __name__ == "__main__":
    marius()
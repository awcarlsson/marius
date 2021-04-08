import sys
sys.path.insert(0, 'build/') # need to add the build directory to the system path so python can find the bindings
import pymarius as m
import torch

def marius():
    config_path = "examples/training/configs/fb15k_gpu.ini"
    config = m.parseConfig(config_path)

    # allocating here is fine because objects will last through end of function stack
    encoder = m.EmptyEncoder() # initialize the encoder
    # decoder = m.LinkPredictionDecoder() # initialize the decoder
    # loss function affected when inititalizing with empty constructor and manually changing var?

    #rel_op = m.HadamardOperator() # initialize the relation operator
    #comp = m.CosineCompare() # comparator

    rel_op = translation()
    comp = L2()
    loss_function = m.SoftMax()

    # decoder.relation_operator = rel_op
    # decoder.comparator = comp
    # decoder.loss_function = m.SoftMax()

    decoder = m.LinkPredictionDecoder(comp, rel_op, loss_function) # initialize the decoder
    print(decoder.comparator)
    print(decoder.relation_operator)
    print(decoder.loss_function)

    custom_model = transE(encoder, decoder)
    print(custom_model.decoder.relation_operator)
    print(custom_model.decoder.comparator)
    print(custom_model.decoder.loss_function)

    config.model.encoder_model = m.EncoderModelType.Custom # don't need this

    # model instantiation and train
    train_set, eval_set = m.initializeDatasets(config)
    #model = m.initializeModel(config.model.encoder_model, config.model.decoder_model) # don't need for custom
    trainer = m.SynchronousTrainer(train_set, custom_model)
    evaluator = m.SynchronousEvaluator(eval_set, custom_model)
    trainer.train(1)
    evaluator.evaluate(True)

class translation(m.RelationOperator):
    def __init__(self):
        m.RelationOperator.__init__(self)
    def __call__(self, node_embs, rel_embs):
        print("relation called")
        return node_embs + rel_embs

class L2(m.Comparator):
    def __init__(self):
        m.Comparator.__init__(self)
    def __call__(self, src, dst, negs):
        print("comp called")
        return torch.cdist(src, dst, p=2)

class transE(m.Model):
    def __init__(self, encoder, decoder):
        m.Model.__init__(self, encoder, decoder)

if __name__ == "__main__":
    marius()
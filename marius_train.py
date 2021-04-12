import sys
sys.path.insert(0, 'build/') # need to add the build directory to the system path so python can find the bindings
import pymarius as m

def marius():
    config_path = "examples/training/configs/fb15k_gpu.ini"
    config = m.parseConfig(config_path)
    train_set, eval_set = m.initializeDatasets(config)
    model = m.initializeModel(config.model.encoder_model, config.model.decoder_model)
    trainer = m.SynchronousTrainer(train_set, model)
    evaluator = m.SynchronousEvaluator(eval_set, model)
    trainer.train(1)
    evaluator.evaluate(True)
if __name__ == "__main__":
    marius()
import sys
sys.path.insert(0, 'build/') # need to add the build directory to the system path so python can find the bindings
import _pymarius as m

def marius():

    # initialize Marius parameters to those specified in configuration file or default
    # values by calling parseConfig(config_path)
    config_path = "examples/training/configs/fb15k_cpu.ini"
    config = m.parseConfig(config_path)

    # initialize the training and evaluation sets
    train_set, eval_set = m.initializeDatasets(config)

    # initialize the model with the encoder and decoder type specified in the config
    # Note: if encoder and decoder are already initialized, you can initialize the model
    # with a constructor call, i.e. model = m.Model(encoder, decoder). See other Python
    # example file (custom_models_example.py) for how to define custom models
    model = m.initializeModel(config.model.encoder_model, config.model.decoder_model)

    # initialize the trainer and evaluator using the data and model
    trainer = m.SynchronousTrainer(train_set, model)
    evaluator = m.SynchronousEvaluator(eval_set, model)

    # train and evaluate
    for epoch in range(config.training.num_epochs):
        trainer.train(1)
        evaluator.evaluate(True)
    
if __name__ == "__main__":
    marius()

import unittest
import subprocess
import shutil
from pathlib import Path
import pytest
import os
import sys
import math
import torch.nn.functional as F
import torch
#from marius.tools import preprocess
sys.path.insert(0, 'build/')
import _pymarius as m


class TestCustomModels(unittest.TestCase):

    @classmethod
    def tearDown(self):
        if Path("output_dir").exists():
            shutil.rmtree(Path("output_dir"))
        if Path("training_data").exists():
            shutil.rmtree(Path("training_data"))

    @pytest.mark.skipif(os.environ.get("MARIUS_ONLY_PYTHON", None) == "TRUE", reason="Requires building the bindings")
    def test_relation_op(self):
        #preprocess.fb15k(output_dir="output_dir/")
        config_path = "examples/training/configs/fb15k_cpu.ini"
        config = m.parseConfig(config_path)

        encoder = m.EmptyEncoder()
        comp = m.DotCompare()
        rel_op = translation()
        loss_function = m.SoftMax()
        decoder = m.LinkPredictionDecoder(comp, rel_op, loss_function)

        model = m.Model(encoder, decoder)

        train_set, eval_set = m.initializeDatasets(config)
        trainer = m.SynchronousTrainer(train_set, model)
        evaluator = m.SynchronousEvaluator(eval_set, model)
        trainer.train(1)
        evaluator.evaluate(True)
    
    def test_comparator(self):
        #preprocess.fb15k(output_dir="output_dir/")
        config_path = "examples/training/configs/fb15k_cpu.ini"
        config = m.parseConfig(config_path)

        encoder = m.EmptyEncoder()
        comp = PyDotCompare()
        rel_op = m.HadamardOperator()
        loss_function = m.SoftMax()
        decoder = m.LinkPredictionDecoder(comp, rel_op, loss_function)

        model = m.Model(encoder, decoder)

        train_set, eval_set = m.initializeDatasets(config)
        trainer = m.SynchronousTrainer(train_set, model)
        evaluator = m.SynchronousEvaluator(eval_set, model)
        trainer.train(1)
        evaluator.evaluate(True)

class translation(m.RelationOperator):
    def __init__(self):
        m.RelationOperator.__init__(self)
    def __call__(self, node_embs, rel_embs):
        return node_embs + rel_embs

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

class transE(m.Model):
    def __init__(self, encoder, decoder):
        m.Model.__init__(self, encoder, decoder)
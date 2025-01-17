#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <decoder.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

// Trampoline classes
class PyRelationOperator : RelationOperator {
  public:
    using RelationOperator::RelationOperator;
    Embeddings operator()(const Embeddings &embs, const Relations &rels) override {
        PYBIND11_OVERRIDE_PURE_NAME(Embeddings, RelationOperator, "__call__", operator(), embs, rels); }
};

class PyComparator : Comparator {
  public:
    using Comparator::Comparator;
    using ReturnTensorTuple = tuple<torch::Tensor, torch::Tensor>;
    tuple<torch::Tensor, torch::Tensor> operator()(const Embeddings &src, const Embeddings &dst, const Embeddings &negs) override { 
      PYBIND11_OVERRIDE_PURE_NAME(ReturnTensorTuple, Comparator, "__call__", operator(), src, dst, negs); }
};

class PyLossFunction : LossFunction {
  public:
    using LossFunction::LossFunction;
    torch::Tensor operator()(const torch::Tensor &pos_scores, const torch::Tensor &neg_scores) override { 
      PYBIND11_OVERRIDE_PURE_NAME(torch::Tensor, LossFunction, "__call__", operator(), pos_scores, neg_scores); }
};

class PyDecoder : Decoder {
  public:
    using Decoder::Decoder;
    void forward(Batch *batch, bool train) override { 
      PYBIND11_OVERRIDE_PURE(void, Decoder, forward, batch, train); }
};

void init_decoder(py::module &m) {

  py::class_<RelationOperator, PyRelationOperator>(m, "RelationOperator")
    .def(py::init<>())
    .def("__call__", &RelationOperator::operator(), py::arg("embs"), py::arg("rels"));
  py::class_<HadamardOperator, RelationOperator>(m, "HadamardOperator")
    .def(py::init<>());
  py::class_<ComplexHadamardOperator, RelationOperator>(m, "ComplexHadamardOperator")
    .def(py::init<>());
  py::class_<TranslationOperator, RelationOperator>(m, "TranslationOperator")
    .def(py::init<>());
  py::class_<NoOp, RelationOperator>(m, "NoOp")
    .def(py::init<>());
  
  py::class_<Comparator, PyComparator>(m, "Comparator")
    .def(py::init<>())
    .def("__call__", &Comparator::operator(), py::arg("src"), py::arg("dst"), py::arg("negs"));
  py::class_<CosineCompare, Comparator>(m, "CosineCompare")
    .def(py::init<>());
  py::class_<DotCompare, Comparator>(m, "DotCompare")
    .def(py::init<>());

  py::class_<LossFunction, PyLossFunction>(m, "LossFunction")
    .def(py::init<>())
    .def("__call__", &LossFunction::operator(), py::arg("pos_scores"), py::arg("neg_scores"));
  py::class_<SoftMax, LossFunction>(m, "SoftMax")
    .def(py::init<>());
  py::class_<RankingLoss, LossFunction>(m, "RankingLoss")
    .def(py::init<float>(), py::arg("margin"));

  py::class_<Decoder, PyDecoder>(m, "Decoder")
    .def(py::init<>())
    .def("forward", &Decoder::forward, py::arg("batch"), py::arg("train"));
  py::class_<LinkPredictionDecoder, Decoder>(m, "LinkPredictionDecoder")
    .def_readwrite("comparator", &LinkPredictionDecoder::comparator_)
    .def_readwrite("relation_operator", &LinkPredictionDecoder::relation_operator_)
    .def_readwrite("loss_function", &LinkPredictionDecoder::loss_function_)
    .def(py::init<>())
    .def(py::init<Comparator *, RelationOperator *, LossFunction *>());
  py::class_<DistMult, LinkPredictionDecoder>(m, "DistMult")
    .def(py::init<>());
  py::class_<TransE, LinkPredictionDecoder>(m, "TransE")
    .def(py::init<>());
  py::class_<ComplEx, LinkPredictionDecoder>(m, "ComplEx")
    .def(py::init<>());
  py::class_<NodeClassificationDecoder, Decoder>(m, "NodeClassificationDecoder")
    .def(py::init<>());
  py::class_<RelationClassificationDecoder, Decoder>(m, "RelationClassificationDecoder");
}

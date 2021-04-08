//
// Created by Jason Mohoney on 3/23/21.
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <model.h>

namespace py = pybind11;

// class PyModel : Model {
//   public:
//     using Model::Model;
//     void train(Batch *batch) override { PYBIND11_OVERRIDE(void, Model, train, batch); }
//     void evaluate(Batch *batch) override { PYBIND11_OVERRIDE(void, Model, evaluate, batch); }
// };

void init_model(py::module &m) {
    py::class_<Model>(m, "Model")
        .def_readwrite("encoder", &Model::encoder_)
        .def_readwrite("decoder", &Model::decoder_)
        .def(py::init<Encoder *, Decoder *>(), py::arg("encoder"), py::arg("decoder"), py::return_value_policy::reference)
        .def("train", &Model::train, py::arg("batch"))
        .def("evaluate", &Model::evaluate, py::arg("batch"));

    m.def("initializeModel", &initializeModel);
}
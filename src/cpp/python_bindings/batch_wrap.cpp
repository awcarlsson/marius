#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <batch.h>

namespace py = pybind11;

void init_batch(py::module &m) {
    
	py::enum_<BatchStatus>(m, "BatchStatus")
        .value("Waiting", BatchStatus::Waiting)
        .value("AccumulatedIndices", BatchStatus::AccumulatedIndices)
        .value("LoadedEmbeddings", BatchStatus::LoadedEmbeddings)
        .value("TransferredToDevice", BatchStatus::TransferredToDevice)
        .value("PreparedForCompute", BatchStatus::PreparedForCompute)
        .value("ComputedGradients", BatchStatus::ComputedGradients)
        .value("AccumulatedGradients", BatchStatus::AccumulatedGradients)
        .value("TransferredToHost", BatchStatus::TransferredToHost)
        .value("Done", BatchStatus::Done);

	py::class_<Batch>(m, "Batch")
        .def_readwrite("batch_id_", &Batch::batch_id_)
        .def_readwrite("start_idx_", &Batch::start_idx_)
        .def_readwrite("batch_size_", &Batch::batch_size_)
        .def_readwrite("train_", &Batch::train_)
        .def_readwrite("unique_node_indices_", &Batch::unique_node_indices_)
        .def_readwrite("unique_node_embeddings_", &Batch::unique_node_embeddings_)
        .def_readwrite("unique_node_gradients_", &Batch::unique_node_gradients_)
        .def_readwrite("unique_node_gradients2_", &Batch::unique_node_gradients2_)
        .def_readwrite("unique_node_embeddings_state_", &Batch::unique_node_embeddings_state_)
        .def_readwrite("unique_relation_indices_", &Batch::unique_relation_indices_)
        .def_readwrite("unique_relation_indices_", &Batch::unique_relation_indices_)
        .def_readwrite("unique_relation_embeddings_", &Batch::unique_relation_embeddings_)
        .def_readwrite("unique_relation_gradients_", &Batch::unique_relation_gradients_)
        .def_readwrite("unique_relation_gradients2_", &Batch::unique_relation_gradients2_)
        .def_readwrite("unique_relation_embeddings_state_", &Batch::unique_relation_embeddings_state_)
        .def_readwrite("src_pos_indices_mapping_", &Batch::src_pos_indices_mapping_)
        .def_readwrite("dst_pos_indices_mapping_", &Batch::dst_pos_indices_mapping_)
        .def_readwrite("rel_indices_mapping_", &Batch::rel_indices_mapping_)
        .def_readwrite("src_neg_indices_mapping_", &Batch::src_neg_indices_mapping_)
        .def_readwrite("dst_neg_indices_mapping_", &Batch::dst_neg_indices_mapping_)
        .def_readwrite("src_pos_indices_", &Batch::src_pos_indices_)
        .def_readwrite("dst_pos_indices_", &Batch::dst_pos_indices_)
        .def_readwrite("rel_indices_", &Batch::rel_indices_)
        .def_readwrite("src_neg_indices_", &Batch::src_neg_indices_)
        .def_readwrite("dst_neg_indices_", &Batch::dst_neg_indices_)
        .def_readwrite("src_pos_embeddings_", &Batch::src_pos_embeddings_)
        .def_readwrite("dst_pos_embeddings_", &Batch::dst_pos_embeddings_)
        .def_readwrite("src_relation_emebeddings_", &Batch::src_relation_emebeddings_)
        .def_readwrite("dst_relation_emebeddings_", &Batch::dst_relation_emebeddings_)
        .def_readwrite("src_global_neg_embeddings_", &Batch::src_global_neg_embeddings_)
        .def_readwrite("dst_global_neg_embeddings_", &Batch::dst_global_neg_embeddings_)
        .def_readwrite("src_all_neg_embeddings_", &Batch::src_all_neg_embeddings_)
        .def_readwrite("dst_all_neg_embeddings_", &Batch::dst_all_neg_embeddings_)
        .def_readwrite("load_timestamp_", &Batch::load_timestamp_)
        .def_readwrite("compute_timestamp_", &Batch::compute_timestamp_)
	    .def_readwrite("device_transfer_", &Batch::device_transfer_)
        .def_readwrite("host_transfer_", &Batch::host_transfer_)
        .def_readwrite("device_id_", &Batch::device_id_)
        .def_readwrite("timer_", &Batch::timer_)
        .def_readwrite("ranks_", &Batch::ranks_)
        .def_readwrite("auc_", &Batch::auc_)
        .def_readwrite("src_neg_filter_eval_", &Batch::src_neg_filter_eval_)
        .def_readwrite("dst_neg_filter_eval_", &Batch::dst_neg_filter_eval_)
        .def_readwrite("status_", &Batch::status_)
        .def(py::init<bool>(), py::arg("train"))
        .def("localSample", &Batch::localSample)
        .def("accumulateUniqueIndices", &Batch::accumulateUniqueIndices)
        .def("embeddingsToDevice", &Batch::embeddingsToDevice, py::arg("device_id"))
        .def("prepareBatch", &Batch::prepareBatch)
        .def("accumulateGradients", &Batch::accumulateGradients)
        .def("embeddingsToHost", &Batch::embeddingsToHost)
        .def("clear", &Batch::clear);
}
#include <torch/script.h>

#include "metatensor/torch/labels.hpp"
#include "metatensor/torch/block.hpp"
#include "metatensor/torch/tensor.hpp"
#include "metatensor/torch/misc.hpp"
#include "metatensor/torch/atomistic.hpp"

#include "internal/utils.hpp"

using namespace metatensor_torch;


static TorchLabelsEntry labels_entry(const TorchLabels& self, int64_t index) {
    return torch::make_intrusive<LabelsEntryHolder>(self, index);
}

// this function can not be implemented as a member of LabelsHolder, since it
// needs to receive a `TorchLabels` to give it to the `LabelsEntryHolder`
// constructor.
static torch::IValue labels_getitem(const TorchLabels& self, torch::IValue index) {
    if (index.isInt()) {
        return labels_entry(self, index.toInt());
    } else if (index.isString()) {
        return self->column(index.toStringRef());
    } else {
        C10_THROW_ERROR(TypeError,
            "Labels can only be indexed by int or str, got '" + index.type()->str() + "' instead"
        );
    }
}

template <typename T>
bool custom_class_is(torch::IValue ivalue) {
    assert(ivalue.isCustomClass());

    // this is inspired by the code inside `torch::IValue.toCustomClass<T>()`
    auto* expected_type = torch::getCustomClassType<torch::intrusive_ptr<T>>().get();
    return ivalue.type().get() == expected_type;
}

static void save_ivalue(const std::string& path, torch::IValue data) {
    if (data.isCustomClass()) {
        if (custom_class_is<TensorMapHolder>(data)) {
            auto tensor = data.toCustomClass<TensorMapHolder>();
            return metatensor_torch::save(path, tensor);
        } else if (custom_class_is<LabelsHolder>(data)) {
            auto labels = data.toCustomClass<LabelsHolder>();
            return metatensor_torch::save(path, labels);
        }
    }

    C10_THROW_ERROR(TypeError,
        "data` must be either 'Labels' or 'TensorMap' in `save`, not "
        + data.type()->str()
    );
}

static torch::Tensor save_ivalue_buffer(torch::IValue data) {
    if (data.isCustomClass()) {
        if (custom_class_is<TensorMapHolder>(data)) {
            auto tensor = data.toCustomClass<TensorMapHolder>();
            return metatensor_torch::save_buffer(tensor);
        } else if (custom_class_is<LabelsHolder>(data)) {
            auto labels = data.toCustomClass<LabelsHolder>();
            return metatensor_torch::save_buffer(labels);
        }
    }

    C10_THROW_ERROR(TypeError,
        "data` must be either 'Labels' or 'TensorMap' in `save_buffer`, not "
        + data.type()->str()
    );
}

TORCH_LIBRARY(metatensor, m) {
    // There is no way to access the docstrings from Python, so we don't bother
    // setting them to something useful here.
    //
    // Whenever this file is changed, please also reproduce the changes in
    // python/metatensor-torch/metatensor/torch/documentation.py, and include the
    // docstring over there
    const std::string DOCSTRING;


    m.class_<LabelsEntryHolder>("LabelsEntry")
        .def("__str__", &LabelsEntryHolder::repr)
        .def("__repr__", &LabelsEntryHolder::repr)
        .def("__len__", &LabelsEntryHolder::size)
        .def("__getitem__", &LabelsEntryHolder::getitem, DOCSTRING,
            {torch::arg("index")}
        )
        .def("__eq__", [](const TorchLabelsEntry& self, const TorchLabelsEntry& other){ return *self == *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def("__ne__", [](const TorchLabelsEntry& self, const TorchLabelsEntry& other){ return *self != *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def_property("names", &LabelsEntryHolder::names)
        .def_property("values", &LabelsEntryHolder::values)
        .def_property("device", &LabelsEntryHolder::device)
        .def("print", &LabelsEntryHolder::print)
        ;

    m.class_<LabelsHolder>("Labels")
        .def(
            torch::init<torch::IValue, torch::Tensor>(), DOCSTRING,
            {torch::arg("names"), torch::arg("values")}
        )
        .def("__str__", &LabelsHolder::str)
        .def("__repr__", &LabelsHolder::repr)
        .def("__len__", &LabelsHolder::count)
        .def("__contains__", [](const TorchLabels& self, torch::IValue entry) {
                return self->position(entry).has_value();
            }, DOCSTRING,
            {torch::arg("entry")}
        )
        .def("__eq__", [](const TorchLabels& self, const TorchLabels& other){ return *self == *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def("__ne__", [](const TorchLabels& self, const TorchLabels& other){ return *self != *other; },
            DOCSTRING,
            {torch::arg("other")}
        )
        .def("__getitem__", labels_getitem, DOCSTRING, {torch::arg("index")})
        .def_static("single", &LabelsHolder::single)
        .def_static("empty", &LabelsHolder::empty)
        .def_static("range", &LabelsHolder::range)
        .def("save", &LabelsHolder::save, DOCSTRING, {torch::arg("file")})
        .def("save_buffer", &LabelsHolder::save_buffer)
        .def_static("load", &LabelsHolder::load)
        .def_static("load_buffer", &LabelsHolder::load_buffer)
        .def("entry", labels_entry, DOCSTRING, {torch::arg("index")})
        .def("column", &LabelsHolder::column, DOCSTRING, {torch::arg("dimension")})
        .def("view", [](const TorchLabels& self, torch::IValue names) {
            auto names_vector = metatensor_torch::details::normalize_names(std::move(names), "names");
            return LabelsHolder::view(self, std::move(names_vector));
        }, DOCSTRING, {torch::arg("names")})
        .def_property("names", &LabelsHolder::names)
        .def_property("values", &LabelsHolder::values)
        .def("append", &LabelsHolder::append, DOCSTRING, {torch::arg("name"), torch::arg("values")})
        .def("insert", &LabelsHolder::insert, DOCSTRING, {torch::arg("index"), torch::arg("name"), torch::arg("values")})
        .def("permute", &LabelsHolder::permute, DOCSTRING, {torch::arg("dimensions_indexes")})
        .def("remove", &LabelsHolder::remove, DOCSTRING, {torch::arg("name")})
        .def("rename", &LabelsHolder::rename, DOCSTRING, {torch::arg("old"), torch::arg("new")})
        .def("to",
            static_cast<TorchLabels (LabelsHolder::*)(torch::IValue) const>(&LabelsHolder::to),
            DOCSTRING, {torch::arg("device")}
        )
        .def_property("device", &LabelsHolder::device)
        .def("position", &LabelsHolder::position, DOCSTRING,
            {torch::arg("entry")}
        )
        .def("print", &LabelsHolder::print, DOCSTRING,
            {torch::arg("max_entries"), torch::arg("indent") = 0}
        )
        .def("is_view", &LabelsHolder::is_view)
        .def("to_owned", &LabelsHolder::to_owned)
        .def("union", &LabelsHolder::set_union, DOCSTRING, {torch::arg("other")})
        .def("union_and_mapping", &LabelsHolder::union_and_mapping, DOCSTRING, {torch::arg("other")})
        .def("intersection", &LabelsHolder::set_intersection, DOCSTRING, {torch::arg("other")})
        .def("intersection_and_mapping", &LabelsHolder::intersection_and_mapping, DOCSTRING, {torch::arg("other")})
        .def_pickle(
            // __getstate__
            [](const TorchLabels& self){ return self->save_buffer(); },
            // __setstate__
            [](torch::Tensor buffer){ return metatensor_torch::load_labels_buffer(buffer); }
        );


    m.class_<TensorBlockHolder>("TensorBlock")
        .def(
            torch::init<torch::Tensor, TorchLabels, std::vector<TorchLabels>, TorchLabels>(), DOCSTRING,
            {torch::arg("values"), torch::arg("samples"), torch::arg("components"), torch::arg("properties")}
        )
        .def("__repr__", &TensorBlockHolder::repr)
        .def("__str__", &TensorBlockHolder::repr)
        .def("copy", &TensorBlockHolder::copy)
        .def_property("values", &TensorBlockHolder::values)
        .def_property("samples", &TensorBlockHolder::samples)
        .def_property("components", &TensorBlockHolder::components)
        .def_property("properties", &TensorBlockHolder::properties)
        .def("add_gradient", &TensorBlockHolder::add_gradient, DOCSTRING,
            {torch::arg("parameter"), torch::arg("gradient")}
        )
        .def("gradients_list", &TensorBlockHolder::gradients_list)
        .def("has_gradient", &TensorBlockHolder::has_gradient, DOCSTRING,
            {torch::arg("parameter")}
        )
        .def("gradient", &TensorBlockHolder::gradient, DOCSTRING,
            {torch::arg("parameter")}
        )
        .def("gradients", &TensorBlockHolder::gradients)
        .def_property("device", &TensorBlockHolder::device)
        .def_property("dtype", &TensorBlockHolder::scalar_type)
        .def("to", &TensorBlockHolder::to_positional, DOCSTRING, {
            torch::arg("_0") = torch::IValue(),
            torch::arg("_1") = torch::IValue(),
            torch::arg("dtype") = torch::nullopt,
            torch::arg("device") = torch::nullopt,
            torch::arg("arrays") = torch::nullopt
        })
        ;

    m.class_<TensorMapHolder>("TensorMap")
        .def(
            torch::init<TorchLabels, std::vector<TorchTensorBlock>>(), DOCSTRING,
            {torch::arg("keys"), torch::arg("blocks")}
        )
        .def("__len__", [](const TorchTensorMap& self){ return self->keys()->count(); })
        .def("__repr__", [](const TorchTensorMap& self){ return self->print(-1); })
        .def("__str__", [](const TorchTensorMap& self){ return self->print(4); })
        .def("__getitem__", &TensorMapHolder::block_torch, DOCSTRING,
            {torch::arg("selection")}
        )
        .def("copy", &TensorMapHolder::copy)
        .def("save", &TensorMapHolder::save, DOCSTRING, {torch::arg("file")})
        .def("save_buffer", &TensorMapHolder::save_buffer)
        .def_static("load", &TensorMapHolder::load)
        .def_static("load_buffer", &TensorMapHolder::load_buffer)
        .def("items", &TensorMapHolder::items)
        .def_property("keys", &TensorMapHolder::keys)
        .def("blocks_matching", &TensorMapHolder::blocks_matching, DOCSTRING,
            {torch::arg("selection")}
        )
        .def("block_by_id", &TensorMapHolder::block_by_id, DOCSTRING,
            {torch::arg("index")}
        )
        .def("blocks_by_id", &TensorMapHolder::blocks_by_id, DOCSTRING,
            {torch::arg("indices")}
        )
        .def("block", &TensorMapHolder::block_torch, DOCSTRING,
            {torch::arg("selection") = torch::IValue()}
        )
        .def("blocks", &TensorMapHolder::blocks_torch, DOCSTRING,
            {torch::arg("selection") = torch::IValue()}
        )
        .def("keys_to_samples", &TensorMapHolder::keys_to_samples, DOCSTRING,
            {torch::arg("keys_to_move"), torch::arg("sort_samples") = true}
        )
        .def("keys_to_properties", &TensorMapHolder::keys_to_properties, DOCSTRING,
            {torch::arg("keys_to_move"), torch::arg("sort_samples") = true}
        )
        .def("components_to_properties", &TensorMapHolder::components_to_properties, DOCSTRING,
            {torch::arg("dimensions")}
        )
        .def_property("sample_names", &TensorMapHolder::sample_names)
        .def_property("component_names", &TensorMapHolder::component_names)
        .def_property("property_names", &TensorMapHolder::property_names)
        .def_property("device", &TensorMapHolder::device)
        .def_property("dtype", &TensorMapHolder::scalar_type)
        .def("to", &TensorMapHolder::to_positional, DOCSTRING, {
            torch::arg("_0") = torch::IValue(),
            torch::arg("_1") = torch::IValue(),
            torch::arg("dtype") = torch::nullopt,
            torch::arg("device") = torch::nullopt,
            torch::arg("arrays") = torch::nullopt
        })
        .def("print", &TensorMapHolder::print, DOCSTRING,
            {torch::arg("max_keys")}
        )
        .def_pickle(
            // __getstate__
            [](const TorchTensorMap& self){ return self->save_buffer(); },
            // __setstate__
            [](torch::Tensor buffer){ return metatensor_torch::load_buffer(buffer); }
        );


    // standalone functions
    m.def("version() -> str", metatensor_torch::version);
    m.def("dtype_name(ScalarType dtype) -> str", scalar_type_name);

    m.def(
        "load(str path) -> __torch__.torch.classes.metatensor.TensorMap",
        metatensor_torch::load
    );
    m.def(
        "load_buffer(Tensor buffer) -> __torch__.torch.classes.metatensor.TensorMap",
        metatensor_torch::load_buffer
    );

    m.def(
        "load_labels(str path) -> __torch__.torch.classes.metatensor.Labels",
        metatensor_torch::load_labels
    );
    m.def(
        "load_labels_buffer(Tensor buffer) -> __torch__.torch.classes.metatensor.Labels",
        metatensor_torch::load_labels_buffer
    );

    m.def("save(str path, Any data) -> ()", save_ivalue);
    m.def("save_buffer(Any data) -> Tensor", save_ivalue_buffer);

    // ====================================================================== //
    //               code specific to atomistic simulations                   //
    // ====================================================================== //
    m.class_<NeighborsListOptionsHolder>("NeighborsListOptions")
        .def(
            torch::init<double, bool, std::string>(), DOCSTRING,
            {torch::arg("cutoff"), torch::arg("full_list"), torch::arg("requestor") = ""}
        )
        .def_property("cutoff", &NeighborsListOptionsHolder::cutoff)
        .def_property("length_unit", &NeighborsListOptionsHolder::length_unit, &NeighborsListOptionsHolder::set_length_unit)
        .def("engine_cutoff", &NeighborsListOptionsHolder::engine_cutoff,
            DOCSTRING, {torch::arg("engine_length_unit")}
        )
        .def_property("full_list", &NeighborsListOptionsHolder::full_list)
        .def("requestors", &NeighborsListOptionsHolder::requestors)
        .def("add_requestor", &NeighborsListOptionsHolder::add_requestor, DOCSTRING,
            {torch::arg("requestor")}
        )
        .def("__repr__", &NeighborsListOptionsHolder::repr)
        .def("__str__", &NeighborsListOptionsHolder::str)
        .def("__eq__", static_cast<bool (*)(const NeighborsListOptions&, const NeighborsListOptions&)>(operator==))
        .def("__ne__", static_cast<bool (*)(const NeighborsListOptions&, const NeighborsListOptions&)>(operator!=))
        .def_pickle(
            [](const NeighborsListOptions& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> NeighborsListOptions {
                return NeighborsListOptionsHolder::from_json(data);
            }
        );


    m.class_<SystemHolder>("System")
        .def(
            torch::init<torch::Tensor, torch::Tensor, torch::Tensor>(), DOCSTRING,
            {torch::arg("types"), torch::arg("positions"), torch::arg("cell")}
        )
        .def_property("types", &SystemHolder::types, &SystemHolder::set_types)
        .def_property("positions", &SystemHolder::positions, &SystemHolder::set_positions)
        .def_property("cell", &SystemHolder::cell, &SystemHolder::set_cell)
        .def("__len__", &SystemHolder::size)
        .def("__str__", &SystemHolder::str)
        .def("__repr__", &SystemHolder::str)
        .def_property("device", &SystemHolder::device)
        .def_property("dtype", &SystemHolder::scalar_type)
        .def("to", &SystemHolder::to_positional, DOCSTRING, {
            torch::arg("_0") = torch::IValue(),
            torch::arg("_1") = torch::IValue(),
            torch::arg("dtype") = torch::nullopt,
            torch::arg("device") = torch::nullopt
        })
        .def("add_neighbors_list", &SystemHolder::add_neighbors_list, DOCSTRING,
            {torch::arg("options"), torch::arg("neighbors")}
        )
        .def("get_neighbors_list", &SystemHolder::get_neighbors_list, DOCSTRING,
            {torch::arg("options")}
        )
        .def("known_neighbors_lists", &SystemHolder::known_neighbors_lists)
        .def("add_data", &SystemHolder::add_data, DOCSTRING,
            {torch::arg("name"), torch::arg("data")}
        )
        .def("get_data", &SystemHolder::get_data, DOCSTRING,
            {torch::arg("name")}
        )
        .def("known_data", &SystemHolder::known_data)
        ;


    m.class_<ModelMetadataHolder>("ModelMetadata")
        .def(
            torch::init<
                std::string,
                std::string,
                std::vector<std::string>,
                torch::Dict<std::string, std::vector<std::string>>
            >(),
            DOCSTRING, {
                torch::arg("name") = "",
                torch::arg("description") = "",
                torch::arg("authors") = std::vector<std::string>(),
                torch::arg("references") = torch::Dict<std::string, std::vector<std::string>>(),
            }
        )
        .def("__repr__", &ModelMetadataHolder::print)
        .def("__str__", &ModelMetadataHolder::print)
        .def_readwrite("name", &ModelMetadataHolder::name)
        .def_readwrite("description", &ModelMetadataHolder::description)
        .def_readwrite("authors", &ModelMetadataHolder::authors)
        .def_readwrite("references", &ModelMetadataHolder::references)
        .def_pickle(
            [](const ModelMetadata& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> ModelMetadata {
                return ModelMetadataHolder::from_json(data);
            }
        );


    m.class_<ModelOutputHolder>("ModelOutput")
        .def(
            torch::init<std::string, std::string, bool, std::vector<std::string>>(),
            DOCSTRING, {
                torch::arg("quantity") = "",
                torch::arg("unit") = "",
                torch::arg("per_atom") = false,
                torch::arg("explicit_gradients") = std::vector<std::string>()
            }
        )
        .def_property("quantity", &ModelOutputHolder::quantity, &ModelOutputHolder::set_quantity)
        .def_property("unit", &ModelOutputHolder::unit, &ModelOutputHolder::set_unit)
        .def_readwrite("per_atom", &ModelOutputHolder::per_atom)
        .def_readwrite("explicit_gradients", &ModelOutputHolder::explicit_gradients)
        .def_pickle(
            [](const ModelOutput& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> ModelOutput {
                return ModelOutputHolder::from_json(data);
            }
        );


    m.class_<ModelCapabilitiesHolder>("ModelCapabilities")
        .def(
            torch::init<
                torch::Dict<std::string, ModelOutput>,
                std::vector<int64_t>,
                double,
                std::string,
                std::vector<std::string>,
                std::string
            >(),
            DOCSTRING, {
                torch::arg("outputs") = torch::Dict<std::string, ModelOutput>(),
                torch::arg("atomic_types") = std::vector<int64_t>(),
                torch::arg("interaction_range") = -1.0,
                torch::arg("length_unit") = "",
                torch::arg("supported_devices") = std::vector<std::string>{},
                torch::arg("dtype") = "",
            }
        )
        .def_readwrite("outputs", &ModelCapabilitiesHolder::outputs)
        .def_readwrite("atomic_types", &ModelCapabilitiesHolder::atomic_types)
        .def_readwrite("interaction_range", &ModelCapabilitiesHolder::interaction_range)
        .def("engine_interaction_range", &ModelCapabilitiesHolder::engine_interaction_range)
        .def_property("length_unit", &ModelCapabilitiesHolder::length_unit, &ModelCapabilitiesHolder::set_length_unit)
        .def_readwrite("supported_devices", &ModelCapabilitiesHolder::supported_devices)
        .def_property("dtype", &ModelCapabilitiesHolder::dtype, &ModelCapabilitiesHolder::set_dtype)
        .def_pickle(
            [](const ModelCapabilities& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> ModelCapabilities {
                return ModelCapabilitiesHolder::from_json(data);
            }
        );


    m.class_<ModelEvaluationOptionsHolder>("ModelEvaluationOptions")
        .def(
            torch::init<
                std::string,
                torch::Dict<std::string, ModelOutput>,
                torch::optional<TorchLabels>
            >(),
            DOCSTRING, {
                torch::arg("length_unit") = "",
                torch::arg("outputs") = torch::Dict<std::string, ModelOutput>(),
                torch::arg("selected_atoms") = torch::nullopt,
            }
        )
        .def_property("length_unit", &ModelEvaluationOptionsHolder::length_unit, &ModelEvaluationOptionsHolder::set_length_unit)
        .def_readwrite("outputs", &ModelEvaluationOptionsHolder::outputs)
        .def_property("selected_atoms",
            &ModelEvaluationOptionsHolder::get_selected_atoms,
            &ModelEvaluationOptionsHolder::set_selected_atoms
        )
        .def_pickle(
            [](const ModelEvaluationOptions& self) -> std::string {
                return self->to_json();
            },
            [](const std::string& data) -> ModelEvaluationOptions {
                return ModelEvaluationOptionsHolder::from_json(data);
            }
        );


    m.def("unit_conversion_factor(str quantity, str from_unit, str to_unit) -> float", unit_conversion_factor);
    m.def("check_atomistic_model(str path) -> ()", check_atomistic_model);
    m.def(
        "register_autograd_neighbors("
            "__torch__.torch.classes.metatensor.System system, "
            "__torch__.torch.classes.metatensor.TensorBlock neighbors, "
            "bool check_consistency = False"
        ") -> ()",
        register_autograd_neighbors
    );
}

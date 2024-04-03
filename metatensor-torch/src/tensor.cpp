#include <metatensor.hpp>
#include <string>

#include "metatensor/torch/tensor.hpp"
#include "metatensor/torch/array.hpp"
#include "metatensor/torch/block.hpp"
#include "metatensor/torch/misc.hpp"

#include "internal/utils.hpp"

using namespace metatensor_torch;

template <typename T>
bool custom_class_is(torch::IValue ivalue) {
    assert(ivalue.isCustomClass());

    // this is inspired by the code inside `torch::IValue.toCustomClass<T>()`
    auto* expected_type = torch::getCustomClassType<torch::intrusive_ptr<T>>().get();
    return ivalue.type().get() == expected_type;
}

static metatensor::TensorBlock block_from_torch(const TorchTensorBlock& block) {
    auto components = std::vector<metatensor::Labels>();
    for (const auto& component: block->components()) {
        components.push_back(component->as_metatensor());
    }

    // use copy constructors of everything here, incrementing reference count
    // of the data and metadata
    auto result = metatensor::TensorBlock(
        std::make_unique<TorchDataArray>(block->values()),
        block->samples()->as_metatensor(),
        components,
        block->properties()->as_metatensor()
    );

    for (const auto& parameter: block->gradients_list()) {
        auto gradient = block_from_torch(TensorBlockHolder::gradient(block, parameter));
        result.add_gradient(parameter, std::move(gradient));
    }

    return result;
}

static std::vector<metatensor::TensorBlock> blocks_from_torch(const std::vector<TorchTensorBlock>& blocks) {
    auto results = std::vector<metatensor::TensorBlock>();
    results.reserve(blocks.size());
    for (const auto& block: blocks) {
        results.emplace_back(block_from_torch(block));
    }
    return results;
}


TensorMapHolder::TensorMapHolder(TorchLabels keys, const std::vector<TorchTensorBlock>& blocks):
    tensor_(keys->as_metatensor(), blocks_from_torch(blocks))
{
    if (blocks.empty()) {
        // nothing to check
        return;
    }

    auto device = keys->values().device();
    auto scalar_type = blocks[0]->values().scalar_type();

    for (const auto& block : blocks) {
        if (block->values().device() != device) {
            C10_THROW_ERROR(ValueError,
                "cannot create TensorMap: keys and blocks must be on the same device, "
                "got " + block->values().device().str() + " and " + device.str()
            );
        }
        if (block->values().scalar_type() != scalar_type) {
            C10_THROW_ERROR(ValueError,
                "cannot create TensorMap: all blocks must have the same dtype, "
                "got " + scalar_type_name(block->values().scalar_type()) +
                " and " + scalar_type_name(scalar_type)
            );
        }
    }
}

TorchTensorMap TensorMapHolder::copy() const {
    return torch::make_intrusive<TensorMapHolder>(TensorMapHolder(this->tensor_.clone()));
}

TorchLabels TensorMapHolder::keys() const {
    return torch::make_intrusive<LabelsHolder>(this->tensor_.keys());
}

std::vector<int64_t> TensorMapHolder::blocks_matching(const TorchLabels& selection) const {
    auto results = tensor_.blocks_matching(selection->as_metatensor());

    auto results_int64 = std::vector<int64_t>();
    results_int64.reserve(results.size());
    for (auto matching: results) {
        results_int64.push_back(static_cast<int64_t>(matching));
    }

    return results_int64;
}

TorchTensorBlock TensorMapHolder::block_by_id(TorchTensorMap self, int64_t index) {
    if (index >= self->keys()->count()) {
        // this needs to be an IndexError to enable iteration over a TensorMap
        C10_THROW_ERROR(IndexError,
            "block index out of bounds: we have " + std::to_string(self->keys()->count())
            + " blocks but the index is " + std::to_string(index)
        );
    }
    return torch::make_intrusive<TensorBlockHolder>(
        self->tensor_.block_by_id(index), self
    );
}


TorchTensorBlock TensorMapHolder::block(TorchTensorMap self, const std::map<std::string, int32_t>& selection_dict) {
    auto names = std::vector<std::string>();
    auto values = std::vector<int32_t>();
    for (const auto& it: selection_dict) {
        names.push_back(it.first);
        values.push_back(static_cast<int32_t>(it.second));
    }

    auto selection = metatensor::Labels(names, values.data(), 1);
    return TensorMapHolder::block(std::move(self), torch::make_intrusive<LabelsHolder>(std::move(selection)));
}

TorchTensorBlock TensorMapHolder::block(TorchTensorMap self, TorchLabels selection) {
    if (selection->count() != 1) {
        C10_THROW_ERROR(ValueError,
            "block selection must contain exactly one entry, got " + std::to_string(selection->count())
        );
    }

    return TensorMapHolder::block(std::move(self), torch::make_intrusive<LabelsEntryHolder>(selection, 0));
}

TorchTensorBlock TensorMapHolder::block(TorchTensorMap self, TorchLabelsEntry torch_selection) {
    auto cpu_values = torch_selection->values().to(torch::kCPU);
    auto selection = metatensor::Labels(
        torch_selection->names(), cpu_values.data_ptr<int32_t>(), 1
    );

    auto matching = self->tensor_.blocks_matching(selection);
    if (matching.empty()) {
        C10_THROW_ERROR(ValueError,
            "could not find blocks matching the selection " + torch_selection->print()
        );
    } else if (matching.size() != 1) {
        C10_THROW_ERROR(ValueError,
            "got more than one matching block for " + torch_selection->print() +
            ", use the `blocks` function to select more than one block"
        );
    }

    return TensorMapHolder::block_by_id(self, static_cast<int64_t>(matching[0]));
}

TorchTensorBlock TensorMapHolder::block_torch(TorchTensorMap self, torch::IValue index) {
    if (index.isInt()) {
        return TensorMapHolder::block_by_id(self, index.toInt());
    } else if (index.isNone()) {
        auto count = self->keys()->count();
        if (count == 0) {
            C10_THROW_ERROR(ValueError, "there is no block in this TensorMap");
        } else if (count > 1) {
            C10_THROW_ERROR(ValueError,
                "there is more than one block in this TensorMap, provide a selection"
            );
        }

        return TensorMapHolder::block_by_id(self, 0);
    } else if (index.isGenericDict()) {
        auto selection = std::map<std::string, int32_t>();
        for (const auto& it: index.toGenericDict()) {
            const auto& key = it.key();
            const auto& value = it.value();
            if (it.key().isString() && value.isInt()) {
                selection.emplace(key.toString()->string(), static_cast<int32_t>(value.toInt()));
            } else {
                C10_THROW_ERROR(TypeError,
                    "expected argument to be Dict[str, int], got Dict["
                    + key.type()->str() + ", " + value.type()->str() + "]"
                );
            }
        }
        return TensorMapHolder::block(self, selection);
    } else if (index.isCustomClass()) {
        if (custom_class_is<LabelsHolder>(index)) {
            auto labels = index.toCustomClass<LabelsHolder>();
            return TensorMapHolder::block(self, labels);
        } else if (custom_class_is<LabelsEntryHolder>(index)) {
            auto entry = index.toCustomClass<LabelsEntryHolder>();
            return TensorMapHolder::block(self, entry);
        } else {
            C10_THROW_ERROR(TypeError,
                "expected argument to be Labels or LabelsEntry, got"
                + index.type()->str()
            );
        }
    } else {
        C10_THROW_ERROR(TypeError,
            "expected argument to be int, Dict[str, int], Labels, or LabelsEntry, got "
            + index.type()->str()
        );
    }
}


std::vector<TorchTensorBlock> TensorMapHolder::blocks_by_id(TorchTensorMap self, const std::vector<int64_t>& indices) {
    auto result = std::vector<TorchTensorBlock>();
    for (auto i: indices) {
        result.push_back(TensorMapHolder::block_by_id(self, i));
    }
    return result;
}

std::vector<TorchTensorBlock> TensorMapHolder::blocks(TorchTensorMap self) {
    auto result = std::vector<TorchTensorBlock>();
    for (size_t i=0; i<self->tensor_.keys().count(); i++) {
        result.push_back(TensorMapHolder::block_by_id(self, static_cast<int64_t>(i)));
    }
    return result;
}


std::vector<TorchTensorBlock> TensorMapHolder::blocks(TorchTensorMap self, const std::map<std::string, int32_t>& selection_dict) {
    auto names = std::vector<std::string>();
    auto values = std::vector<int32_t>();
    for (const auto& it: selection_dict) {
        names.push_back(it.first);
        values.push_back(static_cast<int32_t>(it.second));
    }

    auto selection = metatensor::Labels(names, values.data(), 1);
    return TensorMapHolder::blocks(std::move(self), torch::make_intrusive<LabelsHolder>(std::move(selection)));
}


std::vector<TorchTensorBlock> TensorMapHolder::blocks(TorchTensorMap self, TorchLabels selection) {
    if (selection->count() != 1) {
        C10_THROW_ERROR(ValueError,
            "block selection must contain exactly one entry, got " + std::to_string(selection->count())
        );
    }

    return TensorMapHolder::blocks(std::move(self), torch::make_intrusive<LabelsEntryHolder>(selection, 0));
}


std::vector<TorchTensorBlock> TensorMapHolder::blocks(TorchTensorMap self, TorchLabelsEntry torch_selection) {
    auto cpu_values = torch_selection->values().to(torch::kCPU);
    auto selection = metatensor::Labels(
        torch_selection->names(), cpu_values.data_ptr<int32_t>(), 1
    );

    auto matching = std::vector<int64_t>();
    for (auto m: self->tensor_.blocks_matching(selection)) {
        matching.push_back(static_cast<int64_t>(m));
    }

    return TensorMapHolder::blocks_by_id(self, matching);
}


std::vector<TorchTensorBlock> TensorMapHolder::blocks_torch(TorchTensorMap self,torch::IValue index) {
    if (index.isNone()) {
        return TensorMapHolder::blocks(self);
    } else if (index.isInt()) {
        return {TensorMapHolder::block_by_id(self, index.toInt())};
    } else if (index.isIntList()) {
        return TensorMapHolder::blocks_by_id(self, index.toIntVector());
    } else if (index.isGenericDict()) {
        auto selection = std::map<std::string, int32_t>();
        for (const auto& it: index.toGenericDict()) {
            const auto& key = it.key();
            const auto& value = it.value();
            if (it.key().isString() && value.isInt()) {
                selection.emplace(key.toString()->string(), static_cast<int32_t>(value.toInt()));
            } else {
                C10_THROW_ERROR(ValueError,
                    "expected argument to be Dict[str, int], got Dict["
                    + key.type()->str() + ", " + value.type()->str() + "]"
                );
            }
        }
        return TensorMapHolder::blocks(self, selection);
    } else if (index.isCustomClass()) {
        if (custom_class_is<LabelsHolder>(index)) {
            auto labels = index.toCustomClass<LabelsHolder>();
            return TensorMapHolder::blocks(self, labels);
        } else if (custom_class_is<LabelsEntryHolder>(index)) {
            auto entry = index.toCustomClass<LabelsEntryHolder>();
            return TensorMapHolder::blocks(self, entry);
        } else {
            C10_THROW_ERROR(TypeError,
                "expected argument to be Labels or LabelsEntry, got"
                + index.type()->str()
            );
        }
    } else {
        C10_THROW_ERROR(ValueError,
            "expected argument to be None, int, List[int], Dict[str, int], Labels, or LabelsEntry, got "
            + index.type()->str()
        );
    }
}


/// Transform a torch::IValue containing either a single string, a list of
/// strings or a tuple of strings to something C++ can use
///
/// The `context` is used for the error message in case the torch::IValue is
/// none of the above.
static std::vector<std::string> extract_list_str(const torch::IValue& keys_to_move, const std::string& context) {
    if (keys_to_move.isString()) {
        return {keys_to_move.toString()->string()};
    } else if (keys_to_move.isList()) {
        auto results = std::vector<std::string>();
        for (const auto& element: keys_to_move.toListRef()) {
            if (element.isString()) {
                results.push_back(element.toString()->string());
            } else {
                C10_THROW_ERROR(TypeError, context + " must be a list of `str`");
            }
        }
        return results;
    } else if (keys_to_move.isTuple()) {
        auto results = std::vector<std::string>();
        for (const auto& element: keys_to_move.toTupleRef().elements()) {
            if (element.isString()) {
                results.push_back(element.toString()->string());
            } else {
                C10_THROW_ERROR(TypeError, context + " must be a tuple of `str`");
            }
        }
        return results;
    } else {
        throw std::runtime_error(
            "internal error: called extract_list_str, but IValue is not a list of str"
        );
    }
}

TorchTensorMap TensorMapHolder::keys_to_properties(torch::IValue keys_to_move, bool sort_samples) const {
    auto device = this->keys()->values().device();
    if (keys_to_move.isString() || keys_to_move.isList() || keys_to_move.isTuple()) {
        auto selection = extract_list_str(keys_to_move, "TensorMap::keys_to_properties first argument");
        auto tensor = tensor_.keys_to_properties(selection, sort_samples);
        auto result = torch::make_intrusive<TensorMapHolder>(TensorMapHolder(std::move(tensor)));
        return result->to(torch::nullopt, device);
    } else if (keys_to_move.isCustomClass()) {
        auto selection = keys_to_move.toCustomClass<LabelsHolder>();
        auto tensor = tensor_.keys_to_properties(selection->as_metatensor(), sort_samples);
        auto result = torch::make_intrusive<TensorMapHolder>(TensorMapHolder(std::move(tensor)));
        return result->to(torch::nullopt, device);
    } else {
        C10_THROW_ERROR(TypeError,
            "TensorMap::keys_to_properties first argument must be a `str`, list of `str` or `Labels`"
        );
    }
}

TorchTensorMap TensorMapHolder::keys_to_samples(torch::IValue keys_to_move, bool sort_samples) const {
    auto device = this->keys()->values().device();
    if (keys_to_move.isString() || keys_to_move.isList() || keys_to_move.isTuple()) {
        auto selection = extract_list_str(keys_to_move, "TensorMap::keys_to_samples first argument");
        auto tensor = tensor_.keys_to_samples(selection, sort_samples);
        auto result = torch::make_intrusive<TensorMapHolder>(TensorMapHolder(std::move(tensor)));
        return result->to(torch::nullopt, device);
    } else if (keys_to_move.isCustomClass()) {
        auto selection = keys_to_move.toCustomClass<LabelsHolder>();
        auto tensor = tensor_.keys_to_samples(selection->as_metatensor(), sort_samples);
        auto result = torch::make_intrusive<TensorMapHolder>(TensorMapHolder(std::move(tensor)));
        return result->to(torch::nullopt, device);
    } else {
        C10_THROW_ERROR(TypeError,
            "TensorMap::keys_to_samples first argument must be a `str`, list of `str` or `Labels`"
        );
    }
}

TorchTensorMap TensorMapHolder::components_to_properties(torch::IValue dimensions) const {
    auto device = this->keys()->values().device();
    auto selection = extract_list_str(dimensions, "TensorMap::components_to_properties argument");
    auto tensor = this->tensor_.components_to_properties(selection);
    auto result = torch::make_intrusive<TensorMapHolder>(TensorMapHolder(std::move(tensor)));
    return result->to(torch::nullopt, device);
}

static std::vector<std::string> labels_names(const metatensor::TensorBlock& block, size_t dimension) {
    auto result = std::vector<std::string>();

    auto labels = block.labels(dimension);
    for (const auto& name: labels.names()) {
        result.emplace_back(name);
    }

    return result;
}

std::vector<std::string> TensorMapHolder::sample_names() {
    if (tensor_.keys().count() == 0) {
        return {};
    }

    return labels_names(this->tensor_.block_by_id(0), 0);
}

std::vector<std::string> TensorMapHolder::component_names() {
    auto result = std::vector<std::string>();

    if (tensor_.keys().count() != 0) {
        auto block = this->tensor_.block_by_id(0);
        auto n_dimensions = block.values_shape().size();

        for (size_t dimension=1; dimension<n_dimensions-1; dimension++) {
            auto labels = block.labels(dimension);
            assert(labels.names().size() == 1);

            result.emplace_back(labels.names()[0]);
        }
    }

    return result;
}

std::vector<std::string> TensorMapHolder::property_names() {
    if (tensor_.keys().count() == 0) {
        return {};
    }

    auto block = this->tensor_.block_by_id(0);
    auto n_dimensions = block.values_shape().size();

    return labels_names(block, n_dimensions - 1);
}

std::vector<std::tuple<TorchLabelsEntry, TorchTensorBlock>> TensorMapHolder::items(TorchTensorMap self) {
    auto result = std::vector<std::tuple<TorchLabelsEntry, TorchTensorBlock>>();

    auto keys = self->keys();
    for (size_t i = 0; i<keys->count(); i++) {
        result.emplace_back(
            torch::make_intrusive<LabelsEntryHolder>(keys, i),
            TensorMapHolder::block_by_id(self, static_cast<int64_t>(i))
        );
    }
    return result;
}

torch::Device TensorMapHolder::device() const {
    return this->keys()->device();
}

torch::Dtype TensorMapHolder::scalar_type() const {
    if (this->keys()->count() == 0) {
        return torch::get_default_dtype_as_scalartype();
    }

    // const_cast is fine here since we will just extract and return a scalar
    auto block = const_cast<metatensor::TensorMap&>(this->tensor_).block_by_id(0);
    auto first_block = torch::make_intrusive<TensorBlockHolder>(std::move(block), torch::IValue());
    return first_block->scalar_type();
}

TorchTensorMap TensorMapHolder::to(
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device
) const {
    auto new_blocks = std::vector<TorchTensorBlock>();
    for (int64_t block_i=0; block_i<this->keys()->count(); block_i++) {
        // const_cast is fine here since we will return a new copy of the data
        // with the different dtype/device
        auto block = const_cast<metatensor::TensorMap&>(this->tensor_).block_by_id(block_i);
        auto torch_block = torch::make_intrusive<TensorBlockHolder>(std::move(block), torch::IValue());
        new_blocks.emplace_back(torch_block->to(dtype, device));
    }
    return torch::make_intrusive<TensorMapHolder>(this->keys()->to(device), new_blocks);
}


TorchTensorMap TensorMapHolder::to_positional(
    torch::IValue positional_1,
    torch::IValue positional_2,
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device,
    torch::optional<std::string> arrays
) const {
    if (arrays.value_or("torch") != "torch") {
        C10_THROW_ERROR(ValueError,
            "`arrays` must be None or 'torch', got '" + arrays.value() + "' instead"
        );
    }

    auto [parsed_dtype, parsed_device] = to_arguments_parse(
        positional_1,
        positional_2,
        dtype,
        device,
        "`TensorMap.to`"
    );

    return this->to(parsed_dtype, parsed_device);
}


std::string TensorMapHolder::print(int64_t max_keys) const {
    std::ostringstream output;
    auto keys = this->keys();
    output << "TensorMap with " << keys->count() << " blocks\n";
    output << "keys:" << keys->print(max_keys, 5);
    return output.str();
}


TorchTensorMap TensorMapHolder::load(const std::string& path) {
    return torch::make_intrusive<TensorMapHolder>(
        TensorMapHolder(metatensor::io::load(path, details::create_torch_array))
    );
}

TorchTensorMap TensorMapHolder::load_buffer(torch::Tensor buffer) {
    if (buffer.scalar_type() != torch::kUInt8) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a tensor of uint8, not " +
            scalar_type_name(buffer.scalar_type())
        );
    }

    if (buffer.sizes().size() != 1) {
        C10_THROW_ERROR(ValueError,
            "`buffer` must be a 1-dimensional tensor"
        );
    }

    return torch::make_intrusive<TensorMapHolder>(
        TensorMapHolder(metatensor::io::load_buffer(
            buffer.data_ptr<uint8_t>(),
            static_cast<size_t>(buffer.size(0)),
            details::create_torch_array
        ))
    );
}


void TensorMapHolder::save(const std::string& path) const {
    return metatensor::io::save(path, this->as_metatensor());
}

torch::Tensor TensorMapHolder::save_buffer() const {
    auto buffer = metatensor::io::save_buffer(this->as_metatensor());
    // move the buffer to the heap so it can escape this function
    // `torch::from_blob` does not take ownership of the data,
    // so we need to register a custom deleter to clean up when
    // the tensor is no longer used
    auto* buffer_data = new std::vector<uint8_t>(std::move(buffer));

    auto options = torch::TensorOptions().dtype(torch::kU8).device(torch::kCPU);
    auto deleter = [=](void* data) {
        delete buffer_data;
    };

    // use a tensor of bytes to store the data
    return torch::from_blob(
        buffer_data->data(),
        {static_cast<int64_t>(buffer_data->size())},
        deleter,
        options
    );
}

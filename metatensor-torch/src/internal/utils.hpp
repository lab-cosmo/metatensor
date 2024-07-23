#ifndef METATENSOR_TORCH_UTILS_HPP
#define METATENSOR_TORCH_UTILS_HPP

#include <torch/types.h>

namespace metatensor_torch {
namespace {

// torch::toString(scalar_type) returns different names compared to the Python
// representation of dtypes, so we use this custom function to get similar names
// as much as possible.
inline std::string scalar_type_name(torch::ScalarType scalar_type) {
    switch (scalar_type) {
        case torch::ScalarType::Byte:
        case torch::ScalarType::Char:
            return "torch.int8";
        case torch::ScalarType::Short:
            return "torch.int16";
        case torch::ScalarType::Int:
            return "torch.int32";
        case torch::ScalarType::Long:
            return "torch.int64";
        case torch::ScalarType::Half:
            return "torch.float16";
        case torch::ScalarType::Float:
            return "torch.float32";
        case torch::ScalarType::Double:
            return "torch.float64";
        case torch::ScalarType::ComplexHalf:
            return "torch.complex32";
        case torch::ScalarType::ComplexFloat:
            return "torch.complex64";
        case torch::ScalarType::ComplexDouble:
            return "torch.complex128";
        case torch::ScalarType::Bool:
            return "torch.bool";
        default:
            return torch::toString(scalar_type);
    }
}

/// Convert a string from `scalar_type_name` to a `torch::ScalarType`
inline torch::ScalarType scalar_type_from_name(std::string name) {
    if (name == "torch.int8") {
        return torch::ScalarType::Char;
    } else if (name == "torch.int16") {
        return torch::ScalarType::Short;
    } else if (name == "torch.int32") {
        return torch::ScalarType::Int;
    } else if (name == "torch.int64") {
        return torch::ScalarType::Long;
    } else if (name == "torch.float16") {
        return torch::ScalarType::Half;
    } else if (name == "torch.float32") {
        return torch::ScalarType::Float;
    } else if (name == "torch.float64") {
        return torch::ScalarType::Double;
    } else if (name == "torch.complex32") {
        return torch::ScalarType::ComplexHalf;
    } else if (name == "torch.complex64") {
        return torch::ScalarType::ComplexFloat;
    } else if (name == "torch.complex128") {
        return torch::ScalarType::ComplexDouble;
    } else if (name == "torch.bool") {
        return torch::ScalarType::Bool;
    } else {
        throw std::runtime_error(
            "Found unknown scalar type name `" + name + "` while " +
            "converting it to a torch scalar type."
        );
    }
}

/// Parse the arguments to the `to` function
inline std::tuple<torch::optional<torch::Dtype>, torch::optional<torch::Device>>
to_arguments_parse(
    torch::IValue positional_1,
    torch::IValue positional_2,
    torch::optional<torch::Dtype> dtype,
    torch::optional<torch::Device> device,
    std::string context
) {
    // handle first positional argument
    if (positional_1.isNone()) {
        // all good, nothing to do
    } else if (positional_1.isDevice()) {
        if (device.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a device twice in " + context);
        } else {
            device = positional_1.toDevice();
        }
    } else if (positional_1.isString()) {
        if (device.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a device twice in " + context);
        } else {
            device = torch::Device(positional_1.toString()->string());
        }
    } else if (positional_1.isInt()) {
        if (dtype.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a dtype twice in " + context);
        } else {
            dtype = static_cast<torch::Dtype>(positional_1.toInt());
        }
    } else {
        C10_THROW_ERROR(TypeError, "unexpected type in " + context + ": "+ positional_1.type()->str());
    }

    // handle second positional argument
    if (positional_2.isNone()) {
        // all good, nothing to do
    } else if (positional_2.isDevice()) {
        if (device.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a device twice in " + context);
        } else {
            device = positional_2.toDevice();
        }
    } else if (positional_2.isString()) {
        if (device.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a device twice in " + context);
        } else {
            device = torch::Device(positional_2.toString()->string());
        }
    } else if (positional_2.isInt()) {
        if (dtype.has_value()) {
            C10_THROW_ERROR(ValueError, "can not give a dtype twice in " + context);
        } else {
            dtype = static_cast<torch::Dtype>(positional_2.toInt());
        }
    } else {
        C10_THROW_ERROR(TypeError, "unexpected type in " + context + ": "+ positional_2.type()->str());
    }

    return std::make_tuple(dtype, device);
}

}}

#endif

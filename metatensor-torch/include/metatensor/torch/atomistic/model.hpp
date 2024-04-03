#ifndef METATENSOR_TORCH_ATOMISTIC_MODEL_HPP
#define METATENSOR_TORCH_ATOMISTIC_MODEL_HPP

#include <vector>
#include <string>

#include <torch/script.h>

#include <metatensor.hpp>

#include "metatensor/torch/labels.hpp"
#include "metatensor/torch/exports.h"

namespace metatensor_torch {

class ModelOutputHolder;
/// TorchScript will always manipulate `ModelOutputHolder` through a `torch::intrusive_ptr`
using ModelOutput = torch::intrusive_ptr<ModelOutputHolder>;

class ModelCapabilitiesHolder;
/// TorchScript will always manipulate `ModelCapabilitiesHolder` through a `torch::intrusive_ptr`
using ModelCapabilities = torch::intrusive_ptr<ModelCapabilitiesHolder>;

class ModelEvaluationOptionsHolder;
/// TorchScript will always manipulate `ModelEvaluationOptionsHolder` through a `torch::intrusive_ptr`
using ModelEvaluationOptions = torch::intrusive_ptr<ModelEvaluationOptionsHolder>;

class ModelMetadataHolder;
/// TorchScript will always manipulate `ModelMetadataHolder` through a `torch::intrusive_ptr`
using ModelMetadata = torch::intrusive_ptr<ModelMetadataHolder>;

/// Check that a given physical quantity is valid and known. This is
/// intentionally not exported with `METATENSOR_TORCH_EXPORT`, and is only
/// intended for internal use.
bool valid_quantity(const std::string& quantity);

/// Check that a given unit is valid and known for some physical quantity. This
/// is intentionally not exported with `METATENSOR_TORCH_EXPORT`, and is only
/// intended for internal use.
void validate_unit(const std::string& quantity, const std::string& unit);


/// Description of one of the quantity a model can compute
class METATENSOR_TORCH_EXPORT ModelOutputHolder: public torch::CustomClassHolder {
public:
    ModelOutputHolder() = default;

    /// Initialize `ModelOutput` with the given data
    ModelOutputHolder(
        std::string quantity,
        std::string unit,
        bool per_atom_,
        std::vector<std::string> explicit_gradients_
    ):
        per_atom(per_atom_),
        explicit_gradients(std::move(explicit_gradients_))
    {
        this->set_quantity(std::move(quantity));
        this->set_unit(std::move(unit));
    }

    ~ModelOutputHolder() override = default;

    /// quantity of the output (e.g. energy, dipole, …).  If this is an empty
    /// string, no unit conversion will be performed.
    const std::string& quantity() const {
        return quantity_;
    }

    /// set the quantity of the output
    void set_quantity(std::string quantity);

    /// unit of the output. If this is an empty string, no unit conversion will
    /// be performed.
    const std::string& unit() const {
        return unit_;
    }
    /// set the unit of the output
    void set_unit(std::string unit);

    /// is the output defined per-atom or for the overall structure
    bool per_atom = false;

    /// Which gradients should be computed eagerly and stored inside the output
    /// `TensorMap`
    std::vector<std::string> explicit_gradients;

    /// Serialize a `ModelOutput` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelOutput` from a JSON string.
    static ModelOutput from_json(std::string_view json);

private:
    std::string quantity_;
    std::string unit_;
};


/// Description of a model's capabilities, i.e. everything a model can do.
class METATENSOR_TORCH_EXPORT ModelCapabilitiesHolder: public torch::CustomClassHolder {
public:
    ModelCapabilitiesHolder() = default;

    /// Initialize `ModelCapabilities` with the given data
    ModelCapabilitiesHolder(
        torch::Dict<std::string, ModelOutput> outputs_,
        std::vector<int64_t> atomic_types_,
        double interaction_range_,
        std::string length_unit,
        std::vector<std::string> supported_devices_,
        std::string dtype
    ):
        outputs(outputs_),
        atomic_types(std::move(atomic_types_)),
        interaction_range(interaction_range_),
        supported_devices(std::move(supported_devices_))
    {
        this->set_length_unit(std::move(length_unit));

        if (!dtype.empty()) {
            this->set_dtype(std::move(dtype));
        }
    }

    ~ModelCapabilitiesHolder() override = default;

    /// all possible outputs from this model and corresponding settings
    torch::Dict<std::string, ModelOutput> outputs;

    /// which types the model can handle
    std::vector<int64_t> atomic_types;

    /// How far a given atom needs to know about other atoms, in the length unit
    /// of the model.
    ///
    /// This is used to properly implement domain decomposition with this model.
    ///
    /// For a short range model, this is the same as the largest neighbor list
    /// cutoff. For a message passing model, this is the cutoff of one
    /// environment times the number of message passing steps. For an explicit
    /// long range model, this should be set to infinity.
    ///
    /// This will default to -1 if not explicitly set by the user.
    double interaction_range = -1.0;

    /// unit of lengths the model expects as input
    const std::string& length_unit() const {
        return length_unit_;
    }
    /// set the unit of length for this model
    void set_length_unit(std::string unit);

    /// Get the `interaction_range` in the length unit of the engine
    double engine_interaction_range(const std::string& engine_length_unit) const;

    /// What devices can this model run on? This should only contain the
    /// `device_type` part of the device, and not the device number (i.e. this
    /// should be `"cuda"`, not `"cuda:0"`).
    ///
    /// Devices should be ordered in order of preference: first one should be
    /// the best device for this model, and so on.
    std::vector<std::string> supported_devices;

    /// Get the dtype of this model. This can be "float32" or "float64", and
    /// must be used by the engine as the dtype of all inputs and outputs for
    /// this model.
    const std::string& dtype() const {
        return dtype_;
    }

    /// Set the dtype of this model.
    void set_dtype(std::string dtype);

    /// Serialize a `ModelCapabilities` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelCapabilities` from a JSON string.
    static ModelCapabilities from_json(std::string_view json);

private:
    std::string length_unit_;
    std::string dtype_;
};


/// Options requested by the simulation engine when running with a model
class METATENSOR_TORCH_EXPORT ModelEvaluationOptionsHolder: public torch::CustomClassHolder {
public:
    ModelEvaluationOptionsHolder() = default;

    /// Initialize `ModelEvaluationOptions` with the given data
    ModelEvaluationOptionsHolder(
        std::string length_unit,
        torch::Dict<std::string, ModelOutput> outputs,
        torch::optional<TorchLabels> selected_atoms
    );

    ~ModelEvaluationOptionsHolder() override = default;

    /// unit of lengths the engine uses in the data it calls the model with
    const std::string& length_unit() const {
        return length_unit_;
    }
    /// set the unit of length used by the engine
    void set_length_unit(std::string unit);

    /// requested outputs for this run and corresponding settings
    torch::Dict<std::string, ModelOutput> outputs;

    /// Only run the calculation for a selected subset of atoms. If this is set
    /// to `None`, run the calculation on all atoms. If this is a set of
    /// `Labels`, it will have two dimensions named `"system"` and `"atom"`,
    /// containing the 0-based indices of all the atoms in the selected subset.
    torch::optional<TorchLabels> get_selected_atoms() const {
        return selected_atoms_;
    }

    /// Setter for `selected_atoms`
    void set_selected_atoms(torch::optional<TorchLabels> selected_atoms);

    /// Serialize a `ModelEvaluationOptions` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelEvaluationOptions` from a JSON string.
    static ModelEvaluationOptions from_json(std::string_view json);

private:
    std::string length_unit_;
    torch::optional<TorchLabels> selected_atoms_ = torch::nullopt;
};


/// Metadata about a specific exported model
class METATENSOR_TORCH_EXPORT ModelMetadataHolder: public torch::CustomClassHolder {
public:
    ModelMetadataHolder() = default;

    /// Initialize `ModelMetadata` with the given information
    ModelMetadataHolder(
        std::string name_,
        std::string description_,
        std::vector<std::string> authors_,
        torch::Dict<std::string, std::vector<std::string>> references_
    ):
        name(std::move(name_)),
        description(std::move(description_)),
        authors(std::move(authors_)),
        references(references_)
    {
        this->validate();
    }

    ~ModelMetadataHolder() override = default;

    /// Name of this model
    std::string name;

    /// Description of this model
    std::string description;

    /// List of authors for this model
    std::vector<std::string> authors;

    /// References for this model. The top level dict can have three keys:
    ///
    /// - "implementation": for reference to software and libraries used in the
    ///   implementation of the model (i.e. for PyTorch
    ///   https://dl.acm.org/doi/10.5555/3454287.3455008)
    /// - "architecture": for reference that introduced the general architecture
    ///   used by this model
    /// - "model": for reference specific to this exact model
    torch::Dict<std::string, std::vector<std::string>> references;

    /// Implementation of Python's `__repr__` and `__str__`, printing all
    /// metadata about this model.
    std::string print() const;

    /// Serialize `ModelMetadata` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `ModelMetadata` from a JSON string.
    static ModelMetadata from_json(std::string_view json);

private:
    /// validate the metadata before using it
    void validate() const;
};

/// Check the exported metatensor atomistic model at the given `path`, and
/// warn/error as required.
METATENSOR_TORCH_EXPORT void check_atomistic_model(std::string path);

/// Check and then load the metatensor atomistic model at the given `path`.
METATENSOR_TORCH_EXPORT torch::jit::Module load_atomistic_model(
    std::string path,
    c10::optional<c10::Device> device = c10::nullopt
);

/// Get the multiplicative conversion factor to use to convert from unit `from`
/// to unit `to`. Both should be units for the given physical `quantity`.
METATENSOR_TORCH_EXPORT double unit_conversion_factor(
    const std::string& quantity,
    const std::string& from_unit,
    const std::string& to_unit
);

}

#endif

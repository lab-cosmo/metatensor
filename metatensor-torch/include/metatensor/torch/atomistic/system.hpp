#ifndef METATENSOR_TORCH_ATOMISTIC_SYSTEM_HPP
#define METATENSOR_TORCH_ATOMISTIC_SYSTEM_HPP

#include <vector>
#include <string>

#include <torch/script.h>

#include <metatensor.hpp>

#include "metatensor/torch/block.hpp"
#include "metatensor/torch/exports.h"

namespace metatensor_torch {

class NeighborListOptionsHolder;
/// TorchScript will always manipulate `NeighborListOptions` through a `torch::intrusive_ptr`
using NeighborListOptions = torch::intrusive_ptr<NeighborListOptionsHolder>;

class SystemHolder;
/// TorchScript will always manipulate `SystemHolder` through a `torch::intrusive_ptr`
using System = torch::intrusive_ptr<SystemHolder>;

/// Options for the calculation of a neighbor list
class METATENSOR_TORCH_EXPORT NeighborListOptionsHolder final: public torch::CustomClassHolder {
public:
    /// Create `NeighborListOptions` with the given `cutoff` and `full_list`.
    ///
    /// `requestor` can be used to store information about who requested the
    /// neighbor list.
    NeighborListOptionsHolder(double cutoff, bool full_list, std::string requestor = "");
    ~NeighborListOptionsHolder() override = default;

    /// Spherical cutoff radius for this neighbor list, in the units of the model
    double cutoff() const {
        return cutoff_;
    }

    /// Get the length unit of the model, this will be used in `engine_cutoff`.
    const std::string& length_unit() const {
        return length_unit_;
    }

    /// Set the length unit to a new value.
    ///
    /// This is typically called by `MetatensorAtomisticModel`, and should not
    /// need to be set by users directly.
    void set_length_unit(std::string length_unit);

    /// Spherical cutoff radius for this neighbor list, in the units of the engine
    double engine_cutoff(const std::string& engine_length_unit) const;

    /// Should the list be a full neighbor list (contains both the pair i->j
    /// and j->i) or a half neighbor list (contains only the pair i->j)
    bool full_list() const {
        return full_list_;
    }

    /// Get the list of strings describing who requested this neighbor list
    std::vector<std::string> requestors() const {
        return requestors_;
    }

    /// Add a new requestor to the list of who requested this neighbor list
    void add_requestor(std::string requestor);

    /// Implementation of Python's `__repr__`
    std::string repr() const;
    /// Implementation of Python's `__str__`
    std::string str() const;

    /// Serialize a `NeighborListOptions` to a JSON string.
    std::string to_json() const;
    /// Load a serialized `NeighborListOptions` from a JSON string.
    static NeighborListOptions from_json(const std::string& json);

private:
    // cutoff in the model units
    double cutoff_;
    std::string length_unit_;
    bool full_list_;
    std::vector<std::string> requestors_;
};

/// Check `NeighborListOptions` for equality. The `requestors` list is ignored
/// when checking for equality
inline bool operator==(const NeighborListOptions& lhs, const NeighborListOptions& rhs) {
    return lhs->cutoff() == rhs->cutoff() && lhs->full_list() == rhs->full_list();
}

/// Check `NeighborListOptions` for inequality.
inline bool operator!=(const NeighborListOptions& lhs, const NeighborListOptions& rhs) {
    return !(lhs == rhs);
}


/// Custom autograd node going (positions, cell) => neighbor list distances
///
/// This does not actually computes the neighbor list, but registers a new node
/// in the overall computational graph, allowing to re-use neighbor list
/// computed outside of torch. The public interface for this is the
/// `register_autograd_neighbors()` function.
class METATENSOR_TORCH_EXPORT NeighborsAutograd: public torch::autograd::Function<NeighborsAutograd> {
public:
    /// This does nothing unless `check_consistency` is true, in which case it
    /// checks that the values in `neighbors` can be re-computed with the
    /// samples and positions/cell
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor positions,
        torch::Tensor cell,
        TorchTensorBlock neighbors,
        bool check_consistency
    );

    /// Compute the gradient of the output w.r.t. positions/cell
    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<torch::Tensor> outputs_grad
    );
};

/// Register a new autograd node going from (`system.positions`, `system.cell`)
/// to the `neighbors` distance vectors.
///
/// This does not recompute the distance vectors, but work as-if all the data in
/// `neighbors.values` was computed directly from `system.positions` and
/// `system.cell`, allowing downstream models to use it directly with full
/// autograd integration.
///
/// `check_consistency` can be set to `true` to run a handful of additional
/// checks in case the data in neighbors does not follow what's expected.
METATENSOR_TORCH_EXPORT void register_autograd_neighbors(
    System system,
    TorchTensorBlock neighbors,
    bool check_consistency
);

/// A System contains all the information about an atomistic system; and should
/// be used as the input of metatensor atomistic models.
class METATENSOR_TORCH_EXPORT SystemHolder final: public torch::CustomClassHolder {
public:
    /// Create a `SystemHolder` with the given `types`, `positions` and
    /// `cell`.
    ///
    /// @param types 1D tensor of 32-bit integer representing the particles
    ///        identity. For atoms, this is typically their atomic numbers.
    /// @param positions 2D tensor of shape (len(types), 3) containing the
    ///        Cartesian positions of all particles in the system.
    /// @param cell 2D tensor of shape (3, 3), describing the bounding box/unit
    ///        cell of the system. Each row should be one of the bounding box
    ///        vector; and columns should contain the x, y, and z components of
    ///        these vectors (i.e. the cell should be given in row-major order).
    ///        Systems are assumed to obey periodic boundary conditions,
    ///        non-periodic systems should set the cell to 0.
    SystemHolder(torch::Tensor types, torch::Tensor positions, torch::Tensor cell);
    ~SystemHolder() override = default;

    /// Get the particle types for all particles in the system.
    torch::Tensor types() const {
        return types_;
    }

    /// Set types for all particles in the system
    void set_types(torch::Tensor types);

    /// Get the positions for the atoms in the system.
    torch::Tensor positions() const {
        return positions_;
    }

    /// Set positions for all particles in the system
    void set_positions(torch::Tensor positions);

    /// Unit cell/bounding box of the system.
    torch::Tensor cell() const {
        return cell_;
    }

    /// Set cell for the system
    void set_cell(torch::Tensor cell);

    /// Get the device used by all the data in this `System`
    torch::Device device() const {
        return this->types_.device();
    }

    /// Get the dtype used by all the floating point data in this `System`
    torch::Dtype scalar_type() const {
        return positions_.scalar_type();
    }

    /// Move all the data in this `System` to the given `dtype` and `device`.
    System to(
        torch::optional<torch::Dtype> dtype = torch::nullopt,
        torch::optional<torch::Device> device = torch::nullopt
    ) const;

    /// Wrapper of the `to` function to enable using it with positional
    /// parameters from Python; for example `to(dtype)`, `to(device)`,
    /// `to(dtype, device=device)`, `to(dtype, device)`, `to(device, dtype)`,
    /// etc.
    System to_positional(
        torch::IValue positional_1,
        torch::IValue positional_2,
        torch::optional<torch::Dtype> dtype,
        torch::optional<torch::Device> device
    ) const;

    /// Get the number of particles in this system
    int64_t size() const {
        return this->types_.size(0);
    }

    /// Add a new neighbor list in this system corresponding to the given
    /// `options`.
    ///
    /// The neighbor list should have the following samples: "first_atom",
    /// "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c",
    /// containing the index of the first and second atom and the number of cell
    /// vector a/b/c to add to the positions difference to get the pair vector.
    ///
    /// The `neighbors` should also have a single component "xyz" with values
    /// `[0, 1, 2]`; and a single property "distance" with value 0.
    ///
    /// The neighbors values must contain the distance vector from the first to
    /// the second atom, i.e. `positions[second_atom] - positions[first_atom] +
    /// cell_shift_a * cell_a + cell_shift_b * cell_b + cell_shift_c * cell_c`.
    void add_neighbor_list(NeighborListOptions options, TorchTensorBlock neighbors);

    /// Retrieve a previously stored neighbor list with the given options, or
    /// throw an error if no such neighbor list exists.
    TorchTensorBlock get_neighbor_list(NeighborListOptions options) const;

    /// Get the options for all neighbor lists registered with this `System`
    std::vector<NeighborListOptions> known_neighbor_lists() const;

    /// Add custom data to this system, stored as `TensorBlock`.
    ///
    /// This is intended for experimentation with models that need more data as
    /// input, and moved into a field of `SystemHolder` later.
    ///
    /// @param name name of the data
    /// @param values values of the data
    /// @param override if true, allow this function to override existing data
    ///                 with the same name
    void add_data(std::string name, TorchTensorBlock values, bool override=false);

    /// Retrieve custom data stored in this System, or throw an error.
    TorchTensorBlock get_data(std::string name) const;

    /// Get the list of data registered with this `System`
    std::vector<std::string> known_data() const;

    /// Implementation of `__str__` and `__repr__` for Python
    std::string str() const;

    /// Serialize a `System` to a JSON string.
    std::string to_json() const;

    /// Load a serialized `System` from a JSON string.
    static System from_json(std::string_view json);

private:
    struct nl_options_compare {
        bool operator()(const NeighborListOptions& a, const NeighborListOptions& b) const {
            assert(a->length_unit() == b->length_unit());
            if (a->full_list() == b->full_list()) {
                return a->cutoff() < b->cutoff();
            } else {
                return static_cast<int>(a->full_list()) < static_cast<int>(b->full_list());
            }
        }
    };

    torch::Tensor types_;
    torch::Tensor positions_;
    torch::Tensor cell_;

    std::map<NeighborListOptions, TorchTensorBlock, nl_options_compare> neighbors_;
    std::unordered_map<std::string, TorchTensorBlock> data_;
};

}

#endif

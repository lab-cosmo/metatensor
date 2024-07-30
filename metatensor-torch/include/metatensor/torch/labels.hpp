#ifndef METATENSOR_TORCH_LABELS_HPP
#define METATENSOR_TORCH_LABELS_HPP

#include <string>
#include <vector>

#include <torch/script.h>

#include <metatensor.hpp>
#include "metatensor/torch/exports.h"

namespace metatensor_torch {

class LabelsHolder;
/// TorchScript will always manipulate `LabelsHolder` through a `torch::intrusive_ptr`
using TorchLabels = torch::intrusive_ptr<LabelsHolder>;

class LabelsEntryHolder;
/// TorchScript will always manipulate `LabelsEntryHolder` through a `torch::intrusive_ptr`
using TorchLabelsEntry = torch::intrusive_ptr<LabelsEntryHolder>;

namespace details {
    /// Transform a torch::IValue containing either a single string, a list of
    /// strings or a tuple of strings into a `std::vector<std::string>`.
    /// `argument_name` is used in the error message.
    std::vector<std::string> normalize_names(torch::IValue names, const std::string& argument_name);
}

/// Wrapper around `metatensor::Labels` for integration with TorchScript
///
/// Python/TorchScript code will typically manipulate
/// `torch::intrusive_ptr<LabelsHolder>` (i.e. `TorchLabels`) instead of
/// instances of `LabelsHolder`.
///
/// The main difference with `metatensor::Labels` is that the values of the
/// labels entries are stored twice: once inside the Rust side labels, and once
/// in a `torch::Tensor`. The data inside the tensor can be moved to different
/// devices if needed.
class METATENSOR_TORCH_EXPORT LabelsHolder: public torch::CustomClassHolder {
public:
    /// Construct `LabelsHolder` from a set of names and the corresponding
    /// values
    ///
    /// The names should be either a single string or a list/tuple of strings;
    /// and the values should be a 2D tensor of integers.
    LabelsHolder(torch::IValue names, torch::Tensor values);

    /// Convenience constructor for building `LabelsHolder` in C++, similar to
    /// `metatensor::Labels`.
    static TorchLabels create(
        std::vector<std::string> names,
        const std::vector<std::initializer_list<int32_t>>& values
    );

    /// Get a view of `labels` corresponding to only the given columns names
    static TorchLabels view(const TorchLabels& labels, std::vector<std::string> names);

    /// Create Labels with a single entry, and a single dimension named `"_"`
    static TorchLabels single();

    /// Create Labels with the given dimension names and zero entries
    static TorchLabels empty(torch::IValue names);

    /// Create Labels with a single dimension with the given name and values in
    /// the [0, stop) range
    static TorchLabels range(std::string name, int64_t end);

    /// Create a `LabelsHolder` from a pre-existing `metatensor::Labels`
    explicit LabelsHolder(metatensor::Labels labels);

    /// Get the names of the dimensions/columns of these Labels
    std::vector<std::string> names() const {
        return names_;
    }

    /// Get the values of these labels as a torch Tensor
    torch::Tensor values() const {
        return values_;
    }

    /// Create new `Labels` with a new dimension with the given `name` and
    /// `values` added to the end of the dimensions list.
    TorchLabels append(std::string name, torch::Tensor values) const;

    /// Create new `Labels` with a new dimension with the given `name` and
    /// `values` before `index`.
    TorchLabels insert(int64_t index, std::string name, torch::Tensor values) const;

    /// Create new `Labels` with permuted dimensions
    TorchLabels permute(std::vector<int64_t> dimensions_indexes) const;

    /// Create new `Labels` with `name` removed from the dimensions list
    TorchLabels remove(std::string name) const;

    /// Create new `Labels` with `old_name` renamed to `new_name` in the
    /// dimensions list
    TorchLabels rename(std::string old_name, std::string new_name) const;

    /// Get the current device for these `Labels`
    torch::Device device() const {
        return values_.device();
    }

    /// Move the values for these Labels to the given `device`
    TorchLabels to(torch::IValue device) const;

    /// Move the values for these Labels to the given `device`
    TorchLabels to(torch::Device device) const;

    /// Get the values associated with a single dimension (i.e. a single column
    /// of `values()`) in these labels.
    torch::Tensor column(std::string dimension);

    /// Get the number of entries in this set of Labels.
    ///
    /// This is the same as `values().size(0)`
    int64_t count() const {
        return values_.size(0);
    }

    /// Get the number of dimensions in this set of Labels.
    ///
    /// This is the same as `values().size(1)`
    int64_t size() const {
        return values_.size(1);
    }

    /// Get the position of the given `entry` in this set of Labels, or None if
    /// the entry is not part of these labels.
    ///
    /// @param entry one of:
    ///    - a `LabelsEntry`
    ///    - a 1-D torch::Tensor containing integers;
    ///    - a list of integers;
    ///    - a tuple of integers;
    torch::optional<int64_t> position(torch::IValue entry) const;

    /// Print the names and values of these Labels to a string, including at
    /// most `max_entries` entries (set this to -1 to print all entries), and
    /// indenting all lines after the first with `indent` spaces.
    std::string print(int64_t max_entries, int64_t indent) const;

    /// Implementation of `__str__` for Python
    std::string str() const;

    /// Implementation of `__repr__` for Python
    std::string repr() const;

    /// Get the underlying metatensor::Labels
    const metatensor::Labels& as_metatensor() const;

    /// Is this a view inside existing Labels or an owned Labels?
    bool is_view() const {
        return !labels_.has_value();
    }

    /// Transform a view of Labels into owned Labels, which can be further given
    /// to metatensor functions. This does nothing if the Labels are already
    /// owned.

    // A view is created by the `view` function (also `__getitem__` in Python),
    // and does not have a corresponding `metatensor::Labels` (`labels_` is
    // `nullopt`)
    TorchLabels to_owned() const;

    /// Get the union of `this` and `other`
    TorchLabels set_union(const TorchLabels& other) const;

    /// Get the union of `this` and `other`, as well as the mapping from
    /// positions of entries in the input to the position of entries in the
    /// output.
    std::tuple<TorchLabels, torch::Tensor, torch::Tensor> union_and_mapping(const TorchLabels& other) const;

    /// Get the intersection of `this` and `other`
    TorchLabels set_intersection(const TorchLabels& other) const;

    /// Get the intersection of `this` and `other`, as well as the mapping from
    /// positions of entries in the input to the position of entries in the
    /// output.
    std::tuple<TorchLabels, torch::Tensor, torch::Tensor> intersection_and_mapping(const TorchLabels& other) const;

    /// Select entries in these `Labels` that match the `selection`.
    ///
    /// The selection's names must be a subset of the names of these labels.
    ///
    /// All entries in these `Labels` that match one of the entry in the
    /// `selection` for all the selection's dimension will be picked. Any entry
    /// in the `selection` but not in these `Labels` will be ignored.
    torch::Tensor select(const TorchLabels& selection) const;

    /// Load serialized Labels from the given path
    static TorchLabels load(const std::string& path);

    /// Load serialized Labels from an in-memory buffer (represented as a
    /// `torch::Tensor` of bytes)
    static TorchLabels load_buffer(torch::Tensor buffer);

    /// Serialize and save Labels to the given path
    void save(const std::string& path) const;

    /// Serialize and save Labels to an in-memory buffer (represented as a
    /// `torch::Tensor` of bytes)
    torch::Tensor save_buffer() const;

private:
    /// main constructor, checking everything in debug mode & registering the
    /// `values` as user data for the `labels`.
    LabelsHolder(std::vector<std::string> names, torch::Tensor values, metatensor::Labels labels);

    /// marker type to differentiate the private constructor below from the main
    /// one
    struct CreateView {};

    /// Create a view for an existing `LabelsHolder`
    LabelsHolder(std::vector<std::string> names, torch::Tensor values, CreateView);

    friend class torch::intrusive_ptr<LabelsHolder>;

    /// names of the Labels, stored here for easier retrieval from Python
    std::vector<std::string> names_;

    /// Keep the values of the Labels inside a Tensor as well
    torch::Tensor values_;

    /// Underlying metatensor labels, this is undefined when the Labels is
    /// actually a view (with selected columns) into another Labels
    torch::optional<metatensor::Labels> labels_;
};

/// Check two `LabelsHolder` for equality
inline bool operator==(const LabelsHolder& lhs, const LabelsHolder& rhs) {
    return lhs.as_metatensor() == rhs.as_metatensor();
}

/// Check two `LabelsHolder` for inequality
inline bool operator!=(const LabelsHolder& lhs, const LabelsHolder& rhs) {
    return !(lhs == rhs);
}

/// Check for equality between `LabelsHolder` and `metatensor::Labels`
inline bool operator==(const LabelsHolder& lhs, const metatensor::Labels& rhs) {
    return lhs.as_metatensor() == rhs;
}

/// Check for inequality between `LabelsHolder` and `metatensor::Labels`
inline bool operator!=(const LabelsHolder& lhs, const metatensor::Labels& rhs) {
    return !(lhs == rhs);
}

/// Check for equality between `LabelsHolder` and `metatensor::Labels`
inline bool operator==(const metatensor::Labels& lhs, const LabelsHolder& rhs) {
    return lhs == rhs.as_metatensor();
}

/// Check for inequality between `LabelsHolder` and `metatensor::Labels`
inline bool operator!=(const metatensor::Labels& lhs, const LabelsHolder& rhs) {
    return !(lhs == rhs);
}


/// A single entry inside a `TorchLabels`
class METATENSOR_TORCH_EXPORT LabelsEntryHolder: public torch::CustomClassHolder {
public:
    /// Create a new `LabelsEntryHolder` corresponding to the entry at the given
    /// `index` in the given `labels`
    LabelsEntryHolder(TorchLabels labels, int64_t index);

    /// Get the names of the dimensions/columns of these Labels
    std::vector<std::string> names() const {
        return labels_->names();
    }

    /// Get the values of these labels as a torch Tensor
    torch::Tensor values() const {
        return values_;
    }

    /// Get the current device for this `LabelsEntry`
    torch::Device device() const {
        return values_.device();
    }

    /// Get the number of dimensions in this `LabelsEntry`.
    ///
    /// This is the same as `values().size(0)`
    int64_t size() const {
        return values_.size(0);
    }

    /// Get the value at `index` in this `LabelsEntry`
    int32_t operator[](int64_t index) const {
        return values_[index].item<int32_t>();
    }

    /// Get the value for the `name` dimension in this `LabelsEntry`
    int32_t operator[](const std::string& name) const;

    /// implementation of __getitem__, forwarding to one of the operator[]
    int64_t getitem(torch::IValue index) const;

    /// Print this entry as a named tuple (i.e. `(key=value, key=value)`).
    std::string print() const;

    /// Implementation of __repr__ for Python
    std::string repr() const;

private:
    torch::Tensor values_;

    TorchLabels labels_;
};


/// Check two `LabelsEntryHolder` for equality
inline bool operator==(const LabelsEntryHolder& lhs, const LabelsEntryHolder& rhs) {
    return lhs.names() == rhs.names() && torch::all(lhs.values() == rhs.values()).item<bool>();
}

/// Check two `LabelsEntryHolder` for inequality
inline bool operator!=(const LabelsEntryHolder& lhs, const LabelsEntryHolder& rhs) {
    return !(lhs == rhs);
}

}

#endif

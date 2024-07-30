#ifndef METATENSOR_HPP
#define METATENSOR_HPP

#include <array>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <exception>
#include <functional>
#include <type_traits>
#include <initializer_list>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include "metatensor.h"

/// This file contains the C++ API to metatensor, manually built on top of the C
/// API defined in `metatensor.h`. This API uses the standard C++ library where
/// convenient, but also allow to drop back to the C API if required, by
/// providing functions to extract the C API handles (named `as_mts_XXX`).

static_assert(sizeof(char) == sizeof(uint8_t), "char must be 8-bits wide");

namespace metatensor_torch {
    class LabelsHolder;
    class TensorBlockHolder;
    class TensorMapHolder;
}

namespace metatensor {
class Labels;
class TensorMap;
class TensorBlock;

/// Exception class used for all errors in metatensor
class Error: public std::runtime_error {
public:
    /// Create a new Error with the given `message`
    Error(const std::string& message): std::runtime_error(message) {}
};

namespace details {
    /// Singleton class storing the last exception throw by a C++ callback.
    ///
    /// When passing callbacks from C++ to Rust, we need to convert exceptions
    /// into status code (see the `catch` blocks in this file). This class
    /// allows to save the message associated with an exception, and rethrow an
    /// exception with the same message later (the actual exception type is lost
    /// in the process).
    class LastCxxError {
    public:
        /// Set the last error message to `message`
        static void set_message(std::string message) {
            auto& stored_message = LastCxxError::get();
            stored_message = std::move(message);
        }

        /// Get the last error message
        static const std::string& message() {
            return LastCxxError::get();
        }

    private:
        static std::string& get() {
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wexit-time-destructors"
            /// we are using a per-thread static value to store the last C++
            /// exception.
            static thread_local std::string STORED_MESSAGE;
            #pragma clang diagnostic pop

            return STORED_MESSAGE;
        }
    };

    /// Check if a return status from the C API indicates an error, and if it is
    /// the case, throw an exception of type `metatensor::Error` with the last
    /// error message from the library.
    inline void check_status(mts_status_t status) {
        if (status == MTS_SUCCESS) {
            return;
        } else if (status > 0) {
            throw Error(mts_last_error());
        } else { // status < 0
            throw Error("error in C++ callback: " + LastCxxError::message());
        }
    }

    /// Call the given `function` with the given `args` (the function should
    /// return an `mts_status_t`), catching any C++ exception, and translating
    /// them to negative metatensor error code.
    ///
    /// This is required to prevent callbacks unwinding through the C API.
    template<typename Function, typename ...Args>
    inline mts_status_t catch_exceptions(Function function, Args ...args) {
        try {
            return function(std::move(args)...);
        } catch (const std::exception& e) {
            details::LastCxxError::set_message(e.what());
            return -1;
        } catch (...) {
            details::LastCxxError::set_message("error was not an std::exception");
            return -128;
        }
    }

    /// Check if a pointer allocated by the C API is null, and if it is the
    /// case, throw an exception of type `metatensor::Error` with the last error
    /// message from the library.
    inline void check_pointer(const void* pointer) {
        if (pointer == nullptr) {
            throw Error(mts_last_error());
        }
    }

    /// Compute the product of all values in the `shape` vector
    inline size_t product(const std::vector<size_t>& shape) {
        size_t result = 1;
        for (auto size: shape) {
            result *= size;
        }
        return result;
    }

    /// Get the N-dimensional index corresponding to the given linear `index`
    /// and array `shape`
    inline std::vector<size_t> cartesian_index(const std::vector<size_t>& shape, size_t index) {
        auto result = std::vector<size_t>(shape.size(), 0);
        for (size_t i=0; i<shape.size(); i++) {
            result[i] = index % shape[i];
            index = index / shape[i];
        }
        assert(index == 0);
        return result;
    }

    /// Get the linear index corresponding to the N-dimensional
    /// `index[index_size]`, according to the given array `shape`
    inline size_t linear_index(const std::vector<size_t>& shape, const size_t* index, size_t index_size) {
        assert(index_size != 0);
        assert(index_size == shape.size());

        if (index_size == 1) {
            assert(index[0] < shape[0] && "out of bounds");
            return index[0];
        } else {
            assert(index[0] < shape[0]);
            auto linear_index = index[0];
            for (size_t i=1; i<index_size; i++) {
                assert(index[i] < shape[i] && "out of bounds");
                linear_index *= shape[i];
                linear_index += index[i];
            }

            return linear_index;
        }
    }

    template<size_t N>
    size_t linear_index(const std::vector<size_t>& shape, const std::array<size_t, N>& index) {
        return linear_index(shape, index.data(), index.size());
    }

    inline size_t linear_index(const std::vector<size_t>& shape, const std::vector<size_t>& index) {
        return linear_index(shape, index.data(), index.size());
    }

    Labels labels_from_cxx(const std::vector<std::string>& names, const int32_t* values, size_t count);
}

/******************************************************************************/
/******************************************************************************/
/******************************************************************************/

/******************************************************************************/
/******************************************************************************/
/*                                                                            */
/*                 N-Dimensional arrays handling                              */
/*                                                                            */
/******************************************************************************/
/******************************************************************************/


/// Simple N-dimensional array interface
///
/// This class can either be a non-owning view inside some existing memory (for
/// example memory allocated by Rust); or own its memory (in the form of an
/// `std::vector<double>`). If the array does not own its memory, accessing it
/// is only valid for as long as the memory is kept alive.
///
/// The API of this class is very intentionally minimal to keep metatensor as
/// simple as possible. Feel free to wrap the corresponding data inside types
/// with richer API such as Eigen, Boost, etc.
template<typename T>
class NDArray {
public:
    /// Create a new empty `NDArray`, with shape `[0, 0]`.
    NDArray(): NDArray(nullptr, {0, 0}, true) {}

    /// Create a new `NDArray` using a non-owning view in `const` memory with
    /// the given `shape`.
    ///
    /// `data` must point to contiguous memory containing the right number of
    /// elements as described by the `shape`, which will be interpreted as an
    /// N-dimensional array in row-major order. The resulting `NDArray` is
    /// only valid for as long as `data` is.
    NDArray(const T* data, std::vector<size_t> shape):
        NDArray(data, std::move(shape), /*is_const*/ true) {}

    /// Create a new `NDArray` using a non-owning view in non-`const` memory
    /// with the given `shape`.
    ///
    /// `data` must point to contiguous memory containing the right number of
    /// elements as described by the `shape`, which will be interpreted as an
    /// N-dimensional array in row-major order. The resulting `NDArray` is
    /// only valid for as long as `data` is.
    NDArray(T* data, std::vector<size_t> shape):
        NDArray(data, std::move(shape), /*is_const*/ false) {}

    /// Create a new `NDArray` *owning* its `data` with the given `shape`.
    NDArray(std::vector<T> data, std::vector<size_t> shape):
        NDArray(data.data(), std::move(shape), /*is_const*/ false)
    {
        using vector_t = std::vector<T>;

        owned_data_ = reinterpret_cast<void*>(new vector_t(std::move(data)));
        deleter_ = [](void* data){
            auto data_vector = reinterpret_cast<vector_t*>(data);
            delete data_vector;
        };
    }

    ~NDArray() {
        deleter_(this->owned_data_);
    }

    /// NDArray is not copy-constructible
    NDArray(const NDArray&) = delete;
    /// NDArray can not be copy-assigned
    NDArray& operator=(const NDArray& other) = delete;

    /// NDArray is move-constructible
    NDArray(NDArray&& other) noexcept: NDArray() {
        *this = std::move(other);
    }

    /// NDArray can be move-assigned
    NDArray& operator=(NDArray&& other) noexcept {
        this->deleter_(this->owned_data_);

        this->data_ = std::move(other.data_);
        this->shape_ = std::move(other.shape_);
        this->is_const_ = std::move(other.is_const_);
        this->owned_data_ = std::move(other.owned_data_);
        this->deleter_ = std::move(other.deleter_);

        other.data_ = nullptr;
        other.owned_data_ = nullptr;
        other.deleter_ = [](void*){};

        return *this;
    }

    /// Is this NDArray a view into external data?
    bool is_view() const {
        return owned_data_ == nullptr;
    }

    /// Get the value inside this `NDArray` at the given index
    ///
    /// ```
    /// auto array = NDArray(...);
    ///
    /// double value = array(2, 3, 1);
    /// ```
    template<typename ...Args>
    T operator()(Args... args) const & {
        auto index = std::array<size_t, sizeof... (Args)>{static_cast<size_t>(args)...};
        if (index.size() != shape_.size()) {
            throw Error(
                "expected " + std::to_string(shape_.size()) +
                " indexes in NDArray::operator(), got " + std::to_string(index.size())
            );
        }
        return data_[details::linear_index(shape_, index)];
    }

    /// Get a reference to the value inside this `NDArray` at the given index
    ///
    /// ```
    /// auto array = NDArray(...);
    ///
    /// array(2, 3, 1) = 5.2;
    /// ```
    template<typename ...Args>
    T& operator()(Args... args) & {
        if (is_const_) {
            throw Error("This NDArray is const, can not get non const access to it");
        }

        auto index = std::array<size_t, sizeof... (Args)>{static_cast<size_t>(args)...};
        if (index.size() != shape_.size()) {
            throw Error(
                "expected " + std::to_string(shape_.size()) +
                " indexes in Labels::operator(), got " + std::to_string(index.size())
            );
        }
        return data_[details::linear_index(shape_, index)];
    }

    template<typename ...Args>
    T& operator()(Args... args) && = delete;

    /// Get the data pointer for this array, i.e. the pointer to the first
    /// element.
    const T* data() const & {
        return data_;
    }

    /// Get the data pointer for this array, i.e. the pointer to the first
    /// element.
    T* data() & {
        if (is_const_) {
            throw Error("This NDArray is const, can not get non const access to it");
        }
        return data_;
    }

    const T* data() && = delete;

    /// Get the shape of this array
    const std::vector<size_t>& shape() const & {
        return shape_;
    }

    const std::vector<size_t>& shape() && = delete;

    /// Check if this array is empty, i.e. if at least one of the shape element
    /// is 0.
    bool is_empty() const {
        for (auto s: shape_) {
            if (s == 0) {
                return true;
            }
        }
        return false;
    }

private:
    /// Create an `NDArray` from a pointer to the (row-major) data & shape.
    ///
    /// The `is_const` parameter controls whether this class should allow
    /// non-const access to the data.
    NDArray(const T* data, std::vector<size_t> shape, bool is_const):
        data_(const_cast<T*>(data)),
        shape_(std::move(shape)),
        is_const_(is_const),
        deleter_([](void*){})
    {
        validate();
    }

    /// Create a 2D NDArray from a vector of initializer lists. All the inner
    /// lists must have the same `size`.
    ///
    /// This allows creating an array (and in particular a set of Labels) with
    /// `NDArray({{1, 2}, {3, 4}, {5, 6}}, 2)`
    NDArray(std::vector<std::initializer_list<T>> data, size_t size):
        data_(nullptr),
        shape_({data.size(), size}),
        is_const_(false),
        deleter_([](void*){})
    {
        using vector_t = std::vector<T>;
        auto vector = std::vector<T>();
        vector.reserve(data.size() * size);
        for (auto row: std::move(data)) {
            if (row.size() != size) {
                throw Error(
                    "invalid size for row: expected " + std::to_string(size) +
                    " got " + std::to_string(row.size())
                );
            }

            for (auto entry: row) {
                vector.emplace_back(entry);
            }
        }

        data_ = vector.data();
        owned_data_ = reinterpret_cast<void*>(new vector_t(std::move(vector)));
        deleter_ = [](void* data){
            auto data_vector = reinterpret_cast<vector_t*>(data);
            delete data_vector;
        };
        validate();
    }

    friend class Labels;

    void validate() const {
        static_assert(
            std::is_arithmetic<T>::value,
            "NDArray only works with integers and floating points"
        );

        if (shape_.empty()) {
            throw Error("invalid parameters to NDArray, shape should contain at least one element");
        }

        size_t size = 1;
        for (auto s: shape_) {
            size *= s;
        }

        if (size != 0 && data_ == nullptr) {
            throw Error("invalid parameters to NDArray, got null data pointer and non zero size");
        }
    }

    /// Pointer to the data used by this array
    T* data_ = nullptr;
    /// Full shape of this array
    std::vector<size_t> shape_ = {0, 0};
    /// Is this array const? This will dynamically prevent calling non-const
    /// function on it.
    bool is_const_ = true;
    /// Type-erased owned data for this array. This is a `nullptr` if this array
    /// is a view.
    void* owned_data_ = nullptr;
    /// Custom delete function for the `owned_data_`.
    std::function<void(void*)> deleter_;
};


/// Compare this `NDArray` with another `NDarray`. The array are equal if
/// and only if both the shape and data are equal.
template<typename T>
bool operator==(const NDArray<T>& lhs, const NDArray<T>& rhs) {
    if (lhs.shape() != rhs.shape()) {
        return false;
    }
    return std::memcmp(lhs.data(), rhs.data(), sizeof(T) * details::product(lhs.shape())) == 0;
}

/// Compare this `NDArray` with another `NDarray`. The array are equal if
/// and only if both the shape and data are equal.
template<typename T>
bool operator!=(const NDArray<T>& lhs, const NDArray<T>& rhs) {
    return !(lhs == rhs);
}


/// `DataArrayBase` manages n-dimensional arrays used as data in a block or
/// tensor map. The array itself if opaque to this library and can come from
/// multiple sources: Rust program, a C/C++ program, a Fortran program, Python
/// with numpy or torch. The data does not have to live on CPU, or even on the
/// same machine where this code is executed.
///
/// **WARNING**: all function implementations **MUST** be thread-safe, and can
/// be called from multiple threads at the same time. The `DataArrayBase` itself
/// might be moved from one thread to another.
class DataArrayBase {
public:
    DataArrayBase() = default;
    virtual ~DataArrayBase() = default;

    /// DataArrayBase can be copy-constructed
    DataArrayBase(const DataArrayBase&) = default;
    /// DataArrayBase can be copy-assigned
    DataArrayBase& operator=(const DataArrayBase&) = default;
    /// DataArrayBase can be move-constructed
    DataArrayBase(DataArrayBase&&) noexcept = default;
    /// DataArrayBase can be move-assigned
    DataArrayBase& operator=(DataArrayBase&&) noexcept = default;

    /// Convert a concrete `DataArrayBase` to a C-compatible `mts_array_t`
    ///
    /// The `mts_array_t` takes ownership of the data, which should be released
    /// with `mts_array_t::destroy`.
    static mts_array_t to_mts_array_t(std::unique_ptr<DataArrayBase> data) {
        mts_array_t array;
        std::memset(&array, 0, sizeof(array));

        array.ptr = data.release();

        array.destroy = [](void* array) {
            auto ptr = std::unique_ptr<DataArrayBase>(static_cast<DataArrayBase*>(array));
            // let ptr go out of scope
        };

        array.origin = [](const void* array, mts_data_origin_t* origin) {
            return details::catch_exceptions([](const void* array, mts_data_origin_t* origin){
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                *origin = cxx_array->origin();
                return MTS_SUCCESS;
            }, array, origin);
        };

        array.copy = [](const void* array, mts_array_t* new_array) {
            return details::catch_exceptions([](const void* array, mts_array_t* new_array){
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                auto copy = cxx_array->copy();
                *new_array = DataArrayBase::to_mts_array_t(std::move(copy));
                return MTS_SUCCESS;
            }, array, new_array);
        };

        array.create = [](const void* array, const uintptr_t* shape, uintptr_t shape_count, mts_array_t* new_array) {
            return details::catch_exceptions([](
                const void* array,
                const uintptr_t* shape,
                uintptr_t shape_count,
                mts_array_t* new_array
            ) {
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                auto cxx_shape = std::vector<size_t>();
                for (size_t i=0; i<static_cast<size_t>(shape_count); i++) {
                    cxx_shape.push_back(static_cast<size_t>(shape[i]));
                }
                auto copy = cxx_array->create(std::move(cxx_shape));
                *new_array = DataArrayBase::to_mts_array_t(std::move(copy));
                return MTS_SUCCESS;
            }, array, shape, shape_count, new_array);
        };

        array.data = [](void* array, double** data) {
            return details::catch_exceptions([](void* array, double** data){
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                *data = cxx_array->data();
                return MTS_SUCCESS;
            }, array, data);
        };

        array.shape = [](const void* array, const uintptr_t** shape, uintptr_t* shape_count) {
            return details::catch_exceptions([](const void* array, const uintptr_t** shape, uintptr_t* shape_count){
                const auto* cxx_array = static_cast<const DataArrayBase*>(array);
                const auto& cxx_shape = cxx_array->shape();
                *shape = cxx_shape.data();
                *shape_count = static_cast<uintptr_t>(cxx_shape.size());
                return MTS_SUCCESS;
            }, array, shape, shape_count);
        };

        array.reshape = [](void* array, const uintptr_t* shape, uintptr_t shape_count) {
            return details::catch_exceptions([](void* array, const uintptr_t* shape, uintptr_t shape_count){
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                auto cxx_shape = std::vector<uintptr_t>(shape, shape + shape_count);
                cxx_array->reshape(std::move(cxx_shape));
                return MTS_SUCCESS;
            }, array, shape, shape_count);
        };

        array.swap_axes = [](void* array, uintptr_t axis_1, uintptr_t axis_2) {
            return details::catch_exceptions([](void* array, uintptr_t axis_1, uintptr_t axis_2){
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                cxx_array->swap_axes(axis_1, axis_2);
                return MTS_SUCCESS;
            }, array, axis_1, axis_2);
        };

        array.move_samples_from = [](
            void* array,
            const void* input,
            const mts_sample_mapping_t* samples,
            uintptr_t samples_count,
            uintptr_t property_start,
            uintptr_t property_end
        ) {
            return details::catch_exceptions([](
                void* array,
                const void* input,
                const mts_sample_mapping_t* samples,
                uintptr_t samples_count,
                uintptr_t property_start,
                uintptr_t property_end
            ) {
                auto* cxx_array = static_cast<DataArrayBase*>(array);
                const auto* cxx_input = static_cast<const DataArrayBase*>(input);
                auto cxx_samples = std::vector<mts_sample_mapping_t>(samples, samples + samples_count);

                cxx_array->move_samples_from(*cxx_input, cxx_samples, property_start, property_end);
                return MTS_SUCCESS;
            }, array, input, samples, samples_count, property_start, property_end);
        };

        return array;
    }

    /// Get "data origin" for this array in.
    ///
    /// Users of `DataArrayBase` should register a single data
    /// origin with `mts_register_data_origin`, and use it for all compatible
    /// arrays.
    virtual mts_data_origin_t origin() const = 0;

    /// Make a copy of this DataArrayBase and return the new array. The new
    /// array is expected to have the same data origin and parameters (data
    /// type, data location, etc.)
    virtual std::unique_ptr<DataArrayBase> copy() const = 0;

    /// Create a new array with the same options as the current one (data type,
    /// data location, etc.) and the requested `shape`.
    ///
    /// The new array should be filled with zeros.
    virtual std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape) const = 0;

    /// Get a pointer to the underlying data storage.
    ///
    /// This function is allowed to fail if the data is not accessible in RAM,
    /// not stored as 64-bit floating point values, or not stored as a
    /// C-contiguous array.
    virtual double* data() & = 0;

    double* data() && = delete;

    /// Get the shape of this array
    virtual const std::vector<uintptr_t>& shape() const & = 0;

    const std::vector<uintptr_t>& shape() && = delete;

    /// Set the shape of this array to the given `shape`
    virtual void reshape(std::vector<uintptr_t> shape) = 0;

    /// Swap the axes `axis_1` and `axis_2` in this `array`.
    virtual void swap_axes(uintptr_t axis_1, uintptr_t axis_2) = 0;

    /// Set entries in the current array taking data from the `input` array.
    ///
    /// This array is guaranteed to be created by calling `mts_array_t::create`
    /// with one of the arrays in the same block or tensor map as the `input`.
    ///
    /// The `samples` indicate where the data should be moved from `input` to
    /// the current DataArrayBase.
    ///
    /// This function should copy data from `input[samples[i].input, ..., :]` to
    /// `array[samples[i].output, ..., property_start:property_end]` for `i` up
    /// to `samples_count`. All indexes are 0-based.
    virtual void move_samples_from(
        const DataArrayBase& input,
        std::vector<mts_sample_mapping_t> samples,
        uintptr_t property_start,
        uintptr_t property_end
    ) = 0;
};


/// Very basic implementation of DataArrayBase in C++.
///
/// This is included as an example implementation of DataArrayBase, and to make
/// metatensor usable without additional dependencies. For other uses cases, it
/// might be better to implement DataArrayBase on your data, using
/// functionalities from `Eigen`, `Boost.Array`, etc.
class SimpleDataArray: public metatensor::DataArrayBase {
public:
    /// Create a SimpleDataArray with the given `shape`, and all elements set to
    /// `value`
    SimpleDataArray(std::vector<uintptr_t> shape, double value = 0.0):
        shape_(std::move(shape)), data_(details::product(shape_), value) {}

    /// Create a SimpleDataArray with the given `shape` and `data`.
    ///
    /// The data is interpreted as a row-major n-dimensional array.
    SimpleDataArray(std::vector<uintptr_t> shape, std::vector<double> data):
        shape_(std::move(shape)),
        data_(std::move(data))
    {
        if (data_.size() != details::product(shape_)) {
            throw Error("the shape and size of the data don't match in SimpleDataArray");
        }
    }

    ~SimpleDataArray() override = default;

    /// SimpleDataArray can be copy-constructed
    SimpleDataArray(const SimpleDataArray&) = default;
    /// SimpleDataArray can be copy-assigned
    SimpleDataArray& operator=(const SimpleDataArray&) = default;
    /// SimpleDataArray can be move-constructed
    SimpleDataArray(SimpleDataArray&&) noexcept = default;
    /// SimpleDataArray can be move-assigned
    SimpleDataArray& operator=(SimpleDataArray&&) noexcept = default;

    mts_data_origin_t origin() const override {
        mts_data_origin_t origin = 0;
        mts_register_data_origin("metatensor::SimpleDataArray", &origin);
        return origin;
    }

    double* data() & override {
        return data_.data();
    }

    const std::vector<uintptr_t>& shape() const & override {
        return shape_;
    }

    void reshape(std::vector<uintptr_t> shape) override {
        if (details::product(shape_) != details::product(shape)) {
            throw metatensor::Error("invalid shape in reshape");
        }
        shape_ = std::move(shape);
    }

    void swap_axes(uintptr_t axis_1, uintptr_t axis_2) override {
        auto new_data = std::vector<double>(details::product(shape_), 0.0);
        auto new_shape = shape_;
        std::swap(new_shape[axis_1], new_shape[axis_2]);

        for (size_t i=0; i<details::product(shape_); i++) {
            auto index = details::cartesian_index(shape_, i);
            std::swap(index[axis_1], index[axis_2]);

            new_data[details::linear_index(new_shape, index)] = data_[i];
        }

        shape_ = std::move(new_shape);
        data_ = std::move(new_data);
    }

    std::unique_ptr<DataArrayBase> copy() const override {
        return std::unique_ptr<DataArrayBase>(new SimpleDataArray(*this));
    }

    std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape) const override {
        return std::unique_ptr<DataArrayBase>(new SimpleDataArray(std::move(shape)));
    }

    void move_samples_from(
        const DataArrayBase& input,
        std::vector<mts_sample_mapping_t> samples,
        uintptr_t property_start,
        uintptr_t property_end
    ) override {
        const auto& input_array = dynamic_cast<const SimpleDataArray&>(input);
        assert(input_array.shape_.size() == this->shape_.size());

        size_t property_count = property_end - property_start;
        size_t property_dim = shape_.size() - 1;
        assert(input_array.shape_[property_dim] == property_count);

        auto input_index = std::vector<size_t>(shape_.size(), 0);
        auto output_index = std::vector<size_t>(shape_.size(), 0);

        for (const auto& sample: samples) {
            input_index[0] = sample.input;
            output_index[0] = sample.output;

            if (property_dim == 1) {
                // no components
                for (size_t property_i=0; property_i<property_count; property_i++) {
                    input_index[property_dim] = property_i;
                    output_index[property_dim] = property_i + property_start;

                    auto value = input_array.data_[details::linear_index(input_array.shape_, input_index)];
                    this->data_[details::linear_index(shape_, output_index)] = value;
                }
            } else {
                auto last_component_dim = shape_.size() - 2;
                for (size_t component_i=1; component_i<shape_.size() - 1; component_i++) {
                    input_index[component_i] = 0;
                }

                bool done = false;
                while (!done) {
                    for (size_t component_i=1; component_i<shape_.size() - 1; component_i++) {
                        output_index[component_i] = input_index[component_i];
                    }

                    for (size_t property_i=0; property_i<property_count; property_i++) {
                        input_index[property_dim] = property_i;
                        output_index[property_dim] = property_i + property_start;

                        auto value = input_array.data_[details::linear_index(input_array.shape_, input_index)];
                        this->data_[details::linear_index(shape_, output_index)] = value;
                    }

                    input_index[last_component_dim] += 1;
                    for (size_t component_i=last_component_dim; component_i>2; component_i--) {
                        if (input_index[component_i] >= shape_[component_i]) {
                            input_index[component_i] = 0;
                            input_index[component_i - 1] += 1;
                        }
                    }

                    if (input_index[1] >= shape_[1]) {
                        done = true;
                    }
                }
            }
        }
    }

    /// Get a const view of the data managed by this SimpleDataArray
    NDArray<double> view() const {
        return NDArray<double>(data_.data(), shape_);
    }

    /// Get a mutable view of the data managed by this SimpleDataArray
    NDArray<double> view() {
        return NDArray<double>(data_.data(), shape_);
    }

    /// Extract a reference to SimpleDataArray out of an `mts_array_t`.
    ///
    /// This function fails if the `mts_array_t` does not contain a
    /// SimpleDataArray.
    static SimpleDataArray& from_mts_array(mts_array_t& array) {
        mts_data_origin_t origin = 0;
        auto status = array.origin(array.ptr, &origin);
        if (status != MTS_SUCCESS) {
            throw Error("failed to get data origin");
        }

        std::array<char, 64> buffer = {0};
        status = mts_get_data_origin(origin, buffer.data(), buffer.size());
        if (status != MTS_SUCCESS || std::string(buffer.data()) != "metatensor::SimpleDataArray") {
            throw Error("this array is not a metatensor::SimpleDataArray");
        }

        auto* base = static_cast<DataArrayBase*>(array.ptr);
        return dynamic_cast<SimpleDataArray&>(*base);
    }

    /// Extract a const reference to SimpleDataArray out of an `mts_array_t`.
    ///
    /// This function fails if the `mts_array_t` does not contain a
    /// SimpleDataArray.
    static const SimpleDataArray& from_mts_array(const mts_array_t& array) {
        mts_data_origin_t origin = 0;
        auto status = array.origin(array.ptr, &origin);
        if (status != MTS_SUCCESS) {
            throw Error("failed to get data origin");
        }

        std::array<char, 64> buffer = {0};
        status = mts_get_data_origin(origin, buffer.data(), buffer.size());
        if (status != MTS_SUCCESS || std::string(buffer.data()) != "metatensor::SimpleDataArray") {
            throw Error("this array is not a metatensor::SimpleDataArray");
        }

        const auto* base = static_cast<const DataArrayBase*>(array.ptr);
        return dynamic_cast<const SimpleDataArray&>(*base);
    }

private:
    std::vector<uintptr_t> shape_;
    std::vector<double> data_;

    friend bool operator==(const SimpleDataArray& lhs, const SimpleDataArray& rhs);
};

/// Two SimpleDataArray compare as equal if they have the exact same shape and
/// data.
inline bool operator==(const SimpleDataArray& lhs, const SimpleDataArray& rhs) {
    return lhs.shape_ == rhs.shape_ && lhs.data_ == rhs.data_;
}

/// Two SimpleDataArray compare as equal if they have the exact same shape and
/// data.
inline bool operator!=(const SimpleDataArray& lhs, const SimpleDataArray& rhs) {
    return !(lhs == rhs);
}


/// An implementation of `DataArrayBase` containing no data.
///
/// This class only tracks it's shape, and can be used when only the metadata
/// of a `TensorBlock` is important, leaving the data unspecified.
class EmptyDataArray: public metatensor::DataArrayBase {
public:
    /// Create ae `EmptyDataArray` with the given `shape`
    EmptyDataArray(std::vector<uintptr_t> shape):
        shape_(std::move(shape)) {}

    ~EmptyDataArray() override = default;

    /// EmptyDataArray can be copy-constructed
    EmptyDataArray(const EmptyDataArray&) = default;
    /// EmptyDataArray can be copy-assigned
    EmptyDataArray& operator=(const EmptyDataArray&) = default;
    /// EmptyDataArray can be move-constructed
    EmptyDataArray(EmptyDataArray&&) noexcept = default;
    /// EmptyDataArray can be move-assigned
    EmptyDataArray& operator=(EmptyDataArray&&) noexcept = default;

    mts_data_origin_t origin() const override {
        mts_data_origin_t origin = 0;
        mts_register_data_origin("metatensor::EmptyDataArray", &origin);
        return origin;
    }

    double* data() & override {
        throw metatensor::Error("can not call `data` for an EmptyDataArray");
    }

    const std::vector<uintptr_t>& shape() const & override {
        return shape_;
    }

    void reshape(std::vector<uintptr_t> shape) override {
        if (details::product(shape_) != details::product(shape)) {
            throw metatensor::Error("invalid shape in reshape");
        }
        shape_ = std::move(shape);
    }

    void swap_axes(uintptr_t axis_1, uintptr_t axis_2) override {
        std::swap(shape_[axis_1], shape_[axis_2]);
    }

    std::unique_ptr<DataArrayBase> copy() const override {
        return std::unique_ptr<DataArrayBase>(new EmptyDataArray(*this));
    }

    std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape) const override {
        return std::unique_ptr<DataArrayBase>(new EmptyDataArray(std::move(shape)));
    }

    void move_samples_from(const DataArrayBase&, std::vector<mts_sample_mapping_t>, uintptr_t, uintptr_t) override {
        throw metatensor::Error("can not call `move_samples_from` for an EmptyDataArray");
    }

private:
    std::vector<uintptr_t> shape_;
};

namespace details {
    /// Default callback for data array creating in `TensorMap::load`, which
    /// will create a `SimpleDataArray`.
    inline mts_status_t default_create_array(
        const uintptr_t* shape_ptr,
        uintptr_t shape_count,
        mts_array_t* array
    ) {
        return details::catch_exceptions([](const uintptr_t* shape_ptr, uintptr_t shape_count, mts_array_t* array){
            auto shape = std::vector<size_t>();
            for (size_t i=0; i<shape_count; i++) {
                shape.push_back(static_cast<size_t>(shape_ptr[i]));
            }

            auto cxx_array = std::unique_ptr<DataArrayBase>(new SimpleDataArray(shape));
            *array = DataArrayBase::to_mts_array_t(std::move(cxx_array));

            return MTS_SUCCESS;
        }, shape_ptr, shape_count, array);
    }
}

/******************************************************************************/
/******************************************************************************/
/*                                                                            */
/*                           I/O functionalities                              */
/*                                                                            */
/******************************************************************************/
/******************************************************************************/

namespace io {
    /// Save a `TensorMap` to the file at `path`.
    ///
    /// If the file exists, it will be overwritten.
    ///
    /// `TensorMap` are serialized using numpy's `.npz` format, i.e. a ZIP file
    /// without compression (storage method is `STORED`), where each file is
    /// stored as a `.npy` array. See the C API documentation for more
    /// information on the format.
    void save(const std::string& path, const TensorMap& tensor);

    /// Save a `TensorMap` to an in-memory buffer.
    ///
    /// The `Buffer` template parameter can be set to any type that can be
    /// constructed from a pair of iterator over `std::vector<uint8_t>`.
    template <typename Buffer = std::vector<uint8_t>>
    Buffer save_buffer(const TensorMap& tensor);

    template<>
    std::vector<uint8_t> save_buffer<std::vector<uint8_t>>(const TensorMap& tensor);

    /// Save `Labels` to the file at `path`.
    ///
    /// If the file exists, it will be overwritten.
    void save(const std::string& path, const Labels& labels);

    /// Save `Labels` to an in-memory buffer.
    ///
    /// The `Buffer` template parameter can be set to any type that can be
    /// constructed from a pair of iterator over `std::vector<uint8_t>`.
    template <typename Buffer = std::vector<uint8_t>>
    Buffer save_buffer(const Labels& labels);

    template<>
    std::vector<uint8_t> save_buffer<std::vector<uint8_t>>(const Labels& labels);

    /*!
     * Load a previously saved `TensorMap` from the given path.
     *
     * \verbatim embed:rst:leading-asterisk
     *
     * ``create_array`` will be used to create new arrays when constructing the
     * blocks and gradients, the default version will create data using
     * :cpp:class:`SimpleDataArray`. See :c:func:`mts_create_array_callback_t`
     * for more information.
     *
     * \endverbatim
     *
     * `TensorMap` are serialized using numpy's `.npz` format, i.e. a ZIP file
     * without compression (storage method is `STORED`), where each file is
     * stored as a `.npy` array. See the C API documentation for more
     * information on the format.
     */
    TensorMap load(
        const std::string& path,
        mts_create_array_callback_t create_array = details::default_create_array
    );

    /*!
     * Load a previously saved `TensorMap` from the given `buffer`, containing
     * `buffer_count` elements.
     *
     * \verbatim embed:rst:leading-asterisk
     *
     * ``create_array`` will be used to create new arrays when constructing the
     * blocks and gradients, the default version will create data using
     * :cpp:class:`SimpleDataArray`. See :c:func:`mts_create_array_callback_t`
     * for more information.
     *
     * \endverbatim
     */
    TensorMap load_buffer(
        const uint8_t* buffer,
        size_t buffer_count,
        mts_create_array_callback_t create_array = details::default_create_array
    );


    /// Load a previously saved `TensorMap` from the given `buffer`.
    ///
    /// The `Buffer` template parameter would typically be a
    /// `std::vector<uint8_t>` or a `std::string`, but any container with
    /// contiguous data and an `item_type` with the same size as a `uint8_t` can
    /// work.
    template <typename Buffer>
    TensorMap load_buffer(
        const Buffer& buffer,
        mts_create_array_callback_t create_array = details::default_create_array
    );

    /// Load previously saved `Labels` from the given path.
    Labels load_labels(const std::string& path);

    /// Load previously saved `Labels` from the given `buffer`, containing
    /// `buffer_count` elements.
    Labels load_labels_buffer(const uint8_t* buffer, size_t buffer_count);

    /// Load a previously saved `Labels` from the given `buffer`.
    ///
    /// The `Buffer` template parameter would typically be a
    /// `std::vector<uint8_t>` or a `std::string`, but any container with
    /// contiguous data and an `item_type` with the same size as a `uint8_t` can
    /// work.
    template <typename Buffer>
    Labels load_labels_buffer(const Buffer& buffer);
}


/******************************************************************************/
/******************************************************************************/
/*                                                                            */
/*                                Labels                                      */
/*                                                                            */
/******************************************************************************/
/******************************************************************************/


/// It is possible to store some user-provided data inside `Labels`, and access
/// it later. This class is used to take ownership of the data and corresponding
/// delete function before giving the data to metatensor.
///
/// User data inside `Labels` is an advanced functionality, that most users
/// should not need to interact with.
class LabelsUserData {
public:
    /// Create `LabelsUserData` containing the given `data`.
    ///
    /// `deleter` will be called when the data is dropped, and should
    /// free the corresponding memory.
    LabelsUserData(void* data, void(*deleter)(void*)): data_(data), deleter_(deleter) {}

    ~LabelsUserData() {
        if (deleter_ !=  nullptr) {
            deleter_(data_);
        }
    }

    /// LabelsUserData is not copy-constructible
    LabelsUserData(const LabelsUserData& other) = delete;
    /// LabelsUserData can not be copy-assigned
    LabelsUserData& operator=(const LabelsUserData& other) = delete;

    /// LabelsUserData is move-constructible
    LabelsUserData(LabelsUserData&& other) noexcept: LabelsUserData(nullptr, nullptr) {
        *this = std::move(other);
    }

    /// LabelsUserData be move-assigned
    LabelsUserData& operator=(LabelsUserData&& other) noexcept {
        if (deleter_ !=  nullptr) {
            deleter_(data_);
        }

        data_ = other.data_;
        deleter_ = other.deleter_;

        other.data_ = nullptr;
        other.deleter_ = nullptr;

        return *this;
    }

private:
    friend class Labels;

    void* data_;
    void(*deleter_)(void*);
};


/// A set of labels used to carry metadata associated with a tensor map.
///
/// This is similar to an array of named tuples, but stored as a 2D array
/// of shape `(count, size)`, with a set of names associated with the columns of
/// this array (often called *dimensions*). Each row/entry in this array is
/// unique, and they are often (but not always) sorted in lexicographic order.
class Labels final {
public:
    /// Create a new set of Labels from the given `names` and `values`.
    ///
    /// Each entry in the values must contain `names.size()` elements.
    ///
    /// ```
    /// auto labels = Labels({"first", "second"}, {
    ///    {0, 1},
    ///    {1, 4},
    ///    {2, 1},
    ///    {2, 3},
    /// });
    /// ```
    Labels(
        const std::vector<std::string>& names,
        const std::vector<std::initializer_list<int32_t>>& values
    ): Labels(names, NDArray<int32_t>(values, names.size()), InternalConstructor{}) {}

    /// Create an empty set of Labels with the given names
    explicit Labels(const std::vector<std::string>& names):
        Labels(names, static_cast<const int32_t*>(nullptr), 0) {}

    /// Create labels with the given `names` and `values`. `values` must be an
    /// array with `count x names.size()` elements.
    Labels(const std::vector<std::string>& names, const int32_t* values, size_t count):
        Labels(details::labels_from_cxx(names, values, count)) {}

    ~Labels() {
        mts_labels_free(&labels_);
    }

    /// Labels is copy-constructible
    Labels(const Labels& other): Labels() {
        *this = other;
    }

    /// Labels can be copy-assigned
    Labels& operator=(const Labels& other) {
        mts_labels_free(&labels_);
        std::memset(&labels_, 0, sizeof(labels_));
        details::check_status(mts_labels_clone(other.labels_, &labels_));
        assert(this->labels_.internal_ptr_ != nullptr);

        this->values_ = NDArray<int32_t>(labels_.values, {labels_.count, labels_.size});

        this->names_.clear();
        for (size_t i=0; i<this->labels_.size; i++) {
            this->names_.push_back(this->labels_.names[i]);
        }

        return *this;
    }

    /// Labels is move-constructible
    Labels(Labels&& other) noexcept: Labels() {
        *this = std::move(other);
    }

    /// Labels can be move-assigned
    Labels& operator=(Labels&& other) noexcept {
        mts_labels_free(&labels_);
        this->labels_ = other.labels_;
        assert(this->labels_.internal_ptr_ != nullptr);
        std::memset(&other.labels_, 0, sizeof(other.labels_));

        this->values_ = std::move(other.values_);
        this->names_ = std::move(other.names_);

        return *this;
    }

    /// Get the names of the dimensions used in these `Labels`.
    const std::vector<const char*>& names() const {
        return names_;
    }

    /// Get the number of entries in this set of Labels.
    ///
    /// This is the same as `shape()[0]` for the corresponding values array
    size_t count() const {
        return labels_.count;
    }

    /// Get the number of dimensions in this set of Labels.
    ///
    /// This is the same as `shape()[1]` for the corresponding values array
    size_t size() const {
        return labels_.size;
    }

    /// Convert from this set of Labels to the C `mts_labels_t`
    mts_labels_t as_mts_labels_t() const {
        assert(labels_.internal_ptr_ != nullptr);
        return labels_;
    }

    /// Get the user data pointer registered with these `Labels`.
    ///
    /// If no user data have been registered, this function will return
    /// `nullptr`.
    void* user_data() & {
        assert(labels_.internal_ptr_ != nullptr);

        void* data = nullptr;
        details::check_status(mts_labels_user_data(labels_, &data));
        return data;
    }

    void* user_data() && = delete;

    /// Register some user data pointer with these `Labels`.
    ///
    /// Any existing user data will be released (by calling the provided
    /// `delete` function) before overwriting with the new data.
    void set_user_data(LabelsUserData user_data) {
        assert(labels_.internal_ptr_ != nullptr);

        details::check_status(mts_labels_set_user_data(
            labels_,
            user_data.data_,
            user_data.deleter_
        ));

        // the user data was moved inside `labels_`
        user_data.data_ = nullptr;
        user_data.deleter_ = nullptr;
    }

    /// Get the position of the `entry` in this set of Labels, or -1 if the
    /// entry is not part of these Labels.
    int64_t position(std::initializer_list<int32_t> entry) const {
        return this->position(entry.begin(), entry.size());
    }

    /// Variant of `Labels::position` taking a fixed-size array as input
    template<size_t N>
    int64_t position(const std::array<int32_t, N>& entry) const {
        return this->position(entry.data(), entry.size());
    }

    /// Variant of `Labels::position` taking a vector as input
    int64_t position(const std::vector<int32_t>& entry) const {
        return this->position(entry.data(), entry.size());
    }

    /// Variant of `Labels::position` taking a pointer and length as input
    int64_t position(const int32_t* entry, size_t length) const {
        assert(labels_.internal_ptr_ != nullptr);

        int64_t result = 0;
        details::check_status(mts_labels_position(labels_, entry, length, &result));
        return result;
    }

    /// Get the array of values for these Labels
    const NDArray<int32_t>& values() const & {
        return values_;
    }

    const NDArray<int32_t>& values() && = delete;

    /// Take the union of these `Labels` with `other`.
    ///
    /// If requested, this function can also give the positions in the union
    /// where each entry of the input `Labels` ended up.
    ///
    /// No user data pointer is registered with the output, even if the inputs
    /// have some.
    ///
    /// @param other the `Labels` we want to take the union with
    /// @param first_mapping if you want the mapping from the positions of
    ///        entries in `this` to the positions in the union, this should be
    ///        a pointer to an array containing `this->count()` elements, to be
    ///        filled by this function. Otherwise it should be a `nullptr`.
    /// @param first_mapping_count number of elements in `first_mapping`
    /// @param second_mapping if you want the mapping from the positions of
    ///        entries in `other` to the positions in the union, this should be
    ///        a pointer to an array containing `other.count()` elements, to be
    ///        filled by this function. Otherwise it should be a `nullptr`.
    /// @param second_mapping_count number of elements in `second_mapping`
    Labels set_union(
        const Labels& other,
        int64_t* first_mapping = nullptr,
        size_t first_mapping_count = 0,
        int64_t* second_mapping = nullptr,
        size_t second_mapping_count = 0
    ) const {
        mts_labels_t result;
        std::memset(&result, 0, sizeof(result));

        details::check_status(mts_labels_union(
            labels_,
            other.labels_,
            &result,
            first_mapping,
            first_mapping_count,
            second_mapping,
            second_mapping_count
        ));

        return Labels(result);
    }

    /// Take the union of these `Labels` with `other`.
    ///
    /// If requested, this function can also give the positions in the
    /// union where each entry of the input `Labels` ended up.
    ///
    /// No user data pointer is registered with the output, even if the inputs
    /// have some.
    ///
    /// @param other the `Labels` we want to take the union with
    /// @param first_mapping if you want the mapping from the positions of
    ///        entries in `this` to the positions in the union, this should be
    ///        a vector containing `this->count()` elements, to be filled by
    ///        this function. Otherwise it should be an empty vector.
    /// @param second_mapping if you want the mapping from the positions of
    ///        entries in `other` to the positions in the union, this should be
    ///        a vector containing `other.count()` elements, to be filled by
    ///        this function. Otherwise it should be an empty vector.
    Labels set_union(
        const Labels& other,
        std::vector<int64_t>& first_mapping,
        std::vector<int64_t>& second_mapping
    ) const {
        auto* first_mapping_ptr = first_mapping.data();
        auto first_mapping_count = first_mapping.size();
        if (first_mapping_count == 0) {
            first_mapping_ptr = nullptr;
        }

        auto* second_mapping_ptr = second_mapping.data();
        auto second_mapping_count = second_mapping.size();
        if (second_mapping_count == 0) {
            second_mapping_ptr = nullptr;
        }

        return this->set_union(
            other,
            first_mapping_ptr,
            first_mapping_count,
            second_mapping_ptr,
            second_mapping_count
        );
    }

    /// Take the intersection of these `Labels` with `other`.
    ///
    /// If requested, this function can also give the positions in the
    /// intersection where each entry of the input `Labels` ended up.
    ///
    /// No user data pointer is registered with the output, even if the inputs
    /// have some.
    ///
    /// @param other the `Labels` we want to take the intersection with
    /// @param first_mapping if you want the mapping from the positions of
    ///        entries in `this` to the positions in the intersection, this
    ///        should be a pointer to an array containing `this->count()`
    ///        elements, to be filled by this function. Otherwise it should be a
    ///        `nullptr`. If an entry in `this` is not used in the intersection,
    ///        the mapping will be set to -1.
    /// @param first_mapping_count number of elements in `first_mapping`
    /// @param second_mapping if you want the mapping from the positions of
    ///        entries in `other` to the positions in the intersection, this
    ///        should be a pointer to an array containing `other.count()`
    ///        elements, to be filled by this function. Otherwise it should be a
    ///        `nullptr`. If an entry in `other` is not used in the
    ///        intersection, the mapping will be set to -1.
    /// @param second_mapping_count number of elements in `second_mapping`
    Labels set_intersection(
        const Labels& other,
        int64_t* first_mapping = nullptr,
        size_t first_mapping_count = 0,
        int64_t* second_mapping = nullptr,
        size_t second_mapping_count = 0
    ) const {
        mts_labels_t result;
        std::memset(&result, 0, sizeof(result));

        details::check_status(mts_labels_intersection(
            labels_,
            other.labels_,
            &result,
            first_mapping,
            first_mapping_count,
            second_mapping,
            second_mapping_count
        ));

        return Labels(result);
    }

    /// Take the intersection of this `Labels` with `other`.
    ///
    /// If requested, this function can also give the positions in the
    /// intersection where each entry of the input `Labels` ended up.
    ///
    /// No user data pointer is registered with the output, even if the inputs
    /// have some.
    ///
    /// @param other the `Labels` we want to take the intersection with
    /// @param first_mapping if you want the mapping from the positions of
    ///        entries in `this` to the positions in the intersection, this
    ///        should be a vector containing `this->count()` elements, to be
    ///        filled by this function. Otherwise it should be an empty vector.
    ///        If an entry in `this` is not used in the intersection, the
    ///        mapping will be set to -1.
    /// @param second_mapping if you want the mapping from the positions of
    ///        entries in `other` to the positions in the intersection, this
    ///        should be a vector containing `other.count()` elements, to be
    ///        filled by this function. Otherwise it should be an empty vector.
    ///        If an entry in `other` is not used in the intersection, the
    ///        mapping will be set to -1.
    Labels set_intersection(
        const Labels& other,
        std::vector<int64_t>& first_mapping,
        std::vector<int64_t>& second_mapping
    ) const {
        auto* first_mapping_ptr = first_mapping.data();
        auto first_mapping_count = first_mapping.size();
        if (first_mapping_count == 0) {
            first_mapping_ptr = nullptr;
        }

        auto* second_mapping_ptr = second_mapping.data();
        auto second_mapping_count = second_mapping.size();
        if (second_mapping_count == 0) {
            second_mapping_ptr = nullptr;
        }

        return this->set_intersection(
            other,
            first_mapping_ptr,
            first_mapping_count,
            second_mapping_ptr,
            second_mapping_count
        );
    }

    /// Select entries in these `Labels` that match the `selection`.
    ///
    /// The selection's names must be a subset of the names of these labels.
    ///
    /// All entries in these `Labels` that match one of the entry in the
    /// `selection` for all the selection's dimension will be picked. Any entry
    /// in the `selection` but not in these `Labels` will be ignored.
    ///
    /// @param selection definition of the selection criteria. Multiple entries
    ///        are interpreted as a logical `or` operation.
    /// @param selected on input, a pointer to an array with space for
    ///        `*selected_count` entries. On output, the first `*selected_count`
    ///        values will contain the index in `labels` of selected entries.
    /// @param selected_count on input, size of the `selected` array. On output,
    ///        this will contain the number of selected entries.
    void select(const Labels& selection, int64_t* selected, size_t *selected_count) const {
        details::check_status(mts_labels_select(
            labels_,
            selection.labels_,
            selected,
            selected_count
        ));
    }

    /// Select entries in these `Labels` that match the `selection`.
    ///
    /// This function does the same thing as the one above, but allocates and
    /// return the list of selected indexes in a `std::vector`
    std::vector<int64_t> select(const Labels& selection) const {
        auto selected_count = this->count();
        auto selected = std::vector<int64_t>(selected_count, -1);

        this->select(selection, selected.data(), &selected_count);

        selected.resize(selected_count);
        return selected;
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load previously saved ``Labels`` from the given path.
     *
     * This is identical to :cpp:func:`metatensor::io::load_labels`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    static Labels load(const std::string& path) {
        return metatensor::io::load_labels(path);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load previously saved ``Labels`` from a in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::load_labels_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    static Labels load_buffer(const uint8_t* buffer, size_t buffer_count) {
        return metatensor::io::load_labels_buffer(buffer, buffer_count);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load previously saved ``Labels`` from a in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::load_labels_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    template <typename Buffer>
    static Labels load_buffer(const Buffer& buffer) {
        return metatensor::io::load_labels_buffer<Buffer>(buffer);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Save ``Labels`` to the given path.
     *
     * This is identical to :cpp:func:`metatensor::io::save`, and provided as a
     * convenience API.
     *
     * \endverbatim
     */
    void save(const std::string& path) const {
        return metatensor::io::save(path, *this);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Save ``Labels`` to an in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::save_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    std::vector<uint8_t> save_buffer() const {
        return metatensor::io::save_buffer(*this);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Save ``Labels`` to an in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::save_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    template <typename Buffer>
    Buffer save_buffer() const {
        return metatensor::io::save_buffer<Buffer>(*this);
    }

private:
    explicit Labels(): values_(static_cast<const int32_t*>(nullptr), {0, 0})
    {
        std::memset(&labels_, 0, sizeof(labels_));
    }

    explicit Labels(mts_labels_t labels):
        values_(labels.values, {labels.count, labels.size}),
        labels_(labels)
    {
        assert(labels_.internal_ptr_ != nullptr);

        for (size_t i=0; i<labels_.size; i++) {
            names_.push_back(labels_.names[i]);
        }
    }

    // the constructor below is ambiguous with the public constructor taking
    // `std::initializer_list`, so we use a private dummy struct argument to
    // remove the ambiguity.
    struct InternalConstructor {};
    Labels(const std::vector<std::string>& names, const NDArray<int32_t>& values, InternalConstructor):
        Labels(names, values.data(), values.shape()[0]) {}

    friend Labels details::labels_from_cxx(const std::vector<std::string>& names, const int32_t* values, size_t count);
    friend Labels io::load_labels(const std::string &path);
    friend Labels io::load_labels_buffer(const uint8_t* buffer, size_t buffer_count);
    friend class TensorMap;
    friend class TensorBlock;

    friend class metatensor_torch::LabelsHolder;

    std::vector<const char*> names_;
    NDArray<int32_t> values_;
    mts_labels_t labels_;

    friend bool operator==(const Labels& lhs, const Labels& rhs);
};

namespace details {
    inline metatensor::Labels labels_from_cxx(
        const std::vector<std::string>& names,
        const int32_t* values,
        size_t count
    ) {
        mts_labels_t labels;
        std::memset(&labels, 0, sizeof(labels));

        auto c_names = std::vector<const char*>();
        for (const auto& name: names) {
            c_names.push_back(name.c_str());
        }

        labels.names = c_names.data();
        labels.size = c_names.size();
        labels.count = count;
        labels.values = values;

        details::check_status(mts_labels_create(&labels));

        return metatensor::Labels(labels);
    }
}


/// Two Labels compare equal only if they have the same names and values in the
/// same order.
inline bool operator==(const Labels& lhs, const Labels& rhs) {
    if (lhs.names_.size() != rhs.names_.size()) {
        return false;
    }

    for (size_t i=0; i<lhs.names_.size(); i++) {
        if (std::strcmp(lhs.names_[i], rhs.names_[i]) != 0) {
            return false;
        }
    }

    return lhs.values() == rhs.values();
}

/// Two Labels compare equal only if they have the same names and values in the
/// same order.
inline bool operator!=(const Labels& lhs, const Labels& rhs) {
    return !(lhs == rhs);
}


/******************************************************************************/
/******************************************************************************/
/*                                                                            */
/*                             TensorBlock                                    */
/*                                                                            */
/******************************************************************************/
/******************************************************************************/


/// Basic building block for a tensor map.
///
/// A single block contains a n-dimensional `mts_array_t` (or `DataArrayBase`),
/// and n sets of `Labels` (one for each dimension). The first dimension is the
/// *samples* dimension, the last dimension is the *properties* dimension. Any
/// intermediate dimension is called a *component* dimension.
///
/// Samples should be used to describe *what* we are representing, while
/// properties should contain information about *how* we are representing it.
/// Finally, components should be used to describe vectorial or tensorial
/// components of the data.
///
/// A block can also contain gradients of the values with respect to a variety
/// of parameters. In this case, each gradient has a separate set of samples,
/// and possibly components but share the same property labels as the values.
class TensorBlock final {
public:
    /// Create a new TensorBlock containing the given `values` array.
    ///
    /// The different dimensions of the values are described by `samples`,
    /// `components` and `properties` `Labels`
    TensorBlock(
        std::unique_ptr<DataArrayBase> values,
        const Labels& samples,
        const std::vector<Labels>& components,
        const Labels& properties
    ):
        block_(nullptr),
        is_view_(false)
    {
        auto c_components = std::vector<mts_labels_t>();
        for (const auto& component: components) {
            c_components.push_back(component.as_mts_labels_t());
        }
        block_ = mts_block(
            DataArrayBase::to_mts_array_t(std::move(values)),
            samples.as_mts_labels_t(),
            c_components.data(),
            c_components.size(),
            properties.as_mts_labels_t()
        );

        details::check_pointer(block_);
    }

    ~TensorBlock() {
        if (!is_view_) {
            mts_block_free(block_);
        }
    }

    /// TensorBlock can NOT be copy constructed, use TensorBlock::clone instead
    TensorBlock(const TensorBlock&) = delete;

    /// TensorBlock can NOT be copy assigned, use TensorBlock::clone instead
    TensorBlock& operator=(const TensorBlock& other) = delete;

    /// TensorBlock can be move constructed
    TensorBlock(TensorBlock&& other) noexcept : TensorBlock() {
        *this = std::move(other);
    }

    /// TensorBlock can be moved assigned
    TensorBlock& operator=(TensorBlock&& other) noexcept {
        if (!is_view_) {
            mts_block_free(block_);
        }

        this->block_ = other.block_;
        this->is_view_ = other.is_view_;
        other.block_ = nullptr;
        other.is_view_ = true;

        return *this;
    }

    /// Make a copy of this `TensorBlock`, including all the data contained inside
    TensorBlock clone() const {
        auto copy = TensorBlock();
        copy.is_view_ = false;
        copy.block_ = mts_block_copy(this->block_);
        details::check_pointer(copy.block_);
        return copy;
    }

    /// Get a copy of the metadata in this block (i.e. samples, components, and
    /// properties), ignoring the data itself.
    ///
    /// The resulting block values will be an `EmptyDataArray` instance, which
    /// does not contain any data.
    TensorBlock clone_metadata_only() const {
        auto block = TensorBlock(
            std::unique_ptr<EmptyDataArray>(new EmptyDataArray(this->values_shape())),
            this->samples(),
            this->components(),
            this->properties()
        );

        for (const auto& parameter: this->gradients_list()) {
            auto gradient = this->gradient(parameter);
            block.add_gradient(parameter, gradient.clone_metadata_only());
        }

        return block;
    }

    /// Get a view in the values in this block
    NDArray<double> values() & {
        auto array = this->mts_array();
        double* data = nullptr;
        details::check_status(array.data(array.ptr, &data));

        return NDArray<double>(data, this->values_shape());
    }

    NDArray<double> values() && = delete;

    /// Access the sample `Labels` for this block.
    ///
    /// The entries in these labels describe the first dimension of the
    /// `values()` array.
    Labels samples() const {
        return this->labels(0);
    }

    /// Access the component `Labels` for this block.
    ///
    /// The entries in these labels describe intermediate dimensions of the
    /// `values()` array.
    std::vector<Labels> components() const {
        auto shape = this->values_shape();

        auto result = std::vector<Labels>();
        for (size_t i=1; i<shape.size() - 1; i++) {
            result.emplace_back(this->labels(i));
        }

        return result;
    }

    /// Access the property `Labels` for this block.
    ///
    /// The entries in these labels describe the last dimension of the
    /// `values()` array. The properties are guaranteed to be the same for
    /// a block and all of its gradients.
    Labels properties() const {
        auto shape = this->values_shape();
        return this->labels(shape.size() - 1);
    }

    /// Add a set of gradients with respect to `parameters` in this block.
    ///
    /// @param parameter add gradients with respect to this `parameter` (e.g.
    ///                 `"positions"`, `"cell"`, ...)
    /// @param gradient a `TensorBlock` whose values contain the gradients with
    ///                 respect to the `parameter`. The labels of the gradient
    ///                 `TensorBlock` should be organized as follows: its
    ///                 `samples` must contain `"sample"` as the first label,
    ///                 which establishes a correspondence with the `samples` of
    ///                 the original `TensorBlock`; its components must contain
    ///                 at least the same components as the original
    ///                 `TensorBlock`, with any additional component coming
    ///                 before those; its properties must match those of the
    ///                 original `TensorBlock`.
    void add_gradient(const std::string& parameter, TensorBlock gradient) {
        if (is_view_) {
            throw Error(
                "can not call TensorBlock::add_gradient on this block since "
                "it is a view inside a TensorMap"
            );
        }

        details::check_status(mts_block_add_gradient(
            block_,
            parameter.c_str(),
            gradient.release()
        ));
    }

    /// Get a list of all gradients defined in this block.
    std::vector<std::string> gradients_list() const {
        const char*const * parameters = nullptr;
        uintptr_t count = 0;
        details::check_status(mts_block_gradients_list(
            block_,
            &parameters,
            &count
        ));

        auto result = std::vector<std::string>();
        for (uint64_t i=0; i<count; i++) {
            result.emplace_back(parameters[i]);
        }

        return result;
    }

    /// Get the gradient in this block with respect to the given `parameter`.
    /// The gradient is returned as a TensorBlock itself.
    ///
    /// @param parameter check for gradients with respect to this `parameter`
    ///                  (e.g. `"positions"`, `"cell"`, ...)
    TensorBlock gradient(const std::string& parameter) const {
        mts_block_t* gradient_block = nullptr;
        details::check_status(
            mts_block_gradient(block_, parameter.c_str(), &gradient_block)
        );
        details::check_pointer(gradient_block);
        return TensorBlock::unsafe_view_from_ptr(gradient_block);
    }

    /// Get the `mts_block_t` pointer corresponding to this block.
    ///
    /// The block pointer is still managed by the current `TensorBlock`
    mts_block_t* as_mts_block_t() & {
        if (is_view_) {
            throw Error(
                "can not call non-const TensorBlock::as_mts_block_t on this "
                "block since it is a view inside a TensorMap"
            );
        }
        return block_;
    }

    /// const version of `as_mts_block_t`
    const mts_block_t* as_mts_block_t() const & {
        return block_;
    }

    const mts_block_t* as_mts_block_t() && = delete;

    /// Create a new TensorBlock taking ownership of a raw `mts_block_t` pointer.
    static TensorBlock unsafe_from_ptr(mts_block_t* ptr) {
        auto block = TensorBlock();
        block.block_ = ptr;
        block.is_view_ = false;
        return block;
    }

    /// Create a new TensorBlock which is a view corresponding to a raw
    /// `mts_block_t` pointer.
    static TensorBlock unsafe_view_from_ptr(mts_block_t* ptr) {
        auto block = TensorBlock();
        block.block_ = ptr;
        block.is_view_ = true;
        return block;
    }

    /// Get a raw `mts_array_t` corresponding to the values in this block.
    mts_array_t mts_array() {
        mts_array_t array;
        std::memset(&array, 0, sizeof(array));

        details::check_status(
            mts_block_data(block_, &array)
        );
        return array;
    }

    /// Get the labels in this block associated with the given `axis`.
    Labels labels(uintptr_t axis) const {
        mts_labels_t labels;
        std::memset(&labels, 0, sizeof(labels));
        details::check_status(mts_block_labels(
            block_, axis, &labels
        ));

        return Labels(labels);
    }

    /// Get the shape of the value array for this block
    std::vector<uintptr_t> values_shape() const {
        auto array = this->const_mts_array();

        const uintptr_t* shape = nullptr;
        uintptr_t shape_count = 0;
        details::check_status(array.shape(array.ptr, &shape, &shape_count));
        assert(shape_count >= 2);

        return {shape, shape + shape_count};
    }

private:
    /// Constructor of a TensorBlock not associated with anything
    explicit TensorBlock(): block_(nullptr), is_view_(true) {}

    /// Get the `mts_array_t` for this block.
    ///
    /// The returned `mts_array_t` should only be used in a const context
    mts_array_t const_mts_array() const {
        mts_array_t array;
        std::memset(&array, 0, sizeof(array));

        details::check_status(
            mts_block_data(block_, &array)
        );
        return array;
    }

    /// Release the `mts_block_t` pointer corresponding to this `TensorBlock`.
    ///
    /// The block pointer is **no longer** managed by the current `TensorBlock`,
    /// and should manually be freed when no longer required.
    mts_block_t* release() {
         if (is_view_) {
            throw Error(
                "can not call TensorBlock::release on this "
                "block since it is a view inside a TensorMap"
            );
        }
        auto* ptr = block_;
        block_ = nullptr;
        is_view_ = false;
        return ptr;
    }

    friend class TensorMap;
    friend class metatensor_torch::TensorBlockHolder;

    mts_block_t* block_;
    bool is_view_;
};


/******************************************************************************/
/******************************************************************************/
/*                                                                            */
/*                               TensorMap                                    */
/*                                                                            */
/******************************************************************************/
/******************************************************************************/

/// A TensorMap is the main user-facing class of this library, and can store any
/// kind of data used in atomistic machine learning.
///
/// A tensor map contains a list of `TensorBlock`, each one associated with a
/// key. Users can access the blocks either one by one with the `block_by_id()`
/// function.
///
/// A tensor map provides functions to move some of these keys to the samples or
/// properties labels of the blocks, moving from a sparse representation of the
/// data to a dense one.
class TensorMap final {
public:
    /// Create a new TensorMap with the given `keys` and `blocks`
    TensorMap(Labels keys, std::vector<TensorBlock> blocks) {
        auto c_blocks = std::vector<mts_block_t*>();
        for (auto& block: blocks) {
            // We will move the data inside the new map, let's release the
            // pointers out of the TensorBlock now
            c_blocks.push_back(block.release());
        }

        tensor_ = mts_tensormap(
            keys.as_mts_labels_t(),
            c_blocks.data(),
            c_blocks.size()
        );

        details::check_pointer(tensor_);
    }

    ~TensorMap() {
        mts_tensormap_free(tensor_);
    }

    /// TensorMap can NOT be copy constructed, use TensorMap::clone instead
    TensorMap(const TensorMap&) = delete;
    /// TensorMap can not be copy assigned, use TensorMap::clone instead
    TensorMap& operator=(const TensorMap&) = delete;

    /// TensorMap can be move constructed
    TensorMap(TensorMap&& other) noexcept : TensorMap(nullptr) {
        *this = std::move(other);
    }

    /// TensorMap can be move assigned
    TensorMap& operator=(TensorMap&& other) noexcept {
        mts_tensormap_free(tensor_);

        this->tensor_ = other.tensor_;
        other.tensor_ = nullptr;

        return *this;
    }

    /// Make a copy of this `TensorMap`, including all the data contained inside
    TensorMap clone() const {
        auto* copy = mts_tensormap_copy(this->tensor_);
        details::check_pointer(copy);
        return TensorMap(copy);
    }

    /// Get a copy of the metadata in this `TensorMap` (i.e. keys, samples,
    /// components, and properties), ignoring the data itself.
    ///
    /// The resulting blocks values will be an `EmptyDataArray` instance, which
    /// does not contain any data.
    TensorMap clone_metadata_only() const {
        auto n_blocks = this->keys().count();

        auto blocks = std::vector<TensorBlock>();
        blocks.reserve(n_blocks);
        for (uintptr_t i=0; i<n_blocks; i++) {
            mts_block_t* block_ptr = nullptr;
            details::check_status(mts_tensormap_block_by_id(tensor_, &block_ptr, i));
            details::check_pointer(block_ptr);
            auto block = TensorBlock::unsafe_view_from_ptr(block_ptr);

            blocks.push_back(block.clone_metadata_only());
        }

        return TensorMap(this->keys(), std::move(blocks));
    }

    /// Get the set of keys labeling the blocks in this tensor map
    Labels keys() const {
        mts_labels_t keys;
        std::memset(&keys, 0, sizeof(keys));

        details::check_status(mts_tensormap_keys(tensor_, &keys));
        return Labels(keys);
    }

    /// Get a (possibly empty) list of block indexes matching the `selection`
    std::vector<uintptr_t> blocks_matching(const Labels& selection) const {
        auto matching = std::vector<uintptr_t>(this->keys().count());
        uintptr_t count = matching.size();

        details::check_status(mts_tensormap_blocks_matching(
            tensor_,
            matching.data(),
            &count,
            selection.as_mts_labels_t()
        ));

        assert(count <= matching.size());
        matching.resize(count);
        return matching;
    }

    /// Get a block inside this TensorMap by it's index/the index of the
    /// corresponding key.
    ///
    /// The returned `TensorBlock` is a view inside memory owned by this
    /// `TensorMap`, and is only valid as long as the `TensorMap` is kept alive.
    TensorBlock block_by_id(uintptr_t index) & {
        mts_block_t* block = nullptr;
        details::check_status(mts_tensormap_block_by_id(tensor_, &block, index));
        details::check_pointer(block);

        return TensorBlock::unsafe_view_from_ptr(block);
    }

    TensorBlock block_by_id(uintptr_t index) && = delete;

    /// Merge blocks with the same value for selected keys dimensions along the
    /// property axis.
    ///
    /// The dimensions (names) of `keys_to_move` will be moved from the keys to
    /// the property labels, and blocks with the same remaining keys dimensions
    /// will be merged together along the property axis.
    ///
    /// If `keys_to_move` does not contains any entries (i.e.
    /// `keys_to_move.count() == 0`), then the new property labels will contain
    /// entries corresponding to the merged blocks only. For example, merging a
    /// block with key `a=0` and properties `p=1, 2` with a block with key `a=2`
    /// and properties `p=1, 3` will produce a block with properties `a, p = (0,
    /// 1), (0, 2), (2, 1), (2, 3)`.
    ///
    /// If `keys_to_move` contains entries, then the property labels must be the
    /// same for all the merged blocks. In that case, the merged property labels
    /// will contains each of the entries of `keys_to_move` and then the current
    /// property labels. For example, using `a=2, 3` in `keys_to_move`, and
    /// blocks with properties `p=1, 2` will result in `a, p = (2, 1), (2, 2),
    /// (3, 1), (3, 2)`.
    ///
    /// The new sample labels will contains all of the merged blocks sample
    /// labels. The order of the samples is controlled by `sort_samples`. If
    /// `sort_samples` is true, samples are re-ordered to keep them
    /// lexicographically sorted. Otherwise they are kept in the order in which
    /// they appear in the blocks.
    ///
    /// @param keys_to_move description of the keys to move
    /// @param sort_samples whether to sort the merged samples or keep them in
    ///                     the order in which they appear in the original blocks
    TensorMap keys_to_properties(const Labels& keys_to_move, bool sort_samples = true) const {
        auto* ptr = mts_tensormap_keys_to_properties(
            tensor_,
            keys_to_move.as_mts_labels_t(),
            sort_samples
        );

        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    /// This function calls `keys_to_properties` with an empty set of `Labels`
    /// with the dimensions defined in `keys_to_move`
    TensorMap keys_to_properties(const std::vector<std::string>& keys_to_move, bool sort_samples = true) const {
        return keys_to_properties(Labels(keys_to_move), sort_samples);
    }

    /// This function calls `keys_to_properties` with an empty set of `Labels`
    /// with a single dimension: `key_to_move`
    TensorMap keys_to_properties(const std::string& key_to_move, bool sort_samples = true) const {
        return keys_to_properties(std::vector<std::string>{key_to_move}, sort_samples);
    }

    /// Merge blocks with the same value for selected keys dimensions along the
    /// samples axis.
    ///
    /// The dimensions (names) of `keys_to_move` will be moved from the keys to
    /// the sample labels, and blocks with the same remaining keys dimensions
    /// will be merged together along the sample axis.
    ///
    /// If `keys_to_move` must be an empty set of `Labels`
    /// (`keys_to_move.count() == 0`). The new sample labels will contain
    /// entries corresponding to the merged blocks' keys.
    ///
    /// The order of the samples is controlled by `sort_samples`. If
    /// `sort_samples` is true, samples are re-ordered to keep them
    /// lexicographically sorted. Otherwise they are kept in the order in which
    /// they appear in the blocks.
    ///
    /// This function is only implemented if all merged block have the same
    /// property labels.
    ///
    /// @param keys_to_move description of the keys to move
    /// @param sort_samples whether to sort the merged samples or keep them in
    ///                     the order in which they appear in the original blocks
    TensorMap keys_to_samples(const Labels& keys_to_move, bool sort_samples = true) const {
        auto* ptr = mts_tensormap_keys_to_samples(
            tensor_,
            keys_to_move.as_mts_labels_t(),
            sort_samples
        );

        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    /// This function calls `keys_to_samples` with an empty set of `Labels`
    /// with the dimensions defined in `keys_to_move`
    TensorMap keys_to_samples(const std::vector<std::string>& keys_to_move, bool sort_samples = true) const {
        return keys_to_samples(Labels(keys_to_move), sort_samples);
    }

    /// This function calls `keys_to_samples` with an empty set of `Labels`
    /// with a single dimension: `key_to_move`
    TensorMap keys_to_samples(const std::string& key_to_move, bool sort_samples = true) const {
        return keys_to_samples(std::vector<std::string>{key_to_move}, sort_samples);
    }

    /// Move the given `dimensions` from the component labels to the property
    /// labels for each block.
    ///
    /// @param dimensions name of the component dimensions to move to the
    ///                  properties
    TensorMap components_to_properties(const std::vector<std::string>& dimensions) const {
        auto c_dimensions = std::vector<const char*>();
        for (const auto& v: dimensions) {
            c_dimensions.push_back(v.c_str());
        }

        auto* ptr = mts_tensormap_components_to_properties(
            tensor_,
            c_dimensions.data(),
            c_dimensions.size()
        );
        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    /// Call `components_to_properties` with a single dimension
    TensorMap components_to_properties(const std::string& dimension) const {
        const char* c_str = dimension.c_str();
        auto* ptr = mts_tensormap_components_to_properties(
            tensor_,
            &c_str,
            1
        );
        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load a previously saved ``TensorMap`` from the given path.
     *
     * This is identical to :cpp:func:`metatensor::io::load`, and provided as a
     * convenience API.
     *
     * \endverbatim
     */
    static TensorMap load(
        const std::string& path,
        mts_create_array_callback_t create_array = details::default_create_array
    ) {
        return metatensor::io::load(path, create_array);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load a previously saved ``TensorMap`` from a in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::load_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    static TensorMap load_buffer(
        const uint8_t* buffer,
        size_t buffer_count,
        mts_create_array_callback_t create_array = details::default_create_array
    ) {
        return metatensor::io::load_buffer(buffer, buffer_count, create_array);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Load a previously saved ``TensorMap`` from a in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::load_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    template <typename Buffer>
    static TensorMap load_buffer(
        const Buffer& buffer,
        mts_create_array_callback_t create_array = details::default_create_array
    ) {
        return metatensor::io::load_buffer<Buffer>(buffer, create_array);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Save a ``TensorMap`` to the given path.
     *
     * This is identical to :cpp:func:`metatensor::io::save`, and provided as a
     * convenience API.
     *
     * \endverbatim
     */
    void save(const std::string& path) const {
        return metatensor::io::save(path, *this);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Save a ``TensorMap`` to an in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::save_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    std::vector<uint8_t> save_buffer() const {
        return metatensor::io::save_buffer(*this);
    }

    /*!
     * \verbatim embed:rst:leading-asterisk
     *
     * Save a ``TensorMap`` to an in-memory buffer.
     *
     * This is identical to :cpp:func:`metatensor::io::save_buffer`, and
     * provided as a convenience API.
     *
     * \endverbatim
     */
    template <typename Buffer>
    Buffer save_buffer() const {
        return metatensor::io::save_buffer<Buffer>(*this);
    }

    /// Get the `mts_tensormap_t` pointer corresponding to this `TensorMap`.
    ///
    /// The tensor map pointer is still managed by the current `TensorMap`
    mts_tensormap_t* as_mts_tensormap_t() & {
        return tensor_;
    }

    /// Get the const `mts_tensormap_t` pointer corresponding to this `TensorMap`.
    ///
    /// The tensor map pointer is still managed by the current `TensorMap`
    const mts_tensormap_t* as_mts_tensormap_t() const & {
        return tensor_;
    }

    mts_tensormap_t* as_mts_tensormap_t() && = delete;

    /// Create a C++ TensorMap from a C `mts_tensormap_t` pointer. The C++
    /// tensor map takes ownership of the C pointer.
    explicit TensorMap(mts_tensormap_t* tensor): tensor_(tensor) {}

private:
    mts_tensormap_t* tensor_;
};


/******************************************************************************/
/******************************************************************************/
/*                                                                            */
/*                   I/O functionalities implementation                       */
/*                                                                            */
/******************************************************************************/
/******************************************************************************/


namespace io {
    inline void save(const std::string& path, const TensorMap& tensor) {
        details::check_status(mts_tensormap_save(path.c_str(), tensor.as_mts_tensormap_t()));
    }

    template <typename Buffer>
    Buffer save_buffer(const TensorMap& tensor) {
        auto buffer = metatensor::io::save_buffer<std::vector<uint8_t>>(tensor);
        return Buffer(buffer.begin(), buffer.end());
    }

    template<>
    inline std::vector<uint8_t> save_buffer<std::vector<uint8_t>>(const TensorMap& tensor) {
        std::vector<uint8_t> buffer;

        auto* ptr = buffer.data();
        auto size = buffer.size();

        auto realloc = [](void* user_data, uint8_t*, uintptr_t new_size) {
            auto* buffer = reinterpret_cast<std::vector<uint8_t>*>(user_data);
            buffer->resize(new_size, '\0');
            return buffer->data();
        };

        details::check_status(mts_tensormap_save_buffer(
            &ptr,
            &size,
            &buffer,
            realloc,
            tensor.as_mts_tensormap_t()
        ));

        buffer.resize(size, '\0');

        return buffer;
    }

    inline void save(const std::string& path, const Labels& labels) {
        details::check_status(mts_labels_save(path.c_str(), labels.as_mts_labels_t()));
    }

    template <typename Buffer>
    Buffer save_buffer(const Labels& labels) {
        auto buffer = metatensor::io::save_buffer<std::vector<uint8_t>>(labels);
        return Buffer(buffer.begin(), buffer.end());
    }

    template<>
    inline std::vector<uint8_t> save_buffer<std::vector<uint8_t>>(const Labels& labels) {
        std::vector<uint8_t> buffer;

        auto* ptr = buffer.data();
        auto size = buffer.size();

        auto realloc = [](void* user_data, uint8_t*, uintptr_t new_size) {
            auto* buffer = reinterpret_cast<std::vector<uint8_t>*>(user_data);
            buffer->resize(new_size, '\0');
            return buffer->data();
        };

        details::check_status(mts_labels_save_buffer(
            &ptr,
            &size,
            &buffer,
            realloc,
            labels.as_mts_labels_t()
        ));

        buffer.resize(size, '\0');

        return buffer;
    }

    inline TensorMap load(
        const std::string& path,
        mts_create_array_callback_t create_array
    ) {
        auto* ptr = mts_tensormap_load(path.c_str(), create_array);
        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    inline TensorMap load_buffer(
        const uint8_t* buffer,
        size_t buffer_count,
        mts_create_array_callback_t create_array
    ) {
        auto* ptr = mts_tensormap_load_buffer(buffer, buffer_count, create_array);
        details::check_pointer(ptr);
        return TensorMap(ptr);
    }

    template <typename Buffer>
    TensorMap load_buffer(
        const Buffer& buffer,
        mts_create_array_callback_t create_array
    ) {
        static_assert(
            sizeof(typename Buffer::value_type) == sizeof(uint8_t),
            "`Buffer` must be a container of uint8_t or equivalent"
        );

        return metatensor::io::load_buffer(
            reinterpret_cast<const uint8_t*>(buffer.data()),
            buffer.size(),
            create_array
        );
    }

    inline Labels load_labels(const std::string& path) {
        mts_labels_t labels;
        std::memset(&labels, 0, sizeof(labels));

        details::check_status(mts_labels_load(
            path.c_str(), &labels
        ));

        return Labels(labels);
    }

    inline Labels load_labels_buffer(const uint8_t* buffer, size_t buffer_count) {
        mts_labels_t labels;
        std::memset(&labels, 0, sizeof(labels));

        details::check_status(mts_labels_load_buffer(
            buffer, buffer_count, &labels
        ));

        return Labels(labels);
    }

    template <typename Buffer>
    Labels load_labels_buffer(const Buffer& buffer) {
        static_assert(
            sizeof(typename Buffer::value_type) == sizeof(uint8_t),
            "`Buffer` must be a container of uint8_t or equivalent"
        );

        return metatensor::io::load_labels_buffer(
            reinterpret_cast<const uint8_t*>(buffer.data()),
            buffer.size()
        );
    }
}

}

#endif /* METATENSOR_HPP */

#include <cstring>
#include <cctype>

#include <array>
#include <algorithm>
#include <sstream>

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <metatensor.hpp>

#include "metatensor/torch/atomistic/model.hpp"
#include "metatensor/torch/misc.hpp"

#include "../internal/shared_libraries.hpp"

using namespace metatensor_torch;

static void read_vector_string_json(
    std::vector<std::string>& output,
    const nlohmann::json& array,
    const std::string& context
) {
    if (!array.is_array()) {
        throw std::runtime_error(context + " must be an array");
    }
    for (const auto& value: array) {
        if (!value.is_string()) {
            throw std::runtime_error(context + " must be an array of string");
        }
        output.emplace_back(value);
    }
}

template<typename T>
static void read_vector_int_json(
    std::vector<T>& output,
    const nlohmann::json& array,
    const std::string& context
) {
    if (!array.is_array()) {
        throw std::runtime_error(context + " must be an array");
    }
    for (const auto& value: array) {
        if (!value.is_number_integer()) {
            throw std::runtime_error(context + " must be an array of integers");
        }
        output.emplace_back(value);
    }
}

/******************************************************************************/

void ModelOutputHolder::set_quantity(std::string quantity) {
    if (valid_quantity(quantity)) {
        validate_unit(quantity, unit_);
    }

    this->quantity_ = std::move(quantity);
}

void ModelOutputHolder::set_unit(std::string unit) {
    validate_unit(quantity_, unit);
    this->unit_ = std::move(unit);
}

static nlohmann::json model_output_to_json(const ModelOutputHolder& self) {
    nlohmann::json result;

    result["class"] = "ModelOutput";
    result["quantity"] = self.quantity();
    result["unit"] = self.unit();
    result["per_atom"] = self.per_atom;
    result["explicit_gradients"] = self.explicit_gradients;

    return result;
}

std::string ModelOutputHolder::to_json() const {
    return model_output_to_json(*this).dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

static ModelOutput model_output_from_json(const nlohmann::json& data) {
    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelOutput, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for ModelOutput, did not find it");
    }

    if (data["class"] != "ModelOutput") {
        throw std::runtime_error("'class' in JSON for ModelOutput must be 'ModelOutput'");
    }

    auto result = torch::make_intrusive<ModelOutputHolder>();
    if (data.contains("quantity")) {
        if (!data["quantity"].is_string()) {
            throw std::runtime_error("'quantity' in JSON for ModelOutput must be a string");
        }
        result->set_quantity(data["quantity"]);
    }

    if (data.contains("unit")) {
        if (!data["unit"].is_string()) {
            throw std::runtime_error("'unit' in JSON for ModelOutput must be a string");
        }
        result->set_unit(data["unit"]);
    }

    if (data.contains("per_atom")) {
        if (!data["per_atom"].is_boolean()) {
            throw std::runtime_error("'per_atom' in JSON for ModelOutput must be a boolean");
        }
        result->per_atom = data["per_atom"];
    }

    if (data.contains("explicit_gradients")) {
        read_vector_string_json(
            result->explicit_gradients,
            data["explicit_gradients"],
            "'explicit_gradients' in JSON for ModelOutput"
        );
    }

    return result;
}

ModelOutput ModelOutputHolder::from_json(std::string_view json) {
    auto data = nlohmann::json::parse(json);
    return model_output_from_json(data);
}

/******************************************************************************/

void ModelCapabilitiesHolder::set_length_unit(std::string unit) {
    validate_unit("length", unit);
    this->length_unit_ = std::move(unit);
}

void ModelCapabilitiesHolder::set_dtype(std::string dtype) {
    if (dtype == "float32" || dtype == "float64") {
        dtype_ = std::move(dtype);
    } else {
        C10_THROW_ERROR(ValueError,
            "`dtype` can be one of ['float32', 'float64'], got '" + dtype + "'"
        );
    }
}

double ModelCapabilitiesHolder::engine_interaction_range(const std::string& engine_length_unit) const {
    return interaction_range * unit_conversion_factor("length", length_unit_, engine_length_unit);
}

std::string ModelCapabilitiesHolder::to_json() const {
    nlohmann::json result;

    result["class"] = "ModelCapabilities";

    auto outputs = nlohmann::json::object();
    for (const auto& it: this->outputs) {
        outputs[it.key()] = model_output_to_json(*it.value());
    }
    result["outputs"] = outputs;
    result["atomic_types"] = this->atomic_types;

    // Store interaction_range using it's binary representation to ensure
    // perfect round-tripping of the data
    static_assert(sizeof(double) == sizeof(int64_t));
    int64_t int_interaction_range = 0;
    std::memcpy(&int_interaction_range, &this->interaction_range, sizeof(double));
    result["interaction_range"] = int_interaction_range;

    result["length_unit"] = this->length_unit();
    result["supported_devices"] = this->supported_devices;
    result["dtype"] = this->dtype();

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

ModelCapabilities ModelCapabilitiesHolder::from_json(std::string_view json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelCapabilities, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for ModelCapabilities, did not find it");
    }

    if (data["class"] != "ModelCapabilities") {
        throw std::runtime_error("'class' in JSON for ModelCapabilities must be 'ModelCapabilities'");
    }

    auto result = torch::make_intrusive<ModelCapabilitiesHolder>();
    if (data.contains("outputs")) {
        if (!data["outputs"].is_object()) {
            throw std::runtime_error("'outputs' in JSON for ModelCapabilities must be an object");
        }

        for (const auto& output: data["outputs"].items()) {
            result->outputs.insert(output.key(), model_output_from_json(output.value()));
        }
    }

    if (data.contains("atomic_types")) {
        read_vector_int_json(
            result->atomic_types,
            data["atomic_types"],
            "'atomic_types' in JSON for ModelCapabilities"
        );
    }

    if (data.contains("interaction_range")) {
        if (!data["interaction_range"].is_number_integer()) {
            throw std::runtime_error("'interaction_range' in JSON for ModelCapabilities must be a number");
        }

        auto int_interaction_range = data["interaction_range"].get<int64_t>();
        double interaction_range = 0;
        std::memcpy(&interaction_range, &int_interaction_range, sizeof(double));

        result->interaction_range = interaction_range;
    }

    if (data.contains("length_unit")) {
        if (!data["length_unit"].is_string()) {
            throw std::runtime_error("'length_unit' in JSON for ModelCapabilities must be a string");
        }
        result->set_length_unit(data["length_unit"]);
    }

    if (data.contains("supported_devices")) {
        read_vector_string_json(
            result->supported_devices,
            data["supported_devices"],
            "'supported_devices' in JSON for ModelCapabilities"
        );
    }

    if (data.contains("dtype")) {
        if (!data["dtype"].is_string()) {
            throw std::runtime_error("'dtype' in JSON for ModelCapabilities must be a string");
        }
        result->set_dtype(data["dtype"]);
    }

    return result;
}

/******************************************************************************/

static void check_selected_atoms(const torch::optional<TorchLabels>& selected_atoms) {
    if (selected_atoms) {
        if (selected_atoms.value()->names() != std::vector<std::string>{"system", "atom"}) {
            std::ostringstream oss;
            oss << '[';
            for (const auto& name: selected_atoms.value()->names()) {
                oss << '\'' << name << "', ";
            }
            oss << ']';

            C10_THROW_ERROR(ValueError,
                "invalid `selected_atoms` names: expected ['system', 'atom'], "
                "got " + oss.str()
            );
        }
    }
}

void ModelEvaluationOptionsHolder::set_length_unit(std::string unit) {
    validate_unit("length", unit);
    this->length_unit_ = std::move(unit);
}

ModelEvaluationOptionsHolder::ModelEvaluationOptionsHolder(
    std::string length_unit_,
    torch::Dict<std::string, ModelOutput> outputs_,
    torch::optional<TorchLabels> selected_atoms
):
    outputs(outputs_),
    selected_atoms_(std::move(selected_atoms))
{
    this->set_length_unit(std::move(length_unit_));
    check_selected_atoms(selected_atoms_);
}


void ModelEvaluationOptionsHolder::set_selected_atoms(torch::optional<TorchLabels> selected_atoms) {
    check_selected_atoms(selected_atoms);
    selected_atoms_ = std::move(selected_atoms);
}


std::string ModelEvaluationOptionsHolder::to_json() const {
    nlohmann::json result;

    result["class"] = "ModelEvaluationOptions";
    result["length_unit"] = this->length_unit();

    if (this->selected_atoms_) {
        const auto& selected_atoms = this->selected_atoms_.value();

        auto selected_json = nlohmann::json::object();
        selected_json["names"] = selected_atoms->names();
        auto values = selected_atoms->values().to(torch::kCPU).contiguous();
        auto size = static_cast<size_t>(selected_atoms->size() * selected_atoms->count());
        selected_json["values"] = std::vector<int32_t>(
            values.data_ptr<int32_t>(),
            values.data_ptr<int32_t>() + size
        );

        result["selected_atoms"] = std::move(selected_json);
    } else {
        result["selected_atoms"] = nlohmann::json();
    }

    auto outputs = nlohmann::json::object();
    for (const auto& it: this->outputs) {
        outputs[it.key()] = model_output_to_json(*it.value());
    }
    result["outputs"] = outputs;

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}

ModelEvaluationOptions ModelEvaluationOptionsHolder::from_json(std::string_view json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelEvaluationOptions, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for ModelEvaluationOptions, did not find it");
    }

    if (data["class"] != "ModelEvaluationOptions") {
        throw std::runtime_error("'class' in JSON for ModelEvaluationOptions must be 'ModelEvaluationOptions'");
    }

    auto result = torch::make_intrusive<ModelEvaluationOptionsHolder>();
    if (data.contains("length_unit")) {
        if (!data["length_unit"].is_string()) {
            throw std::runtime_error("'length_unit' in JSON for ModelEvaluationOptions must be a string");
        }
        result->set_length_unit(data["length_unit"]);
    }

    if (data.contains("selected_atoms")) {
        if (data["selected_atoms"].is_null()) {
            // nothing to do
        } else {
            if (!data["selected_atoms"].is_object()) {
                throw std::runtime_error("'selected_atoms' in JSON for ModelEvaluationOptions must be an object");
            }

            if (!data["selected_atoms"].contains("names")) {
                throw std::runtime_error("'selected_atoms.names' in JSON for ModelEvaluationOptions must be an array");
            }

            auto names = std::vector<std::string>();
            read_vector_string_json(
                names,
                data["selected_atoms"]["names"],
                "'selected_atoms.names' in JSON for ModelEvaluationOptions"
            );

            if (!data["selected_atoms"].contains("values")) {
                throw std::runtime_error("'selected_atoms.values' in JSON for ModelEvaluationOptions must be an array");
            }

            auto values = std::vector<int32_t>();
            read_vector_int_json(
                values,
                data["selected_atoms"]["values"],
                "'selected_atoms.values' in JSON for ModelEvaluationOptions"
            );
            assert(values.size() % 2 == 0);

            result->set_selected_atoms(torch::make_intrusive<LabelsHolder>(
                std::move(names),
                torch::tensor(values).reshape({-1, 2})
            ));
        }
    }

    if (data.contains("outputs")) {
        if (!data["outputs"].is_object()) {
            throw std::runtime_error("'outputs' in JSON for ModelEvaluationOptions must be an object");
        }

        for (const auto& output: data["outputs"].items()) {
            result->outputs.insert(output.key(), model_output_from_json(output.value()));
        }
    }

    return result;
}

/******************************************************************************/

void ModelMetadataHolder::validate() const {
    for (const auto& author: this->authors) {
        if (author.empty()) {
            C10_THROW_ERROR(ValueError, "author can not be empty string in ModelMetadata");
        }
    }

    for (const auto& item: this->references) {
        if (item.key() != "implementation" && item.key() != "architecture" && item.key() != "model") {
            C10_THROW_ERROR(ValueError, "unknown key in references: " + item.key());
        }

        for (const auto& ref: item.value()) {
            if (ref.empty()) {
                C10_THROW_ERROR(ValueError,
                    "reference can not be empty string (in '" + item.key() + "' section)"
                );
            }
        }
    }
}

std::string ModelMetadataHolder::to_json() const {
    nlohmann::json result;

    result["class"] = "ModelMetadata";
    result["name"] = this->name;
    result["description"] = this->description;
    result["authors"] = this->authors;

    auto references = nlohmann::json::object();
    for (const auto& it: this->references) {
        references[it.key()] = it.value();
    }
    result["references"] = references;

    return result.dump(/*indent*/4, /*indent_char*/' ', /*ensure_ascii*/ true);
}


ModelMetadata ModelMetadataHolder::from_json(std::string_view json) {
    auto data = nlohmann::json::parse(json);

    if (!data.is_object()) {
        throw std::runtime_error("invalid JSON data for ModelMetadata, expected an object");
    }

    if (!data.contains("class") || !data["class"].is_string()) {
        throw std::runtime_error("expected 'class' in JSON for ModelMetadata, did not find it");
    }

    if (data["class"] != "ModelMetadata") {
        throw std::runtime_error("'class' in JSON for ModelMetadata must be 'ModelMetadata'");
    }

    auto result = torch::make_intrusive<ModelMetadataHolder>();
    if (data.contains("name")) {
        if (!data["name"].is_string()) {
            throw std::runtime_error("'name' in JSON for ModelMetadata must be a string");
        }
        result->name = data["name"];
    }

    if (data.contains("description")) {
        if (!data["description"].is_string()) {
            throw std::runtime_error("'description' in JSON for ModelMetadata must be a string");
        }
        result->description = data["description"];
    }

    if (data.contains("authors")) {
        read_vector_string_json(
            result->authors,
            data["authors"],
            "'authors' in JSON for ModelMetadata"
        );
    }

    if (data.contains("references")) {
        if (!data["references"].is_object()) {
            throw std::runtime_error("'references' in JSON for ModelMetadata must be an object");
        }

        const auto& references = data["references"];
        if (references.contains("implementation")) {
            auto implementation = std::vector<std::string>();
            read_vector_string_json(
                implementation,
                data["references"]["implementation"],
                "'references.implementation' in JSON for ModelMetadata"
            );
            result->references.insert("implementation", std::move(implementation));
        }

        if (references.contains("architecture")) {
            auto architecture = std::vector<std::string>();
            read_vector_string_json(
                architecture,
                data["references"]["architecture"],
                "'references.architecture' in JSON for ModelMetadata"
            );
            result->references.insert("architecture", std::move(architecture));
        }

        if (references.contains("model")) {
            auto model = std::vector<std::string>();
            read_vector_string_json(
                model,
                data["references"]["model"],
                "'references.model' in JSON for ModelMetadata"
            );
            result->references.insert("model", std::move(model));
        }
    }

    result->validate();

    return result;
}


// replace end of line characters and tabs with a single space
static std::string normalize_whitespace(std::string_view data) {
    auto string = std::string(data);
    for (auto& c : string) {
        if (c == '\n' || c == '\r' || c == '\t') {
            c = ' ';
        }
    }
    return string;
}

static void wrap_80_chars(std::ostringstream& oss, std::string_view data, std::string_view indent) {
    auto string = normalize_whitespace(data);
    auto view = std::string_view(string);

    auto line_length = 80 - indent.length();
    assert(line_length > 50);
    auto first_line = true;
    while (true) {
        if (view.length() <= line_length) {
            // last line
            if (!first_line) {
                oss << indent;
            }
            oss << view;
            break;
        } else {
            // backtrack to find the end of a word
            bool word_found = false;
            for (size_t i=(line_length - 1); i>0; i--) {
                if (view[i] == ' ') {
                    word_found = true;
                    // print the current line
                    if (!first_line) {
                        oss << indent;
                    }
                    oss << view.substr(0, i) << '\n';
                    // Update the view and start with the next line. We can
                    // start the substr at i + 1 since we started the loop at
                    // line_length - 1
                    view = view.substr(i + 1);
                    first_line = false;
                    break;
                }
            }

            if (!word_found) {
                // this is only hit if a single word takes a full line.
                throw std::runtime_error("some words are too long to be wrapped, make them shorter");
            }
        }
    }
}


std::string ModelMetadataHolder::print() const {
    this->validate();
    std::ostringstream oss;

    if (this->name.empty()) {
        oss << "This is an unamed model\n";
        oss << "=======================\n";
    } else {
        oss << "This is the " << this->name << " model\n";
        oss << "============" << std::string(this->name.length(), '=') << "======\n";
    }

    if (!this->description.empty()) {
        oss << "\n";
        wrap_80_chars(oss, this->description, "");
        oss << "\n";
    }

    if (!this->authors.empty()) {
        oss << "\nModel authors\n-------------\n\n";
        for (const auto& author: authors) {
            oss << "- ";
            wrap_80_chars(oss, author, "  ");
            oss << "\n";
        }
    }

    std::ostringstream references_oss;
    if (this->references.contains("model")) {
        references_oss << "- about this specific model:\n";
        for (const auto& reference: this->references.at("model")) {
            references_oss << "  * ";
            wrap_80_chars(references_oss, reference, "    ");
            references_oss << "\n";
        }
    }

    if (this->references.contains("architecture")) {
        references_oss << "- about the architecture of this model:\n";
        for (const auto& reference: this->references.at("architecture")) {
            references_oss << "  * ";
            wrap_80_chars(references_oss, reference, "    ");
            references_oss << "\n";
        }
    }

    if (this->references.contains("implementation") && !this->references.at("implementation").empty()) {
        references_oss << "- about the implementation of this model:\n";
        for (const auto& reference: this->references.at("implementation")) {
            references_oss << "  * ";
            wrap_80_chars(references_oss, reference, "    ");
            references_oss << "\n";
        }
    }

    auto references = references_oss.str();
    if (!references.empty()) {
        oss << "\nModel references\n----------------\n\n";
        oss << "Please cite the following references when using this model:\n";
        oss << references;
    }

    return oss.str();
}

/******************************************************************************/

struct Version {
    Version(std::string version): string(std::move(version)) {
        size_t pos = 0;
        try {
            this->major = std::stoi(this->string, &pos);
        } catch (const std::invalid_argument&) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }

        if (this->string[pos] != '.' || this->string.size() == pos) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }

        auto minor_version = this->string.substr(pos + 1);
        try {
            this->minor = std::stoi(minor_version, &pos);
        } catch (const std::invalid_argument&) {
            C10_THROW_ERROR(ValueError, "invalid version number: " + this->string);
        }
    }

    /// Check if two version are compatible. `same_minor` indicates whether two
    /// versions should have the same major AND minor number to be considered
    /// compatible.
    bool is_compatible(const Version& other, bool same_minor = false) const {
        if (this->major != other.major) {
            return false;
        }

        if (this->major == 0) {
            same_minor = true;
        }

        if (same_minor && this->minor != other.minor) {
            return false;
        }

        return true;
    }

    std::string string;
    int major = 0;
    int minor = 0;
};

struct Extension {
    std::string name;
    std::string path;
};

void from_json(const nlohmann::json& json, Extension& extension) {
    json.at("name").get_to(extension.name);
    json.at("path").get_to(extension.path);
}


/// Convert (ptr, len) tuple to a string
static std::string record_to_string(std::tuple<at::DataPtr, size_t> data) {
    return std::string(
        static_cast<char*>(std::get<0>(data).get()),
        std::get<1>(data)
    );
}

void metatensor_torch::check_atomistic_model(std::string path) {
    auto reader = caffe2::serialize::PyTorchStreamReader(path);

    if (!reader.hasRecord("extra/metatensor-version")) {
        C10_THROW_ERROR(ValueError,
            "file at '" + path + "' does not contain a metatensor atomistic model"
        );
    }

    auto recorded_mts_version = Version(record_to_string(
        reader.getRecord("extra/metatensor-version")
    ));
    auto current_mts_version = Version(metatensor_torch::version());

    if (!current_mts_version.is_compatible(recorded_mts_version)) {
        TORCH_WARN(
            "Current metatensor version (", current_mts_version.string, ") ",
            "is not compatible with the version (", recorded_mts_version.string,
            ") used to export the model at '", path, "'; proceed at your own risk."
        );
    }

    auto recorded_torch_version = Version(record_to_string(
        reader.getRecord("extra/torch-version")
    ));
    auto current_torch_version = Version(TORCH_VERSION);
    if (!current_torch_version.is_compatible(recorded_torch_version, true)) {
        TORCH_WARN(
            "Current torch version (", current_torch_version.string, ") ",
            "is not compatible with the version (", recorded_torch_version.string,
            ") used to export the model at '", path, "'; proceed at your own risk."
        );
    }

    // Check that the extensions loaded while the model was exported are also
    // loaded now. Since the model can be exported from a different machine, or
    // the extensions might change how they organize code, we only try to do
    // fuzzy matching on the file name, and warn if we can not find a match.
    std::vector<Extension> extensions = nlohmann::json::parse(record_to_string(
        reader.getRecord("extra/extensions")
    ));

    auto loaded_libraries = metatensor_torch::details::get_loaded_libraries();

    for (const auto& extension: extensions) {
        auto found = false;
        for (const auto& library: loaded_libraries) {
            if (library.find(extension.name) != std::string::npos) {
                found = true;
                break;
            }
        }

        if (!found) {
            TORCH_WARN(
                "The model at '", path, "' was exported with extension '",
                extension.name, "' loaded (from '", extension.path, "'), ",
                "but it does not seem to be currently loaded; proceed at your own risk."
            );
        }
    }
}

torch::jit::Module metatensor_torch::load_atomistic_model(
    std::string path,
    c10::optional<c10::Device> device
) {
    check_atomistic_model(path);
    return torch::jit::load(path, device);
}

/******************************************************************************/
/******************************************************************************/

/// normalize a string to lower case without any spaces
static std::string normalize_string(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char c){ return std::tolower(c); }
    );

    // remove all spaces in e.g. `kcal /   mol`.
    auto new_end = std::remove_if(value.begin(), value.end(),
        [](unsigned char c){ return std::isspace(c); }
    );
    value.erase(new_end, value.end());
    return value;
}

/// Information for unit conversion for this physical quantity
struct Quantity {
    /// the quantity name
    std::string name;

    /// baseline unit for this quantity
    std::string baseline;
    /// set of conversion from the key to the baseline unit
    std::unordered_map<std::string, double> conversions;

    std::string normalize_unit(const std::string& original_unit) {
        if (original_unit.empty()) {
            return original_unit;
        }

        std::string unit = normalize_string(original_unit);
        if (this->conversions.find(unit) == this->conversions.end()) {
            auto valid_units = std::vector<std::string>();
            for (const auto& it: this->conversions) {
                valid_units.emplace_back(it.first);
            }

            C10_THROW_ERROR(ValueError,
                "unknown unit '" + original_unit + "' for " + name + ", "
                "only [" + torch::str(valid_units) + "] are supported"
            );
        }

        return unit;
    }

    double conversion(const std::string& from_unit, const std::string& to_unit) {
        auto from = this->normalize_unit(from_unit);
        auto to = this->normalize_unit(to_unit);

        if (from.empty() || to.empty()) {
            return 1.0;
        }

        return this->conversions.at(to) / this->conversions.at(from);
    }
};

static std::unordered_map<std::string, Quantity> KNOWN_QUANTITIES = {
    {"length", Quantity{/* name */ "length", /* baseline */ "angstrom", {
        {"angstrom", 1.0},
        {"bohr", 1.8897261258369282},
        {"nanometer", 0.1},
        {"nm", 0.1},
    }}},
    {"energy", Quantity{/* name */ "energy", /* baseline */ "ev", {
        {"ev", 1.0},
        {"mev", 1000.0},
        {"hartree", 0.03674932247495664},
        {"kcal/mol", 23.060548012069496},
        {"kj/mol", 96.48533288249877},
    }}},
};

bool metatensor_torch::valid_quantity(const std::string& quantity) {
    if (quantity.empty()) {
        return false;
    }

    if (KNOWN_QUANTITIES.find(quantity) == KNOWN_QUANTITIES.end()) {
        auto valid_quantities = std::vector<std::string>();
        for (const auto& it: KNOWN_QUANTITIES) {
            valid_quantities.emplace_back(it.first);
        }

        TORCH_WARN(
            "unknown quantity '", quantity, "', only [",
            torch::str(valid_quantities), "] are supported"
        );
        return false;
    } else {
        return true;
    }
}


void metatensor_torch::validate_unit(const std::string& quantity, const std::string& unit) {
    if (quantity.empty() || unit.empty()) {
        return;
    }

    if (valid_quantity(quantity)) {
        KNOWN_QUANTITIES.at(quantity).normalize_unit(unit);
    }
}

double metatensor_torch::unit_conversion_factor(
    const std::string& quantity,
    const std::string& from_unit,
    const std::string& to_unit
) {
    if (valid_quantity(quantity)) {
        return KNOWN_QUANTITIES.at(quantity).conversion(from_unit, to_unit);
    } else {
        return 1.0;
    }
}

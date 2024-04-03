#include <torch/torch.h>

#include <metatensor/torch.hpp>
#include <metatensor/torch/atomistic.hpp>
using namespace metatensor_torch;

#include <catch.hpp>
using namespace Catch::Matchers;

TEST_CASE("Models metadata") {
    SECTION("NeighborsListOptions") {
        // save to JSON
        auto options = torch::make_intrusive<NeighborsListOptionsHolder>(3.5426, true);
        const auto* expected = R"({
    "class": "NeighborsListOptions",
    "cutoff": 4615159644819978768,
    "full_list": true,
    "length_unit": ""
})";
        CHECK(options->to_json() == expected);

        // load from JSON
        std::string json = R"({
    "cutoff": 4615159644819978768,
    "full_list": false,
    "class": "NeighborsListOptions"
})";
        options = NeighborsListOptionsHolder::from_json(json);
        CHECK(options->cutoff() == 3.5426);
        CHECK(options->full_list() == false);

        CHECK_THROWS_WITH(
            NeighborsListOptionsHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for NeighborsListOptions, did not find it")
        );
        CHECK_THROWS_WITH(
            NeighborsListOptionsHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for NeighborsListOptions must be 'NeighborsListOptions'")
        );

        CHECK_THROWS_WITH(options->set_length_unit("unknown"),
            StartsWith("unknown unit 'unknown' for length")
        );
    }

    SECTION("ModelOutput") {
        // save to JSON
        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->set_quantity("energy");
        output->set_unit("kJ / mol");
        output->per_atom = false;
        output->explicit_gradients = {"baz", "not.this-one_"};

        const auto* expected = R"({
    "class": "ModelOutput",
    "explicit_gradients": [
        "baz",
        "not.this-one_"
    ],
    "per_atom": false,
    "quantity": "energy",
    "unit": "kJ / mol"
})";
        CHECK(output->to_json() == expected);

        // load from JSON
        std::string json = R"({
    "class": "ModelOutput",
    "quantity": "length",
    "explicit_gradients": []
})";
        output = ModelOutputHolder::from_json(json);
        CHECK(output->quantity() == "length");
        CHECK(output->unit().empty());
        CHECK(output->per_atom == false);
        CHECK(output->explicit_gradients.empty());

        CHECK_THROWS_WITH(
            ModelOutputHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelOutput, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelOutputHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelOutput must be 'ModelOutput'")
        );

        CHECK_THROWS_WITH(output->set_unit("unknown"),
            StartsWith("unknown unit 'unknown' for length")
        );

    #if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 0

        struct WarningHandler: public torch::WarningHandler {
            virtual ~WarningHandler() override = default;
            void process(const torch::Warning& warning) override {
                CHECK(warning.msg() == "unknown quantity 'unknown', only [energy length] are supported");
            }
        };

        auto* old_handler = torch::WarningUtils::get_warning_handler();
        auto check_expected_warning = WarningHandler();
        torch::WarningUtils::set_warning_handler(&check_expected_warning);

        output->set_quantity("unknown"),

        torch::WarningUtils::set_warning_handler(old_handler);
    #endif
    }

    SECTION("ModelEvaluationOptions") {
        // save to JSON
        auto options = torch::make_intrusive<ModelEvaluationOptionsHolder>();
        options->set_length_unit("nm");

        options->outputs.insert("output_1", torch::make_intrusive<ModelOutputHolder>());

        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->per_atom = true;
        output->set_quantity("something");
        output->set_unit("something");
        options->outputs.insert("output_2", output);

        const auto* expected = R"({
    "class": "ModelEvaluationOptions",
    "length_unit": "nm",
    "outputs": {
        "output_1": {
            "class": "ModelOutput",
            "explicit_gradients": [],
            "per_atom": false,
            "quantity": "",
            "unit": ""
        },
        "output_2": {
            "class": "ModelOutput",
            "explicit_gradients": [],
            "per_atom": true,
            "quantity": "something",
            "unit": "something"
        }
    },
    "selected_atoms": null
})";
        CHECK(options->to_json() == expected);


        // load from JSON
        std::string json =R"({
    "length_unit": "Angstrom",
    "outputs": {
        "foo": {
            "explicit_gradients": ["test"],
            "class": "ModelOutput"
        }
    },
    "selected_atoms": {
        "names": ["system", "atom"],
        "values": [0, 1, 4, 5]
    },
    "class": "ModelEvaluationOptions"
})";

        options = ModelEvaluationOptionsHolder::from_json(json);
        CHECK(options->length_unit() == "Angstrom");
        auto expected_selection = LabelsHolder::create(
            {"system", "atom"},
            {{0, 1}, {4, 5}}
        );
        CHECK(*options->get_selected_atoms().value() == *expected_selection);

        output = options->outputs.at("foo");
        CHECK(output->quantity().empty());
        CHECK(output->unit().empty());
        CHECK(output->per_atom == false);
        CHECK(output->explicit_gradients == std::vector<std::string>{"test"});

        CHECK_THROWS_WITH(
            ModelEvaluationOptionsHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelEvaluationOptions, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelEvaluationOptionsHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelEvaluationOptions must be 'ModelEvaluationOptions'")
        );

        CHECK_THROWS_WITH(options->set_length_unit("unknown"),
            StartsWith("unknown unit 'unknown' for length")
        );
    }

    SECTION("ModelCapabilities") {
        // save to JSON
        auto capabilities = torch::make_intrusive<ModelCapabilitiesHolder>();
        capabilities->set_length_unit("nm");
        capabilities->interaction_range = 1.4;
        capabilities->atomic_types = {1, 2, -43};
        capabilities->set_dtype("float32");
        capabilities->supported_devices = {"cuda", "xla", "cpu"};

        auto output = torch::make_intrusive<ModelOutputHolder>();
        output->per_atom = true;
        output->set_quantity("length");
        output->explicit_gradients.emplace_back("µ-λ");
        capabilities->outputs.insert("bar", output);

        const auto* expected = R"({
    "atomic_types": [
        1,
        2,
        -43
    ],
    "class": "ModelCapabilities",
    "dtype": "float32",
    "interaction_range": 4608983858650965606,
    "length_unit": "nm",
    "outputs": {
        "bar": {
            "class": "ModelOutput",
            "explicit_gradients": [
                "\u00b5-\u03bb"
            ],
            "per_atom": true,
            "quantity": "length",
            "unit": ""
        }
    },
    "supported_devices": [
        "cuda",
        "xla",
        "cpu"
    ]
})";
        CHECK(capabilities->to_json() == expected);


        // load from JSON
        std::string json =R"({
    "length_unit": "nm",
    "outputs": {
        "foo": {
            "explicit_gradients": ["\u00b5-test"],
            "class": "ModelOutput"
        }
    },
    "atomic_types": [
        1,
        -2
    ],
    "class": "ModelCapabilities"
})";

        capabilities = ModelCapabilitiesHolder::from_json(json);
        CHECK(capabilities->length_unit() == "nm");
        CHECK(capabilities->dtype().empty());
        CHECK(capabilities->atomic_types == std::vector<int64_t>{1, -2});

        output = capabilities->outputs.at("foo");
        CHECK(output->quantity().empty());
        CHECK(output->unit().empty());
        CHECK(output->per_atom == false);
        CHECK(output->explicit_gradients == std::vector<std::string>{"µ-test"});

        CHECK_THROWS_WITH(
            ModelCapabilitiesHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelCapabilities, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelCapabilitiesHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelCapabilities must be 'ModelCapabilities'")
        );

        CHECK_THROWS_WITH(capabilities->set_length_unit("unknown"),
            StartsWith("unknown unit 'unknown' for length")
        );
    }

    SECTION("ModelMetadata") {
        // save to JSON
        auto metadata = torch::make_intrusive<ModelMetadataHolder>();
        metadata->name = "some name";
        metadata->description = "describing it";
        metadata->authors = {"John Doe", "Napoleon"};
        metadata->references.insert("model", std::vector<std::string>{"some-ref"});
        metadata->references.insert("architecture", std::vector<std::string>{"ref-2", "ref-3"});

        const auto* expected = R"({
    "authors": [
        "John Doe",
        "Napoleon"
    ],
    "class": "ModelMetadata",
    "description": "describing it",
    "name": "some name",
    "references": {
        "architecture": [
            "ref-2",
            "ref-3"
        ],
        "model": [
            "some-ref"
        ]
    }
})";
        CHECK(metadata->to_json() == expected);


        // load from JSON
        std::string json =R"({
    "class": "ModelMetadata",
    "name": "foo",
    "description": "test",
    "authors": ["me", "myself"],
    "references": {
        "implementation": ["torch-power!"],
        "model": ["took a while to train"]
    }
})";

        metadata = ModelMetadataHolder::from_json(json);
        CHECK(metadata->name == "foo");
        CHECK(metadata->description == "test");
        CHECK(metadata->authors == std::vector<std::string>{"me", "myself"});
        CHECK(metadata->references.at("implementation") == std::vector<std::string>{"torch-power!"});
        CHECK(metadata->references.at("model") == std::vector<std::string>{"took a while to train"});

        CHECK_THROWS_WITH(
            ModelMetadataHolder::from_json("{}"),
            StartsWith("expected 'class' in JSON for ModelMetadata, did not find it")
        );
        CHECK_THROWS_WITH(
            ModelMetadataHolder::from_json("{\"class\": \"nope\"}"),
            StartsWith("'class' in JSON for ModelMetadata must be 'ModelMetadata'")
        );

        // printing
        metadata = torch::make_intrusive<ModelMetadataHolder>();
        metadata->name = "name";
        metadata->description = R"(Lorem ipsum dolor sit amet, consectetur
adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna
aliqua. Ut enim ad minim veniam, quis nostrud exercitation.)";
        metadata->authors = {"Short author", "Some extremely long author that will take more than one line in the printed output"};
        metadata->references.insert("model", std::vector<std::string>{
            "a very long reference that will take more than one line in the printed output"
        });
        metadata->references.insert("architecture", std::vector<std::string>{"ref-2", "ref-3"});

        expected = R"(This is the name model
======================

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis
nostrud exercitation.

Model authors
-------------

- Short author
- Some extremely long author that will take more than one line in the printed
  output

Model references
----------------

Please cite the following references when using this model:
- about this specific model:
  * a very long reference that will take more than one line in the printed
    output
- about the architecture of this model:
  * ref-2
  * ref-3
)";

        CHECK(metadata->print() == expected);
    }
}

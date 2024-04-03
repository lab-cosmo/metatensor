#include <torch/torch.h>

#include <metatensor.hpp>
#include <metatensor/torch.hpp>
using namespace metatensor_torch;

#include <catch.hpp>

static TorchTensorMap test_tensor_map();

TEST_CASE("TensorMap") {
    SECTION("keys") {
        auto tensor = test_tensor_map();

        auto expected = metatensor::Labels({"key_1", "key_2"}, {{0, 0}, {1, 0}, {2, 2}, {2, 3}});
        CHECK(*tensor->keys() == expected);
    }

    SECTION("get blocks") {
        auto tensor = test_tensor_map();

        // block by index
        auto block = TensorMapHolder::block_by_id(tensor, 2);
        const auto values = block->values();
        CHECK(values[0][0][0].item<double>() == 3);

        // block by selection
        auto selection = LabelsHolder::create({"key_1", "key_2"}, {{1, 0}});
        auto matching = tensor->blocks_matching(selection);
        CHECK(matching.size() == 1);
        CHECK(matching[0] == 1);

        selection = LabelsHolder::create({"key_2"}, {{0}});
        matching = tensor->blocks_matching(selection);
        CHECK(matching.size() == 2);
        CHECK(matching[0] == 0);
        CHECK(matching[1] == 1);
    }

    SECTION("keys_to_samples") {
        auto tensor = test_tensor_map()->keys_to_samples("key_2", /* sort_samples */ true);

        CHECK(*tensor->keys() == metatensor::Labels({"key_1"}, {{0}, {1}, {2}}));

        // The first two blocks are not modified
        auto block = TensorMapHolder::block_by_id(tensor, 0);
        CHECK(*block->samples() == metatensor::Labels({"samples", "key_2"}, {{0, 0}, {2, 0}, {4, 0}}));
        CHECK(torch::all(block->values() == torch::full({3, 1, 1}, 1.0)).item<bool>());


        block = TensorMapHolder::block_by_id(tensor, 1);
        CHECK(*block->samples() == metatensor::Labels({"samples", "key_2"}, {{0, 0}, {1, 0}, {3, 0}}));
        CHECK(torch::all(block->values() == torch::full({3, 1, 3}, 2.0)).item<bool>());

        // The new third block contains the old third and fourth blocks merged
        block = TensorMapHolder::block_by_id(tensor, 2);
        CHECK(*block->samples() == metatensor::Labels({"samples", "key_2"}, {
            {0, 2}, {0, 3}, {1, 3}, {2, 3}, {3, 2}, {5, 3}, {6, 2}, {8, 2}
        }));

        auto expected = torch::tensor(std::vector<double>{
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0,
            4.0, 4.0, 4.0,
            4.0, 4.0, 4.0,
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0,
            3.0, 3.0, 3.0,
            3.0, 3.0, 3.0,
        }).reshape({8, 3, 1});

        CHECK(torch::all(block->values() == expected).item<bool>());

        auto gradient = TensorBlockHolder::gradient(block, "parameter");
        CHECK(*gradient->samples() == metatensor::Labels({"sample", "parameter"}, {
            {1, 1}, {4, -2}, {5, 3},
        }));

        expected = torch::tensor(std::vector<double>{
            14.0, 14.0, 14.0,
            13.0, 13.0, 13.0,
            14.0, 14.0, 14.0,
        }).reshape({3, 3, 1});

        CHECK(torch::all(gradient->values() == expected).item<bool>());

        // unsorted samples
        tensor = test_tensor_map()->keys_to_samples("key_2", /*sort_samples*/ false);

        block = TensorMapHolder::block_by_id(tensor, 2);
        CHECK(*block->samples() == metatensor::Labels({"samples", "key_2"}, {
            {0, 2}, {3, 2}, {6, 2}, {8, 2}, {0, 3}, {1, 3}, {2, 3}, {5, 3}
        }));
    }

    SECTION("keys_to_properties") {
        auto tensor = test_tensor_map()->keys_to_properties("key_1", /*sort_samples*/ true);

        CHECK(*tensor->keys() == metatensor::Labels({"key_2"}, {{0}, {2}, {3}}));

        // The new first block contains the old first two blocks merged
        auto block = TensorMapHolder::block_by_id(tensor, 0);
        CHECK(*block->samples() == metatensor::Labels({"samples"}, {{0}, {1}, {2}, {3}, {4}}));

        auto components = block->components();
        CHECK(components.size() == 1);
        CHECK(*components[0] == metatensor::Labels({"component"}, {{0}}));

        CHECK(*block->properties() == metatensor::Labels({"key_1", "properties"}, {
            {0, 0}, {1, 3}, {1, 4}, {1, 5}
        }));

        auto expected = torch::tensor(std::vector<double>{
            1.0, 2.0, 2.0, 2.0,
            0.0, 2.0, 2.0, 2.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 2.0, 2.0,
            1.0, 0.0, 0.0, 0.0,
        }).reshape({5, 1, 4});

        CHECK(torch::all(block->values() == expected).item<bool>());

        auto gradient = TensorBlockHolder::gradient(block, "parameter");
        CHECK(*gradient->samples() == metatensor::Labels({"sample", "parameter"}, {{0, -2}, {0, 3}, {3, -2}, {4, 3}}));

        expected = torch::tensor(std::vector<double>{
            11.0, 12.0, 12.0, 12.0,
            0.0, 12.0, 12.0, 12.0,
            0.0, 12.0, 12.0, 12.0,
            11.0, 0.0, 0.0, 0.0,
        }).reshape({4, 1, 4});

        CHECK(torch::all(gradient->values() == expected).item<bool>());

        // The new second block contains the old third block
        block = TensorMapHolder::block_by_id(tensor, 1);
        CHECK(*block->properties() == metatensor::Labels({"key_1", "properties"}, {{2, 0}}));

        CHECK(torch::all(block->values() == torch::full({4, 3, 1}, 3.0)).item<bool>());

        // the new third block contains the old fourth block
        block = TensorMapHolder::block_by_id(tensor, 2);
        CHECK(*block->properties() == metatensor::Labels({"key_1", "properties"}, {{2, 0}}));

        CHECK(torch::all(block->values() == torch::full({4, 3, 1}, 4.0)).item<bool>());
    }

    SECTION("component_to_properties") {
        auto tensor = test_tensor_map()->components_to_properties("component");

        auto block = TensorMapHolder::block_by_id(tensor, 0);

        CHECK(*block->samples() == metatensor::Labels({"samples"}, {{0}, {2}, {4}}));

        auto components = block->components();
        CHECK(components.empty());

        CHECK(*block->properties() == metatensor::Labels({"component", "properties"}, {{0, 0}}));
    }

    SECTION("different devices") {
        auto tensor = test_tensor_map();
        CHECK_THROWS_WITH(
            TensorMapHolder(
                tensor->keys()->to(torch::kMeta),
                metatensor_torch::TensorMapHolder::blocks(tensor)
            ),
            Catch::StartsWith(
                "cannot create TensorMap: keys and blocks must be on the "
                "same device, got cpu and meta"
            )
        );
    }
}

TEST_CASE("TensorMap serialization") {
    SECTION("loading file") {
        // DATA_NPZ is defined by cmake and expand to the path of tests/data.npz
        auto tensor = metatensor_torch::load(DATA_NPZ);

        auto keys = tensor->keys();
        CHECK(keys->names().size() == 4);
        CHECK(keys->names()[0] == std::string("o3_lambda"));
        CHECK(keys->names()[1] == std::string("o3_sigma"));
        CHECK(keys->names()[2] == std::string("center_type"));
        CHECK(keys->names()[3] == std::string("neighbor_type"));
        CHECK(keys->count() == 27);

        auto block = TensorMapHolder::block_by_id(tensor, 21);

        auto samples = block->samples();
        CHECK(samples->names().size() == 2);
        CHECK(samples->names()[0] == std::string("system"));
        CHECK(samples->names()[1] == std::string("atom"));

        CHECK(block->values().sizes() == std::vector<int64_t>{9, 5, 3});

        auto gradient = TensorBlockHolder::gradient(block, "positions");
        samples = gradient->samples();
        CHECK(samples->names().size() == 3);
        CHECK(samples->names()[0] == std::string("sample"));
        CHECK(samples->names()[1] == std::string("system"));
        CHECK(samples->names()[2] == std::string("atom"));

        CHECK(gradient->values().sizes() == std::vector<int64_t>{59, 3, 5, 3});
    }
}


TorchTensorMap test_tensor_map() {
    auto blocks = std::vector<TorchTensorBlock>();

    auto components = std::vector<TorchLabels>();
    components.emplace_back(LabelsHolder::create({"component"}, {{0}}));

    auto block_1 = torch::make_intrusive<TensorBlockHolder>(
        torch::full({3, 1, 1}, 1.0),
        LabelsHolder::create({"samples"}, {{0}, {2}, {4}}),
        components,
        LabelsHolder::create({"properties"}, {{0}})
    );
    auto gradient_1 = torch::make_intrusive<TensorBlockHolder>(
        torch::full({2, 1, 1}, 11.0),
        LabelsHolder::create({"sample", "parameter"}, {{0, -2}, {2, 3}}),
        components,
        LabelsHolder::create({"properties"}, {{0}})
    );
    block_1->add_gradient("parameter", std::move(gradient_1));

    blocks.emplace_back(std::move(block_1));

    auto block_2 = torch::make_intrusive<TensorBlockHolder>(
        torch::full({3, 1, 3}, 2.0),
        LabelsHolder::create({"samples"}, {{0}, {1}, {3}}),
        components,
        LabelsHolder::create({"properties"}, {{3}, {4}, {5}})
    );
    auto gradient_2 = torch::make_intrusive<TensorBlockHolder>(
        torch::full({3, 1, 3}, 12.0),
        LabelsHolder::create({"sample", "parameter"}, {{0, -2}, {0, 3}, {2, -2}}),
        components,
        LabelsHolder::create({"properties"}, {{3}, {4}, {5}})
    );
    block_2->add_gradient("parameter", std::move(gradient_2));

    blocks.emplace_back(std::move(block_2));

    components = std::vector<TorchLabels>();
    components.emplace_back(LabelsHolder::create({"component"}, {{0}, {1}, {2}}));
    auto block_3 = torch::make_intrusive<TensorBlockHolder>(
        torch::full({4, 3, 1}, 3.0),
        LabelsHolder::create({"samples"}, {{0}, {3}, {6}, {8}}),
        components,
        LabelsHolder::create({"properties"}, {{0}})
    );
    auto gradient_3 = torch::make_intrusive<TensorBlockHolder>(
        torch::full({1, 3, 1}, 13.0),
        LabelsHolder::create({"sample", "parameter"}, {{1, -2}}),
        components,
        LabelsHolder::create({"properties"}, {{0}})
    );
    block_3->add_gradient("parameter", std::move(gradient_3));

    blocks.emplace_back(std::move(block_3));

    auto block_4 = torch::make_intrusive<TensorBlockHolder>(
        torch::full({4, 3, 1}, 4.0),
        LabelsHolder::create({"samples"}, {{0}, {1}, {2}, {5}}),
        components,
        LabelsHolder::create({"properties"}, {{0}})
    );
    auto gradient_4 = torch::make_intrusive<TensorBlockHolder>(
        torch::full({2, 3, 1}, 14.0),
        LabelsHolder::create({"sample", "parameter"}, {{0, 1}, {3, 3}}),
        components,
        LabelsHolder::create({"properties"}, {{0}})
    );
    block_4->add_gradient("parameter", std::move(gradient_4));

    blocks.emplace_back(std::move(block_4));

    auto keys = LabelsHolder::create(
        {"key_1", "key_2"},
        {{0, 0}, {1, 0}, {2, 2}, {2, 3}}
    );

    return torch::make_intrusive<TensorMapHolder>(std::move(keys), std::move(blocks));
}

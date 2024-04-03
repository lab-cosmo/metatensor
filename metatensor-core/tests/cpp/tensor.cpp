#include <fstream>
#include <sstream>
#include <cstdlib>

#include <catch.hpp>

#include <metatensor.hpp>
using namespace metatensor;

static TensorMap test_tensor_map();
static mts_status_t custom_create_array(const uintptr_t* shape_ptr, uintptr_t shape_count, mts_array_t *array);
static void check_loaded_tensor(metatensor::TensorMap& tensor);

static int CUSTOM_CREATE_ARRAY_CALL_COUNT = 0;

TEST_CASE("TensorMap") {
    SECTION("keys") {
        auto tensor = test_tensor_map();
        CHECK(tensor.keys() == Labels({"key_1", "key_2"}, {{0, 0}, {1, 0}, {2, 2}, {2, 3}}));
    }

    SECTION("get blocks") {
        auto tensor = test_tensor_map();

        // block by index
        auto block = tensor.block_by_id(2);
        const auto values = block.values();
        CHECK(values(0, 0, 0) == 3);

        // block by selection
        auto selection = Labels({"key_1", "key_2"}, {{1, 0}});
        auto matching = tensor.blocks_matching(selection);
        CHECK(matching.size() == 1);
        CHECK(matching[0] == 1);

        selection = Labels({"key_2"}, {{0}});
        matching = tensor.blocks_matching(selection);
        CHECK(matching.size() == 2);
        CHECK(matching[0] == 0);
        CHECK(matching[1] == 1);
    }

    SECTION("keys_to_samples") {
        auto tensor = test_tensor_map().keys_to_samples("key_2", /* sort_samples */ true);

        CHECK(tensor.keys() == Labels({"key_1"}, {{0}, {1}, {2}}));

        // The first two blocks are not modified
        auto block = tensor.block_by_id(0);
        CHECK(block.samples() == Labels({"samples", "key_2"}, {{0, 0}, {2, 0}, {4, 0}}));
        const auto& values_1 = SimpleDataArray::from_mts_array(block.mts_array());
        CHECK(values_1 == SimpleDataArray({3, 1, 1}, 1.0));


        block = tensor.block_by_id(1);
        CHECK(block.samples() == Labels({"samples", "key_2"}, {{0, 0}, {1, 0}, {3, 0}}));

        const auto& values_2 = SimpleDataArray::from_mts_array(block.mts_array());
        CHECK(values_2 == SimpleDataArray({3, 1, 3}, 2.0));

        // The new third block contains the old third and fourth blocks merged
        block = tensor.block_by_id(2);
        CHECK(block.samples() == Labels({"samples", "key_2"}, {
            {0, 2}, {0, 3}, {1, 3}, {2, 3}, {3, 2}, {5, 3}, {6, 2}, {8, 2}
        }));

        auto expected = SimpleDataArray({8, 3, 1}, {
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0,
            4.0, 4.0, 4.0,
            4.0, 4.0, 4.0,
            3.0, 3.0, 3.0,
            4.0, 4.0, 4.0,
            3.0, 3.0, 3.0,
            3.0, 3.0, 3.0,
        });

        const auto& values_3 = SimpleDataArray::from_mts_array(block.mts_array());
        CHECK(values_3 == expected);

        auto gradient = block.gradient("parameter");
        CHECK(gradient.samples() == Labels({"sample", "parameter"}, {
            {1, 1}, {4, -2}, {5, 3},
        }));

        expected = SimpleDataArray({3, 3, 1}, {
            14.0, 14.0, 14.0,
            13.0, 13.0, 13.0,
            14.0, 14.0, 14.0,
        });

        auto gradient_3 = SimpleDataArray::from_mts_array(block.gradient("parameter").mts_array());
        CHECK(gradient_3 == expected);

        // unsorted samples
        tensor = test_tensor_map().keys_to_samples("key_2", /*sort_samples*/ false);

        block = tensor.block_by_id(2);
        CHECK(block.samples() == Labels({"samples", "key_2"}, {
            {0, 2}, {3, 2}, {6, 2}, {8, 2}, {0, 3}, {1, 3}, {2, 3}, {5, 3}
        }));
    }

    SECTION("keys_to_properties") {
        auto tensor = test_tensor_map().keys_to_properties("key_1");

        CHECK(tensor.keys() == Labels({"key_2"}, {{0}, {2}, {3}}));

        // The new first block contains the old first two blocks merged
        auto block = tensor.block_by_id(0);
        CHECK(block.samples() == Labels({"samples"}, {{0}, {1}, {2}, {3}, {4}}));

        auto components = block.components();
        CHECK(components.size() == 1);
        CHECK(components[0] == Labels({"component"}, {{0}}));

        CHECK(block.properties() == Labels({"key_1", "properties"}, {
            {0, 0}, {1, 3}, {1, 4}, {1, 5}
        }));

        auto expected = SimpleDataArray({5, 1, 4}, {
            1.0, 2.0, 2.0, 2.0,
            0.0, 2.0, 2.0, 2.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 2.0, 2.0,
            1.0, 0.0, 0.0, 0.0,
        });

        const auto& values_1 = SimpleDataArray::from_mts_array(block.mts_array());
        CHECK(values_1 == expected);

        auto gradient = block.gradient("parameter");
        CHECK(gradient.samples() == Labels({"sample", "parameter"}, {{0, -2}, {0, 3}, {3, -2}, {4, 3}}));

        expected = SimpleDataArray({4, 1, 4}, {
            11.0, 12.0, 12.0, 12.0,
            0.0, 12.0, 12.0, 12.0,
            0.0, 12.0, 12.0, 12.0,
            11.0, 0.0, 0.0, 0.0,
        });

        auto gradient_1 = SimpleDataArray::from_mts_array(block.gradient("parameter").mts_array());
        CHECK(gradient_1 == expected);

        // The new second block contains the old third block
        block = tensor.block_by_id(1);
        CHECK(block.properties() == Labels({"key_1", "properties"}, {{2, 0}}));

        auto values_2 = SimpleDataArray::from_mts_array(block.mts_array());
        CHECK(values_2 == SimpleDataArray({4, 3, 1}, 3.0));

        // the new third block contains the old fourth block
        block = tensor.block_by_id(2);
        CHECK(block.properties() == Labels({"key_1", "properties"}, {{2, 0}}));

        auto values_3 = SimpleDataArray::from_mts_array(block.mts_array());
        CHECK(values_3 == SimpleDataArray({4, 3, 1}, 4.0));
    }

    SECTION("component_to_properties") {
        auto tensor = test_tensor_map().components_to_properties("component");

        auto block = tensor.block_by_id(0);

        CHECK(block.samples() == Labels({"samples"}, {{0}, {2}, {4}}));

        auto components = block.components();
        CHECK(components.empty());

        CHECK(block.properties() == Labels({"component", "properties"}, {{0, 0}}));
    }

    SECTION("clone") {
        auto blocks = std::vector<TensorBlock>();
        blocks.push_back(TensorBlock(
            std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            {},
            Labels({"properties"}, {{5}, {3}})
        ));
        auto tensor = TensorMap(Labels({"keys"}, {{0}}), std::move(blocks));

        // This should be fine
        auto clone = tensor.clone();

        class BrokenDataArray: public metatensor::SimpleDataArray {
        public:
            BrokenDataArray(std::vector<size_t> shape): metatensor::SimpleDataArray(std::move(shape)) {}

            std::unique_ptr<DataArrayBase> copy() const override {
                throw std::runtime_error("can not copy this!");
            }
        };

        blocks = std::vector<TensorBlock>();
        blocks.push_back(TensorBlock(
            std::unique_ptr<BrokenDataArray>(new BrokenDataArray({3, 2})),
            Labels({"samples"}, {{0}, {1}, {4}}),
            {},
            Labels({"properties"}, {{5}, {3}})
        ));
        tensor = TensorMap(Labels({"keys"}, {{0}}), std::move(blocks));

        CHECK_THROWS_WITH(tensor.clone(), "external error: calling mts_array_t.create failed (status -1)");
    }

    SECTION("clone metadata") {
        auto tensor = test_tensor_map();

        auto clone = tensor.clone_metadata_only();
        CHECK(clone.keys() == tensor.keys());

        auto block = clone.block_by_id(0);
        CHECK_THROWS_WITH(block.values(), "error in C++ callback: can not call `data` for an EmptyDataArray");
    }
}


TEST_CASE("TensorMap serialization") {
    SECTION("loading file") {
        // DATA_NPZ is defined by cmake and expand to the path of
        // `tests/data.npz`
        auto tensor = TensorMap::load(TEST_DATA_NPZ_PATH);
        check_loaded_tensor(tensor);

        tensor = metatensor::io::load(TEST_DATA_NPZ_PATH);
        check_loaded_tensor(tensor);
    }

    SECTION("loading file with custom array creation") {
        CHECK(CUSTOM_CREATE_ARRAY_CALL_COUNT == 0);
        auto tensor = TensorMap::load(TEST_DATA_NPZ_PATH, custom_create_array);
        // 27 blocks, one array for values, one array for gradients
        CHECK(CUSTOM_CREATE_ARRAY_CALL_COUNT == 27 * 2);

        CUSTOM_CREATE_ARRAY_CALL_COUNT = 0;
        tensor = metatensor::io::load(TEST_DATA_NPZ_PATH, custom_create_array);
        CHECK(CUSTOM_CREATE_ARRAY_CALL_COUNT == 27 * 2);
    }

    SECTION("Load/Save with buffers") {
        // read the whole file into a buffer
        std::ifstream file(TEST_DATA_NPZ_PATH, std::ios::binary);
        std::ostringstream string_stream;
        string_stream << file.rdbuf();
        auto buffer = string_stream.str();

        auto tensor = TensorMap::load_buffer(buffer);
        tensor = metatensor::io::load_buffer(buffer);
        check_loaded_tensor(tensor);

        auto saved = tensor.save_buffer<std::string>();
        REQUIRE(saved.size() == buffer.size());
        CHECK(saved == buffer);

        saved = metatensor::io::save_buffer<std::string>(tensor);
        REQUIRE(saved.size() == buffer.size());
        CHECK(saved == buffer);

        // using the raw C API, without user_data in the callback, and making
        // the callback a small wrapper around std::realloc
        uint8_t* raw_buffer = nullptr;
        uintptr_t buflen = 0;
        auto status = mts_tensormap_save_buffer(
            &raw_buffer,
            &buflen,
            nullptr,
            [](void*, uint8_t* ptr, uintptr_t new_size){
                return static_cast<uint8_t*>(std::realloc(ptr, new_size));
            },
            tensor.as_mts_tensormap_t()
        );
        REQUIRE(status == MTS_SUCCESS);
        CHECK(saved == std::string(raw_buffer, raw_buffer + buflen));

        std::free(raw_buffer);
    }
}


TensorMap test_tensor_map() {
    auto blocks = std::vector<TensorBlock>();

    auto components = std::vector<Labels>();
    components.emplace_back(Labels({"component"}, {{0}}));

    auto block_1 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 1, 1}, 1.0)),
        Labels({"samples"}, {{0}, {2}, {4}}),
        components,
        Labels({"properties"}, {{0}})
    );
    auto gradient_1 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({2, 1, 1}, 11.0)),
        Labels({"sample", "parameter"}, {{0, -2}, {2, 3}}),
        components,
        Labels({"properties"}, {{0}})
    );
    block_1.add_gradient("parameter", std::move(gradient_1));

    blocks.emplace_back(std::move(block_1));

    auto block_2 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 1, 3}, 2.0)),
        Labels({"samples"}, {{0}, {1}, {3}}),
        components,
        Labels({"properties"}, {{3}, {4}, {5}})
    );
    auto gradient_2 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 1, 3}, 12.0)),
        Labels({"sample", "parameter"}, {{0, -2}, {0, 3}, {2, -2}}),
        components,
        Labels({"properties"}, {{3}, {4}, {5}})
    );
    block_2.add_gradient("parameter", std::move(gradient_2));

    blocks.emplace_back(std::move(block_2));

    components = std::vector<Labels>();
    components.emplace_back(Labels({"component"}, {{0}, {1}, {2}}));
    auto block_3 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({4, 3, 1}, 3.0)),
        Labels({"samples"}, {{0}, {3}, {6}, {8}}),
        components,
        Labels({"properties"}, {{0}})
    );
    auto gradient_3 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({1, 3, 1}, 13.0)),
        Labels({"sample", "parameter"}, {{1, -2}}),
        components,
        Labels({"properties"}, {{0}})
    );
    block_3.add_gradient("parameter", std::move(gradient_3));

    blocks.emplace_back(std::move(block_3));

    auto block_4 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({4, 3, 1}, 4.0)),
        Labels({"samples"}, {{0}, {1}, {2}, {5}}),
        components,
        Labels({"properties"}, {{0}})
    );
    auto gradient_4 = TensorBlock(
        std::unique_ptr<SimpleDataArray>(new SimpleDataArray({2, 3, 1}, 14.0)),
        Labels({"sample", "parameter"}, {{0, 1}, {3, 3}}),
        components,
        Labels({"properties"}, {{0}})
    );
    block_4.add_gradient("parameter", std::move(gradient_4));

    blocks.emplace_back(std::move(block_4));

    auto keys = Labels(
        {"key_1", "key_2"},
        {{0, 0}, {1, 0}, {2, 2}, {2, 3}}
    );

    return TensorMap(std::move(keys), std::move(blocks));
}


mts_status_t custom_create_array(const uintptr_t* shape_ptr, uintptr_t shape_count, mts_array_t *array) {
    auto shape = std::vector<size_t>();
    for (size_t i=0; i<shape_count; i++) {
        shape.push_back(static_cast<size_t>(shape_ptr[i]));
    }

    CUSTOM_CREATE_ARRAY_CALL_COUNT += 1;

    auto cxx_array = std::unique_ptr<DataArrayBase>(new SimpleDataArray(shape));
    *array = DataArrayBase::to_mts_array_t(std::move(cxx_array));

    return MTS_SUCCESS;
}

void check_loaded_tensor(metatensor::TensorMap& tensor) {
    auto keys = tensor.keys();
    CHECK(keys.names().size() == 4);
    CHECK(keys.names()[0] == std::string("o3_lambda"));
    CHECK(keys.names()[1] == std::string("o3_sigma"));
    CHECK(keys.names()[2] == std::string("center_type"));
    CHECK(keys.names()[3] == std::string("neighbor_type"));
    CHECK(keys.count() == 27);

    auto block = tensor.block_by_id(21);

    auto samples = block.samples();
    CHECK(samples.names().size() == 2);
    CHECK(samples.names()[0] == std::string("system"));
    CHECK(samples.names()[1] == std::string("atom"));

    auto values = block.values();
    CHECK(values.shape() == std::vector<size_t>{9, 5, 3});

    auto gradient = block.gradient("positions");
    samples = gradient.samples();
    CHECK(samples.names().size() == 3);
    CHECK(samples.names()[0] == std::string("sample"));
    CHECK(samples.names()[1] == std::string("system"));
    CHECK(samples.names()[2] == std::string("atom"));

    values = gradient.values();
    CHECK(values.shape() == std::vector<size_t>{59, 3, 5, 3});
}

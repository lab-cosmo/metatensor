#include <fstream>
#include <sstream>

#include <catch.hpp>

#include <metatensor.hpp>
using namespace metatensor;

TEST_CASE("Labels") {
    auto labels = Labels({"foo", "bar"}, {{1, 2}, {3, 4}, {5, 6}});

    CHECK(labels.values().shape().size() == 2);
    CHECK(labels.values().shape()[0] == 3);
    CHECK(labels.values().shape()[1] == 2);

    CHECK(labels.count() == 3);
    CHECK(labels.size() == 2);

    CHECK(labels.names().size() == 2);
    CHECK(labels.names()[0] == std::string("foo"));
    CHECK(labels.names()[1] == std::string("bar"));

    CHECK(labels.position({3, 4}) == 1);
    CHECK(labels.position({1, 4}) == -1);

    const auto& values = labels.values();
    CHECK(values(0, 0) == 1);
    CHECK(values(0, 1) == 2);

    CHECK(values(1, 0) == 3);
    CHECK(values(1, 1) == 4);

    CHECK(values(2, 0) == 5);
    CHECK(values(2, 1) == 6);

    CHECK(labels == Labels({"foo", "bar"}, {{1, 2}, {3, 4}, {5, 6}}));
    CHECK(labels != Labels({"bar", "foo"}, {{1, 2}, {3, 4}, {5, 6}}));
    CHECK(labels != Labels({"foo", "bar"}, {{1, 2}, {3, 4}, {5, 5}}));

    CHECK_THROWS_WITH(
        labels.position({3, 4, 5}),
        "invalid parameter: expected label of size 2 in mts_labels_position, got size 3"
    );

    CHECK_THROWS_WITH(Labels({"foo"}, {{1}, {3, 4}}), "invalid size for row: expected 1 got 2");

    CHECK_THROWS_WITH(
        Labels({"not an ident"}, {{0}}),
        "invalid parameter: 'not an ident' is not a valid label name"
    );

    auto empty = Labels({}, {});
    CHECK(empty.size() == 0);
    CHECK(empty.count() == 0);

    CHECK_THROWS_WITH(Labels({}, {{}, {}, {}}), "invalid parameter: can not have labels.count > 0 if labels.size is 0");
}


TEST_CASE("Set operations") {
    SECTION("union") {
        auto first = Labels({"aa", "bb"}, {{0, 1}, {1, 2}});
        auto second = Labels({"aa", "bb"}, {{2, 3}, {1, 2}, {4, 5}});

        auto first_mapping = std::vector<int64_t>(first.count());
        auto second_mapping = std::vector<int64_t>(second.count());

        auto union_ = first.set_union(second, first_mapping, second_mapping);

        CHECK(union_.size() == 2);
        CHECK(union_.names()[0] == std::string("aa"));
        CHECK(union_.names()[1] == std::string("bb"));

        CHECK(union_.count() == 4);
        const auto& values = union_.values();
        CHECK(values(0, 0) == 0);
        CHECK(values(0, 1) == 1);

        CHECK(values(1, 0) == 1);
        CHECK(values(1, 1) == 2);

        CHECK(values(2, 0) == 2);
        CHECK(values(2, 1) == 3);

        CHECK(values(3, 0) == 4);
        CHECK(values(3, 1) == 5);

        auto expected = std::vector<int64_t>{0, 1};
        CHECK(first_mapping == expected);

        expected = std::vector<int64_t>{2, 1, 3};
        CHECK(second_mapping == expected);
    }

    SECTION("intersection") {
        auto first = Labels({"aa", "bb"}, {{0, 1}, {1, 2}});
        auto second = Labels({"aa", "bb"}, {{2, 3}, {1, 2}, {4, 5}});

        auto first_mapping = std::vector<int64_t>(first.count());
        auto second_mapping = std::vector<int64_t>(second.count());

        auto intersection = first.set_intersection(second, first_mapping, second_mapping);

        CHECK(intersection.size() == 2);
        CHECK(intersection.names()[0] == std::string("aa"));
        CHECK(intersection.names()[1] == std::string("bb"));

        CHECK(intersection.count() == 1);
        const auto& values = intersection.values();
        CHECK(values(0, 0) == 1);
        CHECK(values(0, 1) == 2);

        auto expected = std::vector<int64_t>{-1, 0};
        CHECK(first_mapping == expected);

        expected = std::vector<int64_t>{-1, 0, -1};
        CHECK(second_mapping == expected);
    }
}

struct UserData {
    std::string name;
    int64_t count;
    std::vector<double> values;
};

// some data will be stored here to check `user_data_delete` has been called
static int DELETE_CALL_MARKER = 0;

TEST_CASE("User data") {
    {
        // check that we can register and recover user data on Labels
        auto* data = new UserData{"aabbccdd", 0xDEADBEEF, {1.0, 2.0, 3.0, 42.1}};
        auto user_data = metatensor::LabelsUserData(data, [](void* ptr){
            DELETE_CALL_MARKER += 1;
            delete static_cast<UserData*>(ptr);
        });

        auto labels = Labels({"sample"}, {{0}, {1}, {2}});
        labels.set_user_data(std::move(user_data));

        auto* data_ptr = static_cast<UserData*>(labels.user_data());
        REQUIRE(data_ptr != nullptr);
        CHECK(data_ptr->name == "aabbccdd");
        CHECK(data_ptr->count == 0xDEADBEEF);
        CHECK(data_ptr->values == std::vector<double>{1.0, 2.0, 3.0, 42.1});

        // check we can recover user data from Labels inside a TensorBlock
        auto block = TensorBlock(
            std::unique_ptr<SimpleDataArray>(new SimpleDataArray({3, 2})),
            labels,
            {},
            Labels({"properties"}, {{5}, {3}})
        );

        auto samples = block.samples();
        data_ptr = static_cast<UserData*>(samples.user_data());
        REQUIRE(data_ptr != nullptr);
        CHECK(data_ptr->name == "aabbccdd");
        CHECK(data_ptr->count == 0xDEADBEEF);
        CHECK(data_ptr->values == std::vector<double>{1.0, 2.0, 3.0, 42.1});

        // check that we can mutate the data in-place
        data_ptr->count = 42;
        data_ptr->values.push_back(12356);

        samples = block.samples();
        data_ptr = static_cast<UserData*>(samples.user_data());
        REQUIRE(data_ptr != nullptr);
        CHECK(data_ptr->name == "aabbccdd");
        CHECK(data_ptr->count == 42);
        CHECK(data_ptr->values == std::vector<double>{1.0, 2.0, 3.0, 42.1, 12356});

        // check that we can register user data after the construction of the block
        data = new UserData{"properties", 0, {}};
        user_data = metatensor::LabelsUserData(data, [](void* ptr){
            DELETE_CALL_MARKER += 10000000;
            delete static_cast<UserData*>(ptr);
        });

        auto properties = block.properties();
        properties.set_user_data(std::move(user_data));

        properties = block.properties();
        data_ptr = static_cast<UserData*>(properties.user_data());
        REQUIRE(data_ptr != nullptr);
        CHECK(data_ptr->name == "properties");
        CHECK(data_ptr->count == 0);
        CHECK(data_ptr->values.empty());
    }

    // Check that the `user_data_delete` function was called the
    // appropriate number of times
    CHECK(DELETE_CALL_MARKER == 10000001);
}

void check_loaded_labels(metatensor::Labels& labels) {
    CHECK(labels.names().size() == 4);
    CHECK(labels.names()[0] == std::string("o3_lambda"));
    CHECK(labels.names()[1] == std::string("o3_sigma"));
    CHECK(labels.names()[2] == std::string("center_type"));
    CHECK(labels.names()[3] == std::string("neighbor_type"));
    CHECK(labels.count() == 27);
}

TEST_CASE("Serialization") {
    SECTION("loading file") {
        // TEST_KEYS_NPY_PATH is defined by cmake and expand to the path of
        // `tests/keys.npy`
        auto labels = Labels::load(TEST_KEYS_NPY_PATH);
        check_loaded_labels(labels);

        labels = metatensor::io::load_labels(TEST_KEYS_NPY_PATH);
        check_loaded_labels(labels);
    }

    SECTION("Load/Save with buffers") {
        // read the whole file into a buffer
        std::ifstream file(TEST_KEYS_NPY_PATH, std::ios::binary);
        std::ostringstream string_stream;
        string_stream << file.rdbuf();
        auto buffer = string_stream.str();

        auto labels = Labels::load_buffer(buffer);
        check_loaded_labels(labels);

        labels = metatensor::io::load_labels_buffer(buffer);
        check_loaded_labels(labels);

        auto saved = labels.save_buffer<std::string>();
        REQUIRE(saved.size() == buffer.size());
        CHECK(saved == buffer);

        saved = metatensor::io::save_buffer<std::string>(labels);
        REQUIRE(saved.size() == buffer.size());
        CHECK(saved == buffer);

        // using the raw C API, without user_data in the callback, and making
        // the callback a small wrapper around std::realloc
        uint8_t* raw_buffer = nullptr;
        uintptr_t buflen = 0;
        auto status = mts_labels_save_buffer(
            &raw_buffer,
            &buflen,
            nullptr,
            [](void*, uint8_t* ptr, uintptr_t new_size){
                return static_cast<uint8_t*>(std::realloc(ptr, new_size));
            },
            labels.as_mts_labels_t()
        );
        REQUIRE(status == MTS_SUCCESS);
        CHECK(saved == std::string(raw_buffer, raw_buffer + buflen));

        std::free(raw_buffer);
    }
}

mod tensor {
    use std::io::Read;

    use metatensor::TensorMap;

    #[test]
    fn load_file() {
        let tensor = metatensor::io::load("../../metatensor-core/tests/data.npz").unwrap();
        check_tensor(&tensor);

        let tensor = TensorMap::load("../../metatensor-core/tests/data.npz").unwrap();
        check_tensor(&tensor);
    }

    #[test]
    fn load_buffer() {
        let mut file = std::fs::File::open("../../metatensor-core/tests/data.npz").unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let tensor = metatensor::io::load_buffer(&buffer).unwrap();
        check_tensor(&tensor);

        let tensor = TensorMap::load_buffer(&buffer).unwrap();
        check_tensor(&tensor);
    }

    #[test]
    fn save_buffer() {
        let mut file = std::fs::File::open("../../metatensor-core/tests/data.npz").unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let tensor = metatensor::io::load_buffer(&buffer).unwrap();

        let mut saved = Vec::new();
        metatensor::io::save_buffer(&tensor, &mut saved).unwrap();
        assert_eq!(buffer, saved);

        saved.clear();
        tensor.save_buffer(&mut saved).unwrap();
        assert_eq!(buffer, saved);
    }

    fn check_tensor(tensor: &TensorMap) {
        assert_eq!(tensor.keys().names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        assert_eq!(tensor.keys().count(), 27);

        let block = tensor.block_by_id(13);

        assert_eq!(block.values().as_array().shape(), [9, 3, 3]);
        assert_eq!(block.samples().names(), ["system", "atom"]);
        assert_eq!(block.components().len(), 1);
        assert_eq!(block.components()[0].names(), ["o3_mu"]);
        assert_eq!(block.properties().names(), ["n"]);

        assert_eq!(block.gradient_list(), ["positions"]);
        let gradient = block.gradient("positions").unwrap();
        assert_eq!(gradient.values().as_array().shape(), [27, 3, 3, 3]);
        assert_eq!(gradient.samples().names(), ["sample", "system", "atom"]);
        assert_eq!(gradient.components().len(), 2);
        assert_eq!(gradient.components()[0].names(), ["xyz"]);
        assert_eq!(gradient.components()[1].names(), ["o3_mu"]);
        assert_eq!(gradient.properties().names(), ["n"]);
    }
}


mod labels {
    use std::io::Read;
    use metatensor::Labels;

    #[test]
    fn load_file() {
        let labels = metatensor::io::load_labels("../../metatensor-core/tests/keys.npy").unwrap();
        check_labels(&labels);

        let labels = Labels::load("../../metatensor-core/tests/keys.npy").unwrap();
        check_labels(&labels);
    }

    #[test]
    fn load_buffer() {
        let mut file = std::fs::File::open("../../metatensor-core/tests/keys.npy").unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let labels = metatensor::io::load_labels_buffer(&buffer).unwrap();
        check_labels(&labels);

        let labels = Labels::load_buffer(&buffer).unwrap();
        check_labels(&labels);
    }

    #[test]
    fn save_buffer() {
        let mut file = std::fs::File::open("../../metatensor-core/tests/keys.npy").unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let labels = metatensor::io::load_labels_buffer(&buffer).unwrap();

        let mut saved = Vec::new();
        metatensor::io::save_labels_buffer(&labels, &mut saved).unwrap();
        assert_eq!(buffer, saved);

        saved.clear();
        labels.save_buffer(&mut saved).unwrap();
        assert_eq!(buffer, saved);
    }


    fn check_labels(labels: &Labels) {
        assert_eq!(labels.names(), ["o3_lambda", "o3_sigma", "center_type", "neighbor_type"]);
        assert_eq!(labels.count(), 27);
    }
}

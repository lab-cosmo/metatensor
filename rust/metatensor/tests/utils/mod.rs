#![allow(dead_code)]
#![allow(clippy::needless_return)]

use metatensor::{LabelsBuilder, Labels, TensorBlock, TensorMap};

use ndarray::ArrayD;

pub fn example_labels<const N: usize>(names: Vec<&str>, values: Vec<[i32; N]>) -> Labels {
    let mut builder = LabelsBuilder::new(names);
    for entry in values {
        builder.add(&entry);
    }
    return builder.finish();
}

pub fn example_block(
    samples: Vec<[i32; 1]>,
    components: Vec<[i32; 1]>,
    properties: Vec<[i32; 1]>,
    gradient_samples: Vec<[i32; 2]>,
    values: f64,
    gradient_values: f64
) -> TensorBlock {
    let samples = example_labels(vec!["samples"], samples);
    let components = [example_labels(vec!["components"], components)];
    let properties = example_labels(vec!["properties"], properties);

    let shape = vec![samples.count(), components[0].count(), properties.count()];
    let mut block = TensorBlock::new(
        ArrayD::from_elem(shape, values),
        &samples,
        &components,
        &properties,
    ).unwrap();

    let gradient_samples = example_labels(vec!["sample", "parameter"], gradient_samples);

    let shape = vec![gradient_samples.count(), components[0].count(), properties.count()];
    let gradient = TensorBlock::new(
        ArrayD::from_elem(shape, gradient_values),
        &gradient_samples,
        &components,
        &properties,
    ).unwrap();
    block.add_gradient("parameter", gradient).unwrap();

    return block;
}

pub fn example_tensor() -> TensorMap {
    let block_1 = example_block(
        /* samples          */ vec![[0], [2], [4]],
        /* components       */ vec![[0]],
        /* properties       */ vec![[0]],
        /* gradient_samples */ vec![[0, -2], [2, 3]],
        /* values           */ 1.0,
        /* gradient_values  */ 11.0,
    );

    // different property size
    let block_2 = example_block(
        /* samples          */ vec![[0], [1], [3]],
        /* components       */ vec![[0]],
        /* properties       */ vec![[3], [4], [5]],
        /* gradient_samples */ vec![[0, -2], [0, 3], [2, -2]],
        /* values           */ 2.0,
        /* gradient_values  */ 12.0,
    );

    // different component size
    let block_3 = example_block(
        /* samples          */ vec![[0], [3], [6], [8]],
        /* components       */ vec![[0], [1], [2]],
        /* properties       */ vec![[0]],
        /* gradient_samples */ vec![[1, -2]],
        /* values           */ 3.0,
        /* gradient_values  */ 13.0,
    );

    let block_4 = example_block(
        /* samples          */ vec![[0], [1], [2], [5]],
        /* components       */ vec![[0], [1], [2]],
        /* properties       */ vec![[0]],
        /* gradient_samples */ vec![[0, 1], [3, 3]],
        /* values           */ 4.0,
        /* gradient_values  */ 14.0,
    );

    let keys = Labels::new(
        ["key_1", "key_2"],
        &[
            [0, 0],
            [1, 0],
            [2, 2],
            [2, 3],
        ]
    );

    return TensorMap::new(keys, vec![block_1, block_2, block_3, block_4]).unwrap();
}

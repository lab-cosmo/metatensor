use metatensor::{Labels, TensorMap};

mod utils;
use utils::{example_tensor, example_block};

use ndarray::ArrayD;

#[test]
fn sorted_samples() {
    let keys_to_move = Labels::empty(vec!["key_2"]);
    let tensor = example_tensor().keys_to_samples(&keys_to_move, true).unwrap();

    assert_eq!(tensor.keys().count(), 3);
    assert_eq!(tensor.keys().names(), ["key_1"]);
    assert_eq!(tensor.keys()[0], [0]);
    assert_eq!(tensor.keys()[1], [1]);
    assert_eq!(tensor.keys()[2], [2]);

    // The first two blocks are not modified
    let block_1 = tensor.block_by_id(0);
    assert_eq!(block_1.values().as_array(), ArrayD::from_elem(vec![3, 1, 1], 1.0));
    assert_eq!(block_1.samples().count(), 3);
    assert_eq!(block_1.samples()[0], [0, 0]);
    assert_eq!(block_1.samples()[1], [2, 0]);
    assert_eq!(block_1.samples()[2], [4, 0]);

    let block_2 = tensor.block_by_id(1);
    assert_eq!(block_2.values().as_array(), ArrayD::from_elem(vec![3, 1, 3], 2.0));
    assert_eq!(block_2.samples().count(), 3);
    assert_eq!(block_2.samples()[0], [0, 0]);
    assert_eq!(block_2.samples()[1], [1, 0]);
    assert_eq!(block_2.samples()[2], [3, 0]);

    // The new third block contains the old third and fourth blocks merged
    let block_3 = tensor.block_by_id(2);
    assert_eq!(block_3.samples().names(), ["samples", "key_2"]);
    assert_eq!(block_3.samples().count(), 8);
    assert_eq!(block_3.samples()[0], [0, 2]);
    assert_eq!(block_3.samples()[1], [0, 3]);
    assert_eq!(block_3.samples()[2], [1, 3]);
    assert_eq!(block_3.samples()[3], [2, 3]);
    assert_eq!(block_3.samples()[4], [3, 2]);
    assert_eq!(block_3.samples()[5], [5, 3]);
    assert_eq!(block_3.samples()[6], [6, 2]);
    assert_eq!(block_3.samples()[7], [8, 2]);

    assert_eq!(block_3.components().len(), 1);
    assert_eq!(block_3.components()[0].names(), ["components"]);
    assert_eq!(block_3.components()[0].count(), 3);
    assert_eq!(block_3.components()[0][0], [0]);
    assert_eq!(block_3.components()[0][1], [1]);
    assert_eq!(block_3.components()[0][2], [2]);

    assert_eq!(block_3.properties().names(), ["properties"]);
    assert_eq!(block_3.properties().count(), 1);
    assert_eq!(block_3.properties()[0], [0]);

    let expected = ArrayD::from_shape_vec(vec![8, 3, 1], vec![
        3.0, 3.0, 3.0,
        4.0, 4.0, 4.0,
        4.0, 4.0, 4.0,
        4.0, 4.0, 4.0,
        3.0, 3.0, 3.0,
        4.0, 4.0, 4.0,
        3.0, 3.0, 3.0,
        3.0, 3.0, 3.0,
    ]).unwrap();
    assert_eq!(block_3.values().as_array(), expected);

    let gradient_3 = block_3.gradient("parameter").unwrap();
    assert_eq!(gradient_3.samples().names(), ["sample", "parameter"]);
    assert_eq!(gradient_3.samples().count(), 3);
    assert_eq!(gradient_3.samples()[0], [1, 1]);
    assert_eq!(gradient_3.samples()[1], [4, -2]);
    assert_eq!(gradient_3.samples()[2], [5, 3]);

    let expected = ArrayD::from_shape_vec(vec![3, 3, 1], vec![
        14.0, 14.0, 14.0,
        13.0, 13.0, 13.0,
        14.0, 14.0, 14.0,
    ]).unwrap();
    assert_eq!(gradient_3.values().as_array(), expected);
}

#[test]
fn unsorted_samples() {
    let keys_to_move = Labels::empty(vec!["key_2"]);
    let tensor = example_tensor().keys_to_samples(&keys_to_move, false).unwrap();

    let block_3 = tensor.block_by_id(2);
    assert_eq!(block_3.samples().names(), ["samples", "key_2"]);
    assert_eq!(block_3.samples().count(), 8);
    assert_eq!(block_3.samples()[0], [0, 2]);
    assert_eq!(block_3.samples()[1], [3, 2]);
    assert_eq!(block_3.samples()[2], [6, 2]);
    assert_eq!(block_3.samples()[3], [8, 2]);
    assert_eq!(block_3.samples()[4], [0, 3]);
    assert_eq!(block_3.samples()[5], [1, 3]);
    assert_eq!(block_3.samples()[6], [2, 3]);
    assert_eq!(block_3.samples()[7], [5, 3]);
}

#[test]
fn user_provided_entries() {
    let keys_to_move = Labels::new(["key_2"], &[[3]]);
    let result = example_tensor().keys_to_samples(&keys_to_move, false);

    assert_eq!(
        result.unwrap_err().message,
        "invalid parameter: user provided values for the keys to move is \
        not yet implemented, `keys_to_move` should not contain any entry \
        when calling keys_to_samples"
    );
}

#[test]
#[allow(clippy::vec_init_then_push)]
fn empty_samples() {
    let mut blocks = Vec::new();
    blocks.push(example_block(
        /* samples          */ vec![[0], [2], [4]],
        /* components       */ vec![[0]],
        /* properties       */ vec![[0], [1], [2], [3]],
        /* gradient_samples */ vec![],
        /* values           */ 1.0,
        /* gradient_values  */ 11.0,
    ));

    blocks.push(example_block(
        /* samples          */ vec![],
        /* components       */ vec![[0]],
        /* properties       */ vec![[0], [1], [2], [3]],
        /* gradient_samples */ vec![],
        /* values           */ 0.0,
        /* gradient_values  */ 0.0,
    ));
    let keys = Labels::new(
        ["key_1", "key_2"],
        &[[0, 0], [1, 0]]
    );

    let tensor = TensorMap::new(keys, blocks).unwrap();

    let keys_to_move = Labels::empty(vec!["key_1"]);
    let tensor = tensor.keys_to_samples(&keys_to_move, true).unwrap();

    assert_eq!(
        tensor.block_by_id(0).samples(),
        Labels::new(["samples", "key_1"], &[
            [0, 0],
            [2, 0],
            [4, 0],
        ])
    );
}

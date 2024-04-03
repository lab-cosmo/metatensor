#![allow(clippy::needless_return)]

use metatensor::{Labels, TensorMap};

mod utils;
use utils::{example_tensor, example_block, example_labels};

use ndarray::ArrayD;

#[test]
fn sorted_samples() {
    let keys_to_move = Labels::empty(vec!["key_1"]);
    let tensor = example_tensor().keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(tensor.keys(), &example_labels(vec!["key_2"], vec![[0], [2], [3]]));

    // The new first block contains the old first two blocks merged
    let block = tensor.block_by_id(0);
    assert_eq!(
        block.samples(),
        example_labels(vec!["samples"], vec![[0], [1], [2], [3], [4]])
    );

    assert_eq!(block.components().len(), 1);
    assert_eq!(
        block.components()[0],
        example_labels(vec!["components"], vec![[0]])
    );

    assert_eq!(
        block.properties(),
        example_labels(vec!["key_1", "properties"], vec![[0, 0], [1, 3], [1, 4], [1, 5]])
    );

    let expected = ArrayD::from_shape_vec(vec![5, 1, 4], vec![
        1.0, 2.0, 2.0, 2.0,
        0.0, 2.0, 2.0, 2.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 2.0, 2.0,
        1.0, 0.0, 0.0, 0.0,
    ]).unwrap();
    assert_eq!(block.values().as_array(), expected);

    let gradient_1 = block.gradient("parameter").unwrap();
    assert_eq!(
        gradient_1.samples(),
        example_labels(vec!["sample", "parameter"], vec![[0, -2], [0, 3], [3, -2], [4, 3]])
    );

    let expected = ArrayD::from_shape_vec(vec![4, 1, 4], vec![
        11.0, 12.0, 12.0, 12.0,
        0.0, 12.0, 12.0, 12.0,
        0.0, 12.0, 12.0, 12.0,
        11.0, 0.0, 0.0, 0.0,
    ]).unwrap();
    assert_eq!(gradient_1.values().as_array(), expected);

    // The new second block contains the old third block
    let block = tensor.block_by_id(1);
    assert_eq!(block.values().as_array(), ArrayD::from_elem(vec![4, 3, 1], 3.0));

    assert_eq!(
        block.properties(),
        example_labels(vec!["key_1", "properties"], vec![[2, 0]])
    );

    // The new third block contains the old fourth block
    let block = tensor.block_by_id(2);
    assert_eq!(block.values().as_array(), ArrayD::from_elem(vec![4, 3, 1], 4.0));
    assert_eq!(
        block.properties(),
        example_labels(vec!["key_1", "properties"], vec![[2, 0]])
    );
}

#[test]
fn unsorted_samples() {
    let keys_to_move = Labels::empty(vec!["key_1"]);
    let tensor = example_tensor().keys_to_properties(&keys_to_move, false).unwrap();

    assert_eq!(tensor.keys().count(), 3);

    let block = tensor.block_by_id(0);
    assert_eq!(
        block.samples(),
        example_labels(vec!["samples"], vec![[0], [2], [4], [1], [3]])
    );
}

#[test]
fn user_provided_entries_different_properties() {
    let keys_to_move = Labels::new(["key_1"], &[[0]]);
    let result = example_tensor().keys_to_properties(&keys_to_move, false);

    assert_eq!(
        result.unwrap_err().message,
        "invalid parameter: can not provide values for the keys to move to \
        properties if the blocks have different property labels"
    );
}

#[test]
#[allow(clippy::vec_init_then_push)]
fn empty_properties() {
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
        /* samples          */ vec![[0], [1], [3]],
        /* components       */ vec![[0]],
        /* properties       */ vec![],
        /* gradient_samples */ vec![],
        /* values           */ 0.0,
        /* gradient_values  */ 0.0,
    ));
    let keys = Labels::new(
        ["key_1", "key_2"],
        &[[0, 0], [1, 0]],
    );

    let mut tensor = TensorMap::new(keys, blocks).unwrap();

    let keys_to_move = Labels::empty(vec!["key_1"]);
    tensor = tensor.keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(
        tensor.block_by_id(0).properties(),
        Labels::new(["key_1", "properties"], &[
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
        ])
    );
}

#[allow(clippy::vec_init_then_push)]
fn example_tensor_same_properties_in_all_blocks() -> TensorMap {
    let mut blocks = Vec::new();
    blocks.push(example_block(
        /* samples          */ vec![[0], [2], [4]],
        /* components       */ vec![[0]],
        /* properties       */ vec![[0], [1], [2], [3]],
        /* gradient_samples */ vec![[0, 1], [0, 2], [2, 0]],
        /* values           */ 1.0,
        /* gradient_values  */ 11.0,
    ));

    blocks.push(example_block(
        /* samples          */ vec![[0], [1], [3]],
        /* components       */ vec![[0]],
        /* properties       */ vec![[0], [1], [2], [3]],
        /* gradient_samples */ vec![],
        /* values           */ 2.0,
        /* gradient_values  */ 12.0,
    ));

    blocks.push(example_block(
        /* samples          */ vec![[0], [1], [4]],
        /* components       */ vec![[0]],
        /* properties       */ vec![[0], [1], [2], [3]],
        /* gradient_samples */ vec![[0, 1], [0, 2], [2, 0]],
        /* values           */ 3.0,
        /* gradient_values  */ 13.0,
    ));

    let keys = Labels::new(
        ["key_1", "key_2"],
        &[[0, 0], [1, 0], [0, 1]]
    );

    return TensorMap::new(keys, blocks).unwrap();
}

#[test]
fn keys_to_move_in_different_order() {
    let keys_to_move = Labels::empty(vec!["key_1", "key_2"]);
    let reference_tensor = example_tensor_same_properties_in_all_blocks();
    let tensor = reference_tensor.keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(
        tensor.block_by_id(0).properties(),
        Labels::new(["key_1", "key_2", "properties"], &[
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 0, 3],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 1, 3],
        ])
    );

    let keys_to_move = Labels::empty(vec!["key_2", "key_1"]);
    let tensor = reference_tensor.keys_to_properties(&keys_to_move, true).unwrap();

    assert_eq!(
        tensor.block_by_id(0).properties(),
        Labels::new(["key_2", "key_1", "properties"], &[
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 1, 3],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 0, 3],
        ])
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn user_provided_entries() {
    let reference_tensor = example_tensor_same_properties_in_all_blocks();

    let keys_to_move = Labels::new(["key_1"], &[[0], [1]]);
    let tensor = reference_tensor.keys_to_properties(&keys_to_move, true).unwrap();

    // The new first block contains the old first two blocks merged
    let block = tensor.block_by_id(0);
    assert_eq!(
        block.samples(),
        example_labels(vec!["samples"], vec![[0], [1], [2], [3], [4]])
    );

    assert_eq!(block.components().len(), 1);
    assert_eq!(
        block.components()[0],
        example_labels(vec!["components"], vec![[0]])
    );

    assert_eq!(
        block.properties(),
        example_labels(vec!["key_1", "properties"], vec![
            [0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3],
        ])
    );

    let expected = ArrayD::from_shape_vec(vec![5, 1, 8], vec![
        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0,
        0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
    ]).unwrap();
    assert_eq!(block.values().as_array(), expected);

    // second block also contains the new property chanel even if it was not
    // merged with another block
    let block = tensor.block_by_id(1);
    let expected = ArrayD::from_shape_vec(vec![3, 1, 8], vec![
        3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0,
        3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0,
        3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0,
    ]).unwrap();
    assert_eq!(block.values().as_array(), expected);

    assert_eq!(
        block.properties(),
        example_labels(vec!["key_1", "properties"], vec![
            [0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3],
        ])
    );

    /**********************************************************************/

    // only keep a subset of the data
    let keys_to_move = Labels::new(["key_1"], &[[0]]);
    let tensor = reference_tensor.keys_to_properties(&keys_to_move, true).unwrap();

    let block = tensor.block_by_id(0);
    assert_eq!(
        block.samples(),
        example_labels(vec!["samples"], vec![[0], [1], [2], [3], [4]])
    );

    assert_eq!(block.components().len(), 1);
    assert_eq!(
        block.components()[0],
        example_labels(vec!["components"], vec![[0]])
    );

    assert_eq!(
        block.properties(),
        example_labels(vec!["key_1", "properties"], vec![
            [0, 0], [0, 1], [0, 2], [0, 3],
        ])
    );

    // only data from the first block was kept
    let expected = ArrayD::from_shape_vec(vec![5, 1, 4], vec![
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0,
    ]).unwrap();
    assert_eq!(block.values().as_array(), expected);

    // second block stayed the same, only properties labels changed
    let block = tensor.block_by_id(1);
    assert_eq!(block.values().as_array(), ArrayD::from_elem(vec![3, 1, 4], 3.0));

    assert_eq!(
        block.properties(),
        example_labels(vec!["key_1", "properties"], vec![
            [0, 0], [0, 1], [0, 2], [0, 3],
        ])
    );

    /**********************************************************************/
    // request keys not present in the input
    let keys_to_move = Labels::new(["key_1"], &[[0], [1], [2]]);
    let tensor = reference_tensor.keys_to_properties(&keys_to_move, true).unwrap();

    let block = tensor.block_by_id(0);
    assert_eq!(
        block.samples(),
        example_labels(vec!["samples"], vec![[0], [1], [2], [3], [4]])
    );

    assert_eq!(block.components().len(), 1);
    assert_eq!(
        block.components()[0],
        example_labels(vec!["components"], vec![[0]])
    );

    assert_eq!(
        block.properties(),
        example_labels(vec!["key_1", "properties"], vec![
            [0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3],
            [2, 0], [2, 1], [2, 2], [2, 3],
        ])
    );

    let expected = ArrayD::from_shape_vec(vec![5, 1, 12], vec![
        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0,
        1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]).unwrap();
    assert_eq!(block.values().as_array(), expected);

    // Second block
    let block = tensor.block_by_id(1);
    let expected = ArrayD::from_shape_vec(vec![3, 1, 12], vec![
        3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        3.0, 3.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]).unwrap();
    assert_eq!(block.values().as_array(), expected);

    assert_eq!(
        block.properties(),
        example_labels(vec!["key_1", "properties"], vec![
            [0, 0], [0, 1], [0, 2], [0, 3],
            [1, 0], [1, 1], [1, 2], [1, 3],
            [2, 0], [2, 1], [2, 2], [2, 3],
        ])
    );
}

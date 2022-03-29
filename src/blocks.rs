use std::sync::Arc;
use std::ffi::CString;
use std::collections::HashMap;

use crate::utils::ConstCString;
use crate::{Labels, Error, aml_array_t, get_data_origin};

/// Basic building block for descriptor. A single basic block contains a
/// 3-dimensional array, and three sets of labels (one for each dimension). The
/// sample labels are specific to this block, but components & property labels
/// can be shared between blocks, or between values & gradients.
#[derive(Debug)]
pub struct BasicBlock {
    pub data: aml_array_t,
    pub(crate) samples: Labels,
    pub(crate) components: Arc<Labels>,
    pub(crate) properties: Arc<Labels>,
}

fn check_data_label_shape(
    context: &str,
    data: &aml_array_t,
    samples: &Labels,
    components: &Labels,
    properties: &Labels,
) -> Result<(), Error> {
    let (n_samples, n_components, n_properties) = data.shape()?;
    if n_samples != samples.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis 0 is {} but we have {} sample labels",
            context, n_samples, samples.count()
        )));
    }

    if n_components != components.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis 1 is {} but we have {} components labels",
            context, n_components, components.count()
        )));
    }

    if n_properties != properties.count() {
        return Err(Error::InvalidParameter(format!(
            "{}: the array shape along axis 2 is {} but we have {} properties labels",
            context, n_properties, properties.count()
        )));
    }

    Ok(())
}

impl BasicBlock {
    /// Create a new `BasicBlock`, validating the shape of data & labels
    pub fn new(
        data: aml_array_t,
        samples: Labels,
        components: Arc<Labels>,
        properties: Arc<Labels>,
    ) -> Result<BasicBlock, Error> {
        check_data_label_shape(
            "data and labels don't match", &data, &samples, &components, &properties
        )?;

        return Ok(BasicBlock { data, samples, components, properties });
    }

    /// Get the sample labels in this basic block
    pub fn samples(&self) -> &Labels {
        &self.samples
    }

    /// Get the components labels in this basic block
    pub fn components(&self) -> &Arc<Labels> {
        &self.components
    }

    /// Get the property labels in this basic block
    pub fn properties(&self) -> &Arc<Labels> {
        &self.properties
    }
}

/// A single block in a descriptor, containing both values & optionally
/// gradients of these values w.r.t. any relevant quantity.
#[derive(Debug)]
pub struct Block {
    pub values: BasicBlock,
    gradients: HashMap<String, BasicBlock>,
    // all the keys from `self.gradients`, as C-compatible strings
    gradients_names: Vec<ConstCString>,
}

impl Block {
    /// Create a new `Block` containing the given data, described by the
    /// `samples`, `components`, and `properties` labels. The block is initialized
    /// without any gradients.
    pub fn new(
        data: aml_array_t,
        samples: Labels,
        components: Arc<Labels>,
        properties: Arc<Labels>,
    ) -> Result<Block, Error> {
        Ok(Block {
            values: BasicBlock::new(data, samples, components, properties)?,
            gradients: HashMap::new(),
            gradients_names: Vec::new(),
        })
    }

    /// Check if this block contains gradients w.r.t. the `name` parameter
    pub fn has_gradient(&self, name: &str) -> bool {
        self.gradients.contains_key(name)
    }

    /// Get the list of gradients in this block.
    pub fn gradients_list(&self) -> Vec<&str> {
        self.gradients_names.iter().map(|s| s.as_str()).collect()
    }

    /// Get the list of gradients in this block for the C API
    pub fn gradients_list_c(&self) -> &[ConstCString] {
        &self.gradients_names
    }

    /// Add a gradient to this block with the given name, samples and gradient
    /// array. The components and property labels are assumed to match the ones of
    /// the values in this block.
    pub fn add_gradient(&mut self, name: &str, samples: Labels, gradient: aml_array_t) -> Result<(), Error> {
        if self.gradients.contains_key(name) {
            return Err(Error::InvalidParameter(format!(
                "gradient with respect to '{}' already exists for this block", name
            )))
        }

        if gradient.origin()? != self.values.data.origin()? {
            return Err(Error::InvalidParameter(format!(
                "the gradient array has a different origin ('{}') than the value array ('{}')",
                get_data_origin(gradient.origin()?),
                get_data_origin(self.values.data.origin()?),
            )))
        }

        // this is used as a special marker in the C API
        if name == "values" {
            return Err(Error::InvalidParameter(
                "can not store gradient with respect to 'values'".into()
            ))
        }

        if samples.size() < 1 || samples.names()[0] != "sample" {
            return Err(Error::InvalidParameter(
                "first variable in the gradients samples labels must be 'samples'".into()
            ))
        }

        let components = Arc::clone(self.values.components());
        let properties = Arc::clone(self.values.properties());
        check_data_label_shape(
            "gradient data and labels don't match", &gradient, &samples, &components, &properties
        )?;

        self.gradients.insert(name.into(), BasicBlock {
            data: gradient,
            samples,
            components,
            properties
        });

        let name = ConstCString::new(CString::new(name.to_owned()).expect("invalid C string"));
        self.gradients_names.push(name);

        return Ok(())
    }

    /// Get the gradients w.r.t. `name` in this block or None.
    pub fn get_gradient(&self, name: &str) -> Option<&BasicBlock> {
        self.gradients.get(name)
    }
}

#[cfg(test)]
mod tests {
    use crate::{LabelValue, LabelsBuilder};
    use crate::data::TestArray;

    use super::*;

    #[test]
    fn gradients() {
        let mut samples = LabelsBuilder::new(vec!["a", "b"]);
        samples.add(vec![LabelValue::new(0), LabelValue::new(0)]);
        samples.add(vec![LabelValue::new(0), LabelValue::new(1)]);
        samples.add(vec![LabelValue::new(0), LabelValue::new(2)]);
        samples.add(vec![LabelValue::new(3), LabelValue::new(2)]);

        let mut components = LabelsBuilder::new(vec!["c", "d"]);
        components.add(vec![LabelValue::new(-1), LabelValue::new(-4)]);
        components.add(vec![LabelValue::new(-2), LabelValue::new(-5)]);
        components.add(vec![LabelValue::new(-3), LabelValue::new(-6)]);
        let components = Arc::new(components.finish());

        let mut properties = LabelsBuilder::new(vec!["f"]);
        properties.add(vec![LabelValue::new(0)]);
        properties.add(vec![LabelValue::new(1)]);
        properties.add(vec![LabelValue::new(2)]);
        properties.add(vec![LabelValue::new(3)]);
        properties.add(vec![LabelValue::new(4)]);
        properties.add(vec![LabelValue::new(5)]);
        properties.add(vec![LabelValue::new(6)]);
        let properties = Arc::new(properties.finish());

        let data = aml_array_t::new(Box::new(TestArray::new((4, 3, 7))));

        let mut block = Block::new(data, samples.finish(), components, properties).unwrap();
        assert!(block.gradients_list().is_empty());

        let gradient = aml_array_t::new(Box::new(TestArray::new((3, 3, 7))));
        let mut gradient_samples = LabelsBuilder::new(vec!["sample", "bar"]);
        gradient_samples.add(vec![LabelValue::new(0), LabelValue::new(0)]);
        gradient_samples.add(vec![LabelValue::new(1), LabelValue::new(1)]);
        gradient_samples.add(vec![LabelValue::new(3), LabelValue::new(-2)]);

        block.add_gradient("foo", gradient_samples.finish(), gradient).unwrap();

        assert_eq!(block.gradients_list(), ["foo"]);
        assert!(block.has_gradient("foo"));

        assert!(block.get_gradient("bar").is_none());
        let basic_block = block.get_gradient("foo").unwrap();

        assert_eq!(basic_block.samples().names(), ["sample", "bar"]);
        assert_eq!(basic_block.components().names(), ["c", "d"]);
        assert_eq!(basic_block.properties().names(), ["f"]);
    }
}

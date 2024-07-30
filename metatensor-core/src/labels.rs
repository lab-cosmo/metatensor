#![allow(clippy::default_trait_access, clippy::module_name_repetitions)]
use std::sync::RwLock;
use std::ffi::CString;
use std::collections::BTreeSet;
use std::os::raw::c_void;

use hashbrown::HashMap;
use hashbrown::hash_map::RawEntryMut;

use smallvec::SmallVec;

use crate::Error;
use crate::utils::ConstCString;

/// A single value inside a label. This is represented as a 32-bit signed
/// integer, with a couple of helper function to get its value as usize/isize.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct LabelValue(i32);

impl PartialEq<i32> for LabelValue {
    fn eq(&self, other: &i32) -> bool {
        self.0 == *other
    }
}

impl PartialEq<LabelValue> for i32 {
    fn eq(&self, other: &LabelValue) -> bool {
        *self == other.0
    }
}

impl std::fmt::Debug for LabelValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for LabelValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
impl From<u32> for LabelValue {
    fn from(value: u32) -> LabelValue {
        assert!(value < i32::MAX as u32);
        LabelValue(value as i32)
    }
}

impl From<i32> for LabelValue {
    fn from(value: i32) -> LabelValue {
        LabelValue(value)
    }
}

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
impl From<usize> for LabelValue {
    fn from(value: usize) -> LabelValue {
        assert!(value < i32::MAX as usize);
        LabelValue(value as i32)
    }
}

#[allow(clippy::cast_possible_truncation)]
impl From<isize> for LabelValue {
    fn from(value: isize) -> LabelValue {
        assert!(value < i32::MAX as isize && value > i32::MIN as isize);
        LabelValue(value as i32)
    }
}

impl LabelValue {
    /// Create a `LabelValue` with the given `value`
    pub fn new(value: i32) -> LabelValue {
        LabelValue(value)
    }

    /// Get the integer value of this `LabelValue` as a usize
    #[allow(clippy::cast_sign_loss)]
    pub fn usize(self) -> usize {
        debug_assert!(self.0 >= 0);
        self.0 as usize
    }

    /// Get the integer value of this `LabelValue` as an isize
    pub fn isize(self) -> isize {
        self.0 as isize
    }

    /// Get the integer value of this `LabelValue` as an i32
    pub fn i32(self) -> i32 {
        self.0
    }
}

type DefaultHasher = std::hash::BuildHasherDefault<ahash::AHasher>;

/// Builder for `Labels`, this should be used to construct `Labels`.
pub struct LabelsBuilder {
    // cf `Labels` for the documentation of the fields
    names: Vec<ConstCString>,
    values: Vec<LabelValue>,
    positions: HashMap<SmallVec<[LabelValue; 4]>, usize, DefaultHasher>,
}

impl LabelsBuilder {
    /// Create a new empty `LabelsBuilder` with the given `names`
    pub fn new(names: Vec<&str>) -> Result<LabelsBuilder, Error> {
        for name in &names {
            if !is_valid_label_name(name) {
                return Err(Error::InvalidParameter(format!(
                    "all labels names must be valid identifiers, '{}' is not", name
                )));
            }
        }

        let mut unique_names = BTreeSet::new();
        for name in &names {
            if !unique_names.insert(name) {
                return Err(Error::InvalidParameter(format!(
                    "labels names must be unique, got '{}' multiple times", name
                )));
            }
        }

        let names = names.into_iter()
            .map(|s| ConstCString::new(CString::new(s).expect("invalid C string")))
            .collect::<Vec<_>>();

        Ok(LabelsBuilder {
            names: names,
            values: Vec::new(),
            positions: Default::default(),
        })
    }

    /// Reserve space for `additional` other entries in the labels.
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional * self.names.len());
        self.positions.reserve(additional);
    }

    /// Get the number of labels in a single value
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Get the current number of entries
    pub fn count(&self) -> usize {
        if self.size() == 0 {
            return 0;
        } else {
            return self.values.len() / self.size();
        }
    }

    /// Add a single `entry` to this set of labels.
    ///
    /// This function will return an `Error` when attempting to add the same
    /// `label` more than once.
    pub fn add<T>(&mut self, entry: &[T]) -> Result<(), Error>
        where T: Copy + Into<LabelValue>
    {
        let entry = entry.iter().copied().map(Into::into).collect::<SmallVec<_>>();
        match self.add_or_get_position(entry) {
            Ok(_) => return Ok(()),
            Err((existing, entry)) => {
                let values_display = entry.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
                return Err(Error::InvalidParameter(format!(
                    "can not have the same label value multiple time: [{}] is already present at position {}",
                    values_display, existing
                )));
            }
        }
    }

    fn add_or_get_position(&mut self, labels_entry: SmallVec<[LabelValue; 4]>) -> Result<usize, (usize, SmallVec<[LabelValue; 4]>)> {
        assert_eq!(
            self.size(), labels_entry.len(),
            "wrong size for added label: got {}, but expected {}",
            labels_entry.len(), self.size()
        );

        let new_position = self.positions.len();

        match self.positions.raw_entry_mut().from_key(&labels_entry) {
            RawEntryMut::Occupied(entry) => {
                return Err((*entry.get(), labels_entry));
            },
            RawEntryMut::Vacant(entry) => {
                self.values.extend(&labels_entry);
                entry.insert(labels_entry, new_position);
            }
        }

        return Ok(new_position);
    }

    /// Finish building the `Labels`
    pub fn finish(self) -> Labels {
        if self.names.is_empty() {
            assert!(self.values.is_empty());
            return Labels {
                names: Vec::new(),
                values: Vec::new(),
                positions: Default::default(),
                user_data: RwLock::new(UserData::null()),
            }
        }

        return Labels {
            names: self.names,
            values: self.values,
            positions: self.positions,
            user_data: RwLock::new(UserData::null()),
        };
    }
}

/// Check if the given name is a valid identifier, to be used as a
/// column name in `Labels`.
pub fn is_valid_label_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    for (i, c) in name.chars().enumerate() {
        if i == 0 && c.is_ascii_digit() {
            return false;
        }

        if !(c.is_ascii_alphanumeric() || c == '_') {
            return false;
        }
    }

    return true;
}

#[derive(Debug)]
struct UserData {
    ptr: *mut c_void,
    delete: Option<unsafe extern fn(*mut c_void)>,
}

impl UserData {
    /// Create an empty `UserData`
    fn null() -> UserData {
        UserData {
            ptr: std::ptr::null_mut(),
            delete: None,
        }
    }
}

impl Drop for UserData {
    fn drop(&mut self) {
        if let Some(delete) = self.delete {
            unsafe {
                delete(self.ptr);
            }
        }
    }
}

// These should be enforced by the code using metatensor
unsafe impl Sync for UserData {}
unsafe impl Send for UserData {}

/// A set of labels used to carry metadata associated with a tensor map.
///
/// This is similar to a list of named tuples, but stored as a 2D array of shape
/// `(labels.count(), labels.size())`, with a of set names associated with the
/// columns of this array. Each row/entry in this array is unique, and they are
/// often (but not always) sorted in  lexicographic order.
///
/// The main way to construct a new set of labels is to use a `LabelsBuilder`.
pub struct Labels {
    /// Names of the labels, stored as const C strings for easier integration
    /// with the C API
    names: Vec<ConstCString>,
    /// Values of the labels, as a linearized 2D array in row-major order
    values: Vec<LabelValue>,
    /// Store the position of all the known labels, for faster access later.
    /// This uses `XxHash64` instead of the default hasher in std since
    /// `XxHash64` is much faster and we don't need the cryptographic strength
    /// hash from std.
    positions: HashMap<SmallVec<[LabelValue; 4]>, usize, DefaultHasher>,
    /// Some data provided by the user that we should keep around (this is
    /// used to store a pointer to the on-GPU tensor in metatensor-torch).
    user_data: RwLock<UserData>,
}

impl PartialEq for Labels {
    fn eq(&self, other: &Self) -> bool {
        self.names == other.names && self.values == other.values
    }
}

impl Eq for Labels {}


impl std::fmt::Debug for Labels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Labels{{")?;
        writeln!(f, "    {}", self.names().join(", "))?;

        let widths = self.names().iter().map(|s| s.len()).collect::<Vec<_>>();
        for values in self {
            write!(f, "    ")?;
            for (value, width) in values.iter().zip(&widths) {
                write!(f, "{:^width$}  ", value.isize(), width=width)?;
            }
            writeln!(f)?;
        }

        writeln!(f, "}}")?;
        Ok(())
    }
}

impl Labels {
    /// Get the number of entries/named values in a single label
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Get the names of the entries/columns in this set of labels
    pub fn names(&self) -> Vec<&str> {
        self.names.iter().map(|s| s.as_str()).collect()
    }

    /// Get the names of the entries/columns in this set of labels as
    /// C-compatible (null terminated) strings
    pub fn c_names(&self) -> &[ConstCString] {
        &self.names
    }

    /// Get the registered user data (this will be NULL if no data was
    /// registered)
    pub fn user_data(&self) -> *mut c_void {
        let guard = self.user_data.read().expect("poisoned lock");
        return guard.ptr;
    }

    /// Register user data for these Labels.
    ///
    /// The `user_data_delete` will be called with `user_data` when the Labels
    /// are dropped, and should free the memory associated with `user_data`.
    ///
    /// Any existing user data will be released (by calling the provided
    /// `user_data_delete` function) before overwriting with the new data.
    pub fn set_user_data(
        &self,
        user_data: *mut c_void,
        user_data_delete: Option<unsafe extern fn(*mut c_void)>,
    ) {
        let mut guard = self.user_data.write().expect("poisoned lock");
        *guard = UserData {
            ptr: user_data,
            delete: user_data_delete,
        };
    }

    /// Get the total number of entries in this set of labels
    pub fn count(&self) -> usize {
        if self.size() == 0 {
            return 0;
        } else {
            return self.values.len() / self.size();
        }
    }

    /// Check if this set of Labels is empty (contains no entry)
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Check whether the given `label` is part of this set of labels
    pub fn contains(&self, label: &[LabelValue]) -> bool {
        self.positions.contains_key(label)
    }

    /// Get the position (i.e. row index) of the given label in the full labels
    /// array, or None.
    pub fn position(&self, value: &[LabelValue]) -> Option<usize> {
        assert!(value.len() == self.size(), "invalid size of index in Labels::position");

        self.positions.get(value).copied()
    }

    /// Iterate over the entries in this set of labels
    pub fn iter(&self) -> Iter {
        debug_assert!(self.values.len() % self.names.len() == 0);
        return Iter {
            chunks: self.values.chunks_exact(self.names.len())
        };
    }

    /// Compute the union of two labels, and optionally the mapping from the
    /// position of entries in the inputs to positions of entries in the output.
    ///
    /// Mapping will be computed only if slices are not empty.
    #[allow(clippy::needless_range_loop)]
    pub fn union(&self, other: &Labels, first_mapping: &mut [i64], second_mapping: &mut [i64]) -> Result<Labels, Error> {
        if self.names != other.names {
            return Err(Error::InvalidParameter(
                "can not take the union of these Labels, they have different names".into()
            ));
        }

        let mut builder = LabelsBuilder {
            names: self.names.clone(),
            values: self.values.clone(),
            positions: self.positions.clone(),
        };

        if !first_mapping.is_empty() {
            assert!(first_mapping.len() == self.count());
            #[allow(clippy::cast_possible_wrap)]
            for i in 0..self.count() {
                first_mapping[i] = i as i64;
            }
        }

        for (i, entry) in other.iter().enumerate() {
            let entry = entry.iter().copied().map(Into::into).collect::<SmallVec<_>>();
            let position = builder.add_or_get_position(entry);

            if !second_mapping.is_empty() {
                let index = match position {
                    #[allow(clippy::cast_possible_wrap)]
                    Ok(index) | Err((index, _)) => {
                        index as i64
                    }
                };
                second_mapping[i] = index;
            }
        }

        return Ok(builder.finish());
    }

    /// Compute the intersection of two labels, and optionally the mapping from
    /// the position of entries in the inputs to positions of entries in the
    /// output.
    ///
    /// Mapping will be computed only if slices are not empty.
    pub fn intersection(&self, other: &Labels, first_mapping: &mut [i64], second_mapping: &mut [i64]) -> Result<Labels, Error> {
        if self.names != other.names {
            return Err(Error::InvalidParameter(
                "can not take the intersection of these Labels, they have different names".into()
            ));
        }

        // make `first` the Labels with fewest entries
        let (first, first_indexes, second, second_indexes) = if self.count() <= other.count() {
            (&self, first_mapping, &other, second_mapping)
        } else {
            (&other, second_mapping, &self, first_mapping)
        };

        if !first_indexes.is_empty() {
            assert!(first_indexes.len() == first.count());
            first_indexes.fill(-1);
        }

        if !second_indexes.is_empty() {
            assert!(second_indexes.len() == second.count());
            second_indexes.fill(-1);
        }

        let mut builder = LabelsBuilder::new(self.names()).expect("should be valid names");
        for (i, entry) in first.iter().enumerate() {
            if let Some(position) = second.position(entry) {
                #[allow(clippy::cast_possible_wrap)]
                let new_position = builder.count() as i64;
                builder.add(entry).expect("should not already exist");

                if !first_indexes.is_empty() {
                    first_indexes[i] = new_position;
                }

                if !second_indexes.is_empty() {
                    second_indexes[position] = new_position;
                }
            }
        }

        return Ok(builder.finish());
    }

    /// Select entries in these `Labels` that match the `selection`.
    ///
    /// The selection's names must be a subset of the name of these `Labels`
    /// names.
    ///
    /// All entries in `self` that match one of the entry in the selection for
    /// all the selection's dimension will be picked. Any entry in the selection
    /// but not in `self` will be ignored.
    ///
    /// On input, selected should have space for `self.count()` elements. On
    /// output, it will contain the indexes in `self` that match the selection.
    /// This function returns the number of selected entries, i.e. the number of
    /// valid indexes in `selected`.
    pub fn select(&self, selection: &Labels, selected: &mut [i64]) -> Result<usize, Error> {
        assert!(selected.len() == self.count());
        selected.fill(-1);

        let mut n_selected = 0;
        if selection.names == self.names {
            for entry in selection {
                #[allow(clippy::cast_possible_wrap)]
                if let Some(position) = self.position(entry) {
                    selected[n_selected] = position as i64;
                    n_selected += 1;
                }
            }
        } else {
            let mut dimensions_to_match = Vec::new();
            for name in &selection.names {
                let i = match self.names.iter().position(|n| n == name) {
                    Some(index) => index,
                    None => {
                        return Err(Error::InvalidParameter(format!(
                            "'{}' in selection is not part of these Labels", name.as_str()
                        )))
                    }
                };
                dimensions_to_match.push(i);
            }

            let mut candidate = vec![LabelValue::new(0); dimensions_to_match.len()];
            for (entry_i, entry) in self.iter().enumerate() {
                for (i, &d) in dimensions_to_match.iter().enumerate() {
                    candidate[i] = entry[d];
                }

                #[allow(clippy::cast_possible_wrap)]
                if selection.contains(&candidate) {
                    selected[n_selected] = entry_i as i64;
                    n_selected += 1;
                }
            }
        }

        return Ok(n_selected);
    }
}

/// iterator over `Labels` entries
pub struct Iter<'a> {
    chunks: std::slice::ChunksExact<'a, LabelValue>,
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a [LabelValue];

    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next()
    }
}

impl<'a> ExactSizeIterator for Iter<'a> {
    fn len(&self) -> usize {
        self.chunks.len()
    }
}

impl<'a> IntoIterator for &'a Labels {
    type IntoIter = Iter<'a>;
    type Item = &'a [LabelValue];
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl std::ops::Index<usize> for Labels {
    type Output = [LabelValue];
    fn index(&self, i: usize) -> &[LabelValue] {
        let start = i * self.size();
        let stop = (i + 1) * self.size();
        &self.values[start..stop]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_names() {
        let e = LabelsBuilder::new(vec!["not an ident"]).err().unwrap();
        assert_eq!(e.to_string(), "invalid parameter: all labels names must be valid identifiers, 'not an ident' is not");

        let e = LabelsBuilder::new(vec!["not", "there", "not"]).err().unwrap();
        assert_eq!(e.to_string(), "invalid parameter: labels names must be unique, got 'not' multiple times");
    }

    #[test]
    fn union() {
        let mut builder = LabelsBuilder::new(vec!["aa", "bb"]).unwrap();
        builder.add(&[0, 1]).unwrap();
        builder.add(&[1, 2]).unwrap();
        let first = builder.finish();

        let mut builder = LabelsBuilder::new(vec!["aa", "bb"]).unwrap();
        builder.add(&[2, 3]).unwrap();
        builder.add(&[1, 2]).unwrap();
        builder.add(&[4, 5]).unwrap();
        let second = builder.finish();

        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; second.count()];

        let union = first.union(&second, first_mapping, second_mapping).unwrap();
        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values, &[0, 1, 1, 2, 2, 3, 4, 5]);
        assert_eq!(first_mapping, &[0, 1]);
        assert_eq!(second_mapping, &[2, 1, 3]);

        let first_mapping = &mut vec![0; second.count()];
        let second_mapping = &mut vec![0; first.count()];

        let union = second.union(&first, first_mapping, second_mapping).unwrap();
        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values, &[2, 3, 1, 2, 4, 5, 0, 1]);
        assert_eq!(first_mapping, &[0, 1, 2]);
        assert_eq!(second_mapping, &[3, 1]);

        let labels = LabelsBuilder::new(vec!["aa"]).unwrap().finish();
        let err = first.union(&labels, &mut [], &mut []).unwrap_err();
        assert_eq!(
            format!("{}", err),
            "invalid parameter: can not take the union of these Labels, they have different names"
        );

        // Take the union with an empty set of labels
        let empty = LabelsBuilder::new(vec!["aa", "bb"]).unwrap().finish();
        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; empty.count()];

        let union = first.union(&empty, first_mapping, second_mapping).unwrap();
        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values, &[0, 1, 1, 2]);
        assert_eq!(first_mapping, &[0, 1]);
        assert_eq!(second_mapping, &[]);
    }

    #[test]
    fn intersection() {
        let mut builder = LabelsBuilder::new(vec!["aa", "bb"]).unwrap();
        builder.add(&[0, 1]).unwrap();
        builder.add(&[1, 2]).unwrap();
        let first = builder.finish();

        let mut builder = LabelsBuilder::new(vec!["aa", "bb"]).unwrap();
        builder.add(&[2, 3]).unwrap();
        builder.add(&[1, 2]).unwrap();
        builder.add(&[4, 5]).unwrap();
        let second = builder.finish();

        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; second.count()];

        let intersection = first.intersection(&second, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.names(), ["aa", "bb"]);
        assert_eq!(intersection.values, &[1, 2]);
        assert_eq!(first_mapping, &[-1, 0]);
        assert_eq!(second_mapping, &[-1, 0, -1]);

        let first_mapping = &mut vec![0; second.count()];
        let second_mapping = &mut vec![0; first.count()];

        let intersection = second.intersection(&first, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.names(), ["aa", "bb"]);
        assert_eq!(intersection.values, &[1, 2]);
        assert_eq!(first_mapping, &[-1, 0, -1]);
        assert_eq!(second_mapping, &[-1, 0]);

        let labels = LabelsBuilder::new(vec!["aa"]).unwrap().finish();
        let err = first.intersection(&labels, &mut [], &mut []).unwrap_err();
        assert_eq!(
            format!("{}", err),
            "invalid parameter: can not take the intersection of these Labels, they have different names"
        );

        // Take the intersection with an empty set of labels
        let empty = LabelsBuilder::new(vec!["aa", "bb"]).unwrap().finish();
        let first_mapping = &mut vec![0; first.count()];
        let second_mapping = &mut vec![0; empty.count()];

        let intersection = first.intersection(&empty, first_mapping, second_mapping).unwrap();
        assert_eq!(intersection.names(), ["aa", "bb"]);
        assert_eq!(intersection.count(), 0);
        assert_eq!(first_mapping, &[-1, -1]);
        assert_eq!(second_mapping, &[]);
    }

    #[test]
    fn marker_traits() {
        // ensure Arc<Labels> is Send and Sync, assuming the user data is
        fn use_send(_: impl Send) {}
        fn use_sync(_: impl Sync) {}

        let mut builder = LabelsBuilder::new(vec!["aa", "bb"]).unwrap();
        builder.add(&[0, 1]).unwrap();
        builder.add(&[1, 2]).unwrap();
        let labels = std::sync::Arc::new(builder.finish());

        use_send(labels.clone());
        use_sync(labels);
    }
}

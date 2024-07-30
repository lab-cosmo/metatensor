use std:: ffi::CStr;
use std::ffi::CString;
use std::collections::BTreeSet;
use std::iter::FusedIterator;

use smallvec::SmallVec;

use crate::c_api::mts_labels_t;
use crate::errors::{Error, check_status};

/// A single value inside a label.
///
/// This is represented as a 32-bit signed integer, with a couple of helper
/// function to get its value as usize/isize.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct LabelValue(i32);

impl PartialEq<i32> for LabelValue {
    #[inline]
    fn eq(&self, other: &i32) -> bool {
        self.0 == *other
    }
}

impl PartialEq<LabelValue> for i32 {
    #[inline]
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
    #[inline]
    fn from(value: u32) -> LabelValue {
        assert!(value < i32::MAX as u32);
        LabelValue(value as i32)
    }
}

impl From<i32> for LabelValue {
    #[inline]
    fn from(value: i32) -> LabelValue {
        LabelValue(value)
    }
}

#[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
impl From<usize> for LabelValue {
    #[inline]
    fn from(value: usize) -> LabelValue {
        assert!(value < i32::MAX as usize);
        LabelValue(value as i32)
    }
}

#[allow(clippy::cast_possible_truncation)]
impl From<isize> for LabelValue {
    #[inline]
    fn from(value: isize) -> LabelValue {
        assert!(value < i32::MAX as isize && value > i32::MIN as isize);
        LabelValue(value as i32)
    }
}

impl LabelValue {
    /// Create a `LabelValue` with the given `value`
    #[inline]
    pub fn new(value: i32) -> LabelValue {
        LabelValue(value)
    }

    /// Get the integer value of this `LabelValue` as a usize
    #[inline]
    #[allow(clippy::cast_sign_loss)]
    pub fn usize(self) -> usize {
        debug_assert!(self.0 >= 0);
        self.0 as usize
    }

    /// Get the integer value of this `LabelValue` as an isize
    #[inline]
    pub fn isize(self) -> isize {
        self.0 as isize
    }

    /// Get the integer value of this `LabelValue` as an i32
    #[inline]
    pub fn i32(self) -> i32 {
        self.0
    }
}

/// A set of labels used to carry metadata associated with a tensor map.
///
/// This is similar to a list of named tuples, but stored as a 2D array of shape
/// `(labels.count(), labels.size())`, with a of set names associated with the
/// columns of this array. Each row/entry in this array is unique, and they are
/// often (but not always) sorted in  lexicographic order.
///
/// The main way to construct a new set of labels is to use a `LabelsBuilder`.
///
/// Labels are internally reference counted and immutable, so cloning a `Labels`
/// should be a cheap operation.
pub struct Labels {
    pub(crate) raw: mts_labels_t,
}

// Labels can be sent to other thread safely since mts_labels_t uses an
// `Arc<metatensor_core::Labels>`, so freeing them from another thread is fine
unsafe impl Send for Labels {}
// &Labels can be sent to other thread safely since there is no un-synchronized
// interior mutability (`user_data` is protected by RwLock).
unsafe impl Sync for Labels {}

impl std::fmt::Debug for Labels {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        pretty_print_labels(self, "", f)
    }
}

/// Helper function to print labels in a Debug mode
pub(crate) fn pretty_print_labels(
    labels: &Labels,
    offset: &str,
    f: &mut std::fmt::Formatter<'_>
) -> std::fmt::Result {
    let names = labels.names();

    writeln!(f, "Labels @ {:p} {{", labels.raw.internal_ptr_)?;
    writeln!(f, "{}    {}", offset, names.join(", "))?;

    let widths = names.iter().map(|s| s.len()).collect::<Vec<_>>();
    for values in labels {
        write!(f, "{}    ", offset)?;
        for (value, width) in values.iter().zip(&widths) {
            write!(f, "{:^width$}  ", value.isize(), width=width)?;
        }
        writeln!(f)?;
    }

    writeln!(f, "{}}}", offset)
}

impl Clone for Labels {
    #[inline]
    fn clone(&self) -> Self {
        let mut clone = mts_labels_t::null();
        unsafe {
            check_status(crate::c_api::mts_labels_clone(self.raw, &mut clone)).expect("failed to clone Labels");
        }

        return unsafe { Labels::from_raw(clone) };
    }
}

impl std::ops::Drop for Labels {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        unsafe {
            crate::c_api::mts_labels_free(&mut self.raw);
        }
    }
}

impl Labels {
    /// Create a new set of Labels with the given names and values.
    ///
    /// This is a convenience function replacing the manual use of
    /// `LabelsBuilder`. If you need more flexibility or incremental `Labels`
    /// construction, use `LabelsBuilder`.
    ///
    /// # Panics
    ///
    /// If the set of names is not valid, or any of the value is duplicated
    #[inline]
    pub fn new<T, const N: usize>(names: [&str; N], values: &[[T; N]]) -> Labels
        where T: Copy + Into<LabelValue>
    {
        let mut builder = LabelsBuilder::new(names.to_vec());
        for entry in values {
            builder.add(entry);
        }
        return builder.finish();
    }

    /// Create a set of `Labels` with the given names, containing no entries.
    #[inline]
    pub fn empty(names: Vec<&str>) -> Labels {
        return LabelsBuilder::new(names).finish()
    }

    /// Create a set of `Labels` containing a single entry, to be used when
    /// there is no relevant information to store.
    #[inline]
    pub fn single() -> Labels {
        let mut builder = LabelsBuilder::new(vec!["_"]);
        builder.add(&[0]);
        return builder.finish();
    }

    /// Load `Labels` from the file at `path`
    ///
    /// This is a convenience function calling [`crate::io::load_labels`]
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Labels, Error> {
        return crate::io::load_labels(path);
    }

    /// Load a `TensorMap` from an in-memory buffer
    ///
    /// This is a convenience function calling [`crate::io::load_buffer`]
    pub fn load_buffer(buffer: &[u8]) -> Result<Labels, Error> {
        return crate::io::load_labels_buffer(buffer);
    }

    /// Save the given tensor to the file at `path`
    ///
    /// This is a convenience function calling [`crate::io::save`]
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        return crate::io::save_labels(path, self);
    }

    /// Save the given tensor to an in-memory buffer
    ///
    /// This is a convenience function calling [`crate::io::save_buffer`]
    pub fn save_buffer(&self, buffer: &mut Vec<u8>) -> Result<(), Error> {
        return crate::io::save_labels_buffer(self, buffer);
    }

    /// Get the number of entries/named values in a single label
    #[inline]
    pub fn size(&self) -> usize {
        self.raw.size
    }

    /// Get the names of the entries/columns in this set of labels
    #[inline]
    pub fn names(&self) -> Vec<&str> {
        if self.raw.size == 0 {
            return Vec::new();
        } else {
            unsafe {
                let names = std::slice::from_raw_parts(self.raw.names, self.raw.size);
                return names.iter()
                            .map(|&ptr| CStr::from_ptr(ptr).to_str().expect("invalid UTF8"))
                            .collect();
            }
        }
    }

    /// Get the total number of entries in this set of labels
    #[inline]
    pub fn count(&self) -> usize {
        return self.raw.count;
    }

    /// Check if this set of Labels is empty (contains no entry)
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Check whether the given `label` is part of this set of labels
    #[inline]
    pub fn contains(&self, label: &[LabelValue]) -> bool {
        return self.position(label).is_some();
    }

    /// Get the position (i.e. row index) of the given label in the full labels
    /// array, or None.
    #[inline]
    pub fn position(&self, value: &[LabelValue]) -> Option<usize> {
        assert!(value.len() == self.size(), "invalid size of index in Labels::position");

        let mut result = 0;
        unsafe {
            check_status(crate::c_api::mts_labels_position(
                self.raw,
                value.as_ptr().cast(),
                value.len(),
                &mut result,
            )).expect("failed to check label position");
        }

        return result.try_into().ok();
    }

    /// Take the union of `self` with `other`.
    ///
    /// If requested, this function can also give the positions in the union
    /// where each entry of the input `Labels` ended up.
    ///
    /// If `first_mapping` (respectively `second_mapping`) is `Some`, it should
    /// contain a slice of length `self.count()` (respectively `other.count()`)
    /// that will be filled with the position of the entries in `self`
    /// (respectively `other`) in the union.
    #[inline]
    pub fn union(
        &self,
        other: &Labels,
        first_mapping: Option<&mut [i64]>,
        second_mapping: Option<&mut [i64]>,
    ) -> Result<Labels, Error> {
        let mut output = mts_labels_t::null();
        let (first_mapping, first_mapping_count) = if let Some(m) = first_mapping {
            (m.as_mut_ptr(), m.len())
        } else {
            (std::ptr::null_mut(), 0)
        };

        let (second_mapping, second_mapping_count) = if let Some(m) = second_mapping {
            (m.as_mut_ptr(), m.len())
        } else {
            (std::ptr::null_mut(), 0)
        };

        unsafe {
            check_status(crate::c_api::mts_labels_union(
                self.raw,
                other.raw,
                &mut output,
                first_mapping,
                first_mapping_count,
                second_mapping,
                second_mapping_count,
            ))?;

            return Ok(Labels::from_raw(output));
        }
    }

    /// Take the intersection of self with `other`.
    ///
    /// If requested, this function can also give the positions in the
    /// intersection where each entry of the input `Labels` ended up.
    ///
    /// If `first_mapping` (respectively `second_mapping`) is `Some`, it should
    /// contain a slice of length `self.count()` (respectively `other.count()`)
    /// that will be filled by with the position of the entries in `self`
    /// (respectively `other`) in the intersection. If an entry in `self` or
    /// `other` are not used in the intersection, the mapping for this entry
    /// will be set to `-1`.
    #[inline]
    pub fn intersection(
        &self,
        other: &Labels,
        first_mapping: Option<&mut [i64]>,
        second_mapping: Option<&mut [i64]>,
    ) -> Result<Labels, Error> {
        let mut output = mts_labels_t::null();
        let (first_mapping, first_mapping_count) = if let Some(m) = first_mapping {
            (m.as_mut_ptr(), m.len())
        } else {
            (std::ptr::null_mut(), 0)
        };

        let (second_mapping, second_mapping_count) = if let Some(m) = second_mapping {
            (m.as_mut_ptr(), m.len())
        } else {
            (std::ptr::null_mut(), 0)
        };

        unsafe {
            check_status(crate::c_api::mts_labels_intersection(
                self.raw,
                other.raw,
                &mut output,
                first_mapping,
                first_mapping_count,
                second_mapping,
                second_mapping_count,
            ))?;

            return Ok(Labels::from_raw(output));
        }
    }

    /// Iterate over the entries in this set of labels
    #[inline]
    pub fn iter(&self) -> LabelsIter<'_> {
        return LabelsIter {
            chunks: self.values().chunks_exact(self.raw.size)
        };
    }

    /// Iterate over the entries in this set of labels in parallel
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_iter(&self) -> LabelsParIter<'_> {
        use rayon::prelude::*;
        return LabelsParIter {
            chunks: self.values().par_chunks_exact(self.raw.size)
        };
    }

    /// Iterate over the entries in this set of labels as fixed-size arrays
    #[inline]
    pub fn iter_fixed_size<const N: usize>(&self) -> LabelsFixedSizeIter<N> {
        assert!(N == self.size(),
            "wrong label size in `iter_fixed_size`: the entries contains {} element \
            but this function was called with size of {}",
            self.size(), N
        );

        return LabelsFixedSizeIter {
            values: self.values()
        };
    }

    /// Select entries in these `Labels` that match the `selection`.
    ///
    /// The selection's names must be a subset of the names of these labels.
    ///
    /// All entries in these `Labels` that match one of the entry in the
    /// `selection` for all the selection's dimension will be picked. Any entry
    /// in the `selection` but not in these `Labels` will be ignored.
    pub fn select(&self, selection: &Labels) -> Result<Vec<i64>, Error> {
        let mut selected = vec![-1; self.count()];
        let mut selected_count = selected.len();

        unsafe {
            check_status(crate::c_api::mts_labels_select(
                self.as_mts_labels_t(),
                selection.as_mts_labels_t(),
                selected.as_mut_ptr(),
                &mut selected_count
            ))?;
        }

        selected.resize(selected_count, 0);

        return Ok(selected);
    }

    pub(crate) fn values(&self) -> &[LabelValue] {
        if self.count() == 0 || self.size() == 0 {
            return &[]
        } else {
            unsafe {
                std::slice::from_raw_parts(self.raw.values.cast(), self.count() * self.size())
            }
        }
    }
}

impl Labels {
    /// Get the underlying `mts_labels_t`
    pub(crate) fn as_mts_labels_t(&self) -> mts_labels_t {
        return self.raw;
    }

    /// Create a new set of `Labels` from a raw `mts_labels_t`.
    ///
    /// This function takes ownership of the `mts_labels_t` and will call
    /// `mts_labels_free` on it.
    ///
    /// # Safety
    ///
    /// The raw `mts_labels_t` must have been returned by one of the function
    /// returning `mts_labels_t` in metatensor-core
    #[inline]
    pub unsafe fn from_raw(raw: mts_labels_t) -> Labels {
        assert!(!raw.internal_ptr_.is_null(), "expected mts_labels_t.internal_ptr_ to not be NULL");
        Labels {
            raw: raw,
        }
    }
}

impl std::cmp::PartialEq<Labels> for Labels {
    #[inline]
    fn eq(&self, other: &Labels) -> bool {
        self.names() == other.names() && self.values() == other.values()
    }
}

impl std::ops::Index<usize> for Labels {
    type Output = [LabelValue];

    #[inline]
    fn index(&self, i: usize) -> &[LabelValue] {
        let start = i * self.size();
        let stop = (i + 1) * self.size();
        &self.values()[start..stop]
    }
}

/// Iterator over [`Labels`] entries
#[derive(Debug, Clone)]
pub struct LabelsIter<'a> {
    chunks: std::slice::ChunksExact<'a, LabelValue>,
}

impl<'a> Iterator for LabelsIter<'a> {
    type Item = &'a [LabelValue];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.chunks.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.chunks.size_hint()
    }
}

impl<'a> ExactSizeIterator for LabelsIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.chunks.len()
    }
}

impl<'a> FusedIterator for LabelsIter<'a> {}

impl<'a> IntoIterator for &'a Labels {
    type IntoIter = LabelsIter<'a>;
    type Item = &'a [LabelValue];

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Parallel iterator over entries in a set of [`Labels`]
#[cfg(feature = "rayon")]
#[derive(Debug, Clone)]
pub struct LabelsParIter<'a> {
    chunks: rayon::slice::ChunksExact<'a, LabelValue>,
}

#[cfg(feature = "rayon")]
impl<'a> rayon::iter::ParallelIterator for LabelsParIter<'a> {
    type Item = &'a [LabelValue];

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item> {
        self.chunks.drive_unindexed(consumer)
    }
}

#[cfg(feature = "rayon")]
impl<'a> rayon::iter::IndexedParallelIterator for LabelsParIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        self.chunks.drive(consumer)
    }

    #[inline]
    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        self.chunks.with_producer(callback)
    }
}

/// Iterator over entries in a set of [`Labels`] as fixed size arrays
#[derive(Debug, Clone)]
pub struct LabelsFixedSizeIter<'a, const N: usize> {
    values: &'a [LabelValue],
}

impl<'a, const N: usize> Iterator for LabelsFixedSizeIter<'a, N> {
    type Item = &'a [LabelValue; N];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.values.is_empty() {
            return None
        }

        let (value, rest) = self.values.split_at(N);
        self.values = rest;
        return Some(value.try_into().expect("wrong size in FixedSizeIter::next"));
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, const N: usize> ExactSizeIterator for LabelsFixedSizeIter<'a, N> {
    #[inline]
    fn len(&self) -> usize {
        self.values.len() / N
    }
}

/// Builder for [`Labels`]
#[derive(Debug, Clone)]
pub struct LabelsBuilder {
    // cf `Labels` for the documentation of the fields
    names: Vec<String>,
    values: Vec<LabelValue>,
}

impl LabelsBuilder {
    /// Create a new empty `LabelsBuilder` with the given `names`
    #[inline]
    pub fn new(names: Vec<&str>) -> LabelsBuilder {
        let n_unique_names = names.iter().collect::<BTreeSet<_>>().len();
        assert!(n_unique_names == names.len(), "invalid labels: the same name is used multiple times");

        LabelsBuilder {
            names: names.into_iter().map(|s| s.into()).collect(),
            values: Vec::new(),
        }
    }

    /// Reserve space for `additional` other entries in the labels.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional * self.names.len());
    }

    /// Get the number of labels in a single value
    #[inline]
    pub fn size(&self) -> usize {
        self.names.len()
    }

    /// Add a single `entry` to this set of labels.
    ///
    /// This function will panic when attempting to add the same `label` more
    /// than once.
    #[inline]
    pub fn add<T>(&mut self, entry: &[T]) where T: Copy + Into<LabelValue> {
        assert_eq!(
            self.size(), entry.len(),
            "wrong size for added label: got {}, but expected {}",
            entry.len(), self.size()
        );

        // SmallVec allows us to convert everything to `LabelValue` without
        // requiring an extra heap allocation
        let entry = entry.iter().copied().map(Into::into).collect::<SmallVec<[LabelValue; 16]>>();
        self.values.extend(&entry);
    }

    /// Finish building the `Labels`
    #[inline]
    pub fn finish(self) -> Labels {
        let mut raw_names = Vec::new();
        let mut raw_names_ptr = Vec::new();

        let mut raw_labels = if self.names.is_empty() {
            assert!(self.values.is_empty());
            mts_labels_t::null()
        } else {
            for name in &self.names {
                let name = CString::new(&**name).expect("name contains a NULL byte");
                raw_names_ptr.push(name.as_ptr());
                raw_names.push(name);
            }

            mts_labels_t {
                internal_ptr_: std::ptr::null_mut(),
                names: raw_names_ptr.as_ptr(),
                values: self.values.as_ptr().cast(),
                size: self.size(),
                count: self.values.len() / self.size(),
            }
        };

        unsafe {
            check_status(
                crate::c_api::mts_labels_create(&mut raw_labels)
            ).expect("invalid labels?");
        }

        return unsafe { Labels::from_raw(raw_labels) };
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(&[2, 3]);
        builder.add(&[1, 243]);
        builder.add(&[-4, -2413]);

        let idx = builder.finish();
        assert_eq!(idx.names(), &["foo", "bar"]);
        assert_eq!(idx.size(), 2);
        assert_eq!(idx.count(), 3);

        assert_eq!(idx[0], [2, 3]);
        assert_eq!(idx[1], [1, 243]);
        assert_eq!(idx[2], [-4, -2413]);

        let builder = LabelsBuilder::new(vec![]);
        let labels = builder.finish();
        assert_eq!(labels.size(), 0);
        assert_eq!(labels.count(), 0);
    }

    #[test]
    fn direct_construct() {
        let labels = Labels::new(
            ["foo", "bar"],
            &[
                [2, 3],
                [1, 243],
                [-4, -2413],
            ]
        );

        assert_eq!(labels.names(), &["foo", "bar"]);
        assert_eq!(labels.size(), 2);
        assert_eq!(labels.count(), 3);

        assert_eq!(labels[0], [2, 3]);
        assert_eq!(labels[1], [1, 243]);
        assert_eq!(labels[2], [-4, -2413]);
    }

    #[test]
    fn labels_iter() {
        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(&[2, 3]);
        builder.add(&[1, 2]);
        builder.add(&[4, 3]);

        let idx = builder.finish();
        let mut iter = idx.iter();
        assert_eq!(iter.len(), 3);

        assert_eq!(iter.next().unwrap(), &[2, 3]);
        assert_eq!(iter.next().unwrap(), &[1, 2]);
        assert_eq!(iter.next().unwrap(), &[4, 3]);
        assert_eq!(iter.next(), None);
    }

    #[test]
    #[should_panic(expected = "'33 bar' is not a valid label name")]
    fn invalid_label_name() {
        LabelsBuilder::new(vec!["foo", "33 bar"]).finish();
    }

    #[test]
    #[should_panic(expected = "invalid labels: the same name is used multiple times")]
    fn duplicated_label_name() {
        LabelsBuilder::new(vec!["foo", "bar", "foo"]).finish();
    }

    #[test]
    #[should_panic(expected = "can not have the same label value multiple time: [0, 1] is already present at position 0")]
    fn duplicated_label_value() {
        let mut builder = LabelsBuilder::new(vec!["foo", "bar"]);
        builder.add(&[0, 1]);
        builder.add(&[0, 1]);
        builder.finish();
    }

    #[test]
    fn single_label() {
        let labels = Labels::single();
        assert_eq!(labels.names(), &["_"]);
        assert_eq!(labels.size(), 1);
        assert_eq!(labels.count(), 1);
    }

    #[test]
    fn indexing() {
        let labels = Labels::new(
            ["foo", "bar"],
            &[
                [2, 3],
                [1, 243],
                [-4, -2413],
            ]
        );

        assert_eq!(labels[1], [1, 243]);
        assert_eq!(labels[2], [-4, -2413]);
    }

    #[test]
    fn iter() {
        let labels = Labels::new(
            ["foo", "bar"],
            &[
                [2, 3],
                [1, 243],
                [-4, -2413],
            ]
        );

        let mut iter = labels.iter();

        assert_eq!(iter.next().unwrap(), &[2, 3]);
        assert_eq!(iter.next().unwrap(), &[1, 243]);
        assert_eq!(iter.next().unwrap(), &[-4, -2413]);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn debug() {
        let labels = Labels::new(
            ["foo", "bar"],
            &[
                [2, 3],
                [1, 243],
                [-4, -2413],
            ]
        );

        let expected = format!(
            "Labels @ {:p} {{\n    foo, bar\n     2    3   \n     1   243  \n    -4   -2413  \n}}\n",
            labels.as_mts_labels_t().internal_ptr_
        );
        assert_eq!(format!("{:?}", labels), expected);
    }

    #[test]
    fn union() {
        let first = Labels::new(["aa", "bb"], &[[0, 1], [1, 2]]);
        let second = Labels::new(["aa", "bb"], &[[2, 3], [1, 2], [4, 5]]);

        let mut first_mapping = vec![0; first.count()];
        let mut second_mapping = vec![0; second.count()];
        let union = first.union(&second, Some(&mut first_mapping), Some(&mut second_mapping)).unwrap();

        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values(), [0, 1, 1, 2, 2, 3, 4, 5]);

        assert_eq!(first_mapping, [0, 1]);
        assert_eq!(second_mapping, [2, 1, 3]);
    }

    #[test]
    fn intersection() {
        let first = Labels::new(["aa", "bb"], &[[0, 1], [1, 2]]);
        let second = Labels::new(["aa", "bb"], &[[2, 3], [1, 2], [4, 5]]);

        let mut first_mapping = vec![0_i64; first.count()];
        let mut second_mapping = vec![0_i64; second.count()];
        let union = first.intersection(&second, Some(&mut first_mapping), Some(&mut second_mapping)).unwrap();

        assert_eq!(union.names(), ["aa", "bb"]);
        assert_eq!(union.values(), [1, 2]);

        assert_eq!(first_mapping, [-1, 0]);
        assert_eq!(second_mapping, [-1, 0, -1]);
    }

    #[test]
    fn selection() {
        // selection with a subset of names
        let labels = Labels::new(["aa", "bb"], &[[1, 1], [1, 2], [3, 2], [2, 1]]);
        let selection = Labels::new(["aa"], &[[1], [2], [5]]);

        let selected = labels.select(&selection).unwrap();
        assert_eq!(selected, [0, 1, 3]);

        // selection with the same names
        let selection = Labels::new(["aa", "bb"], &[[1, 1], [2, 1], [5, 1], [1, 2]]);
        let selected = labels.select(&selection).unwrap();
        assert_eq!(selected, [0, 3, 1]);

        // empty selection
        let selection = Labels::empty(vec!["aa"]);
        let selected = labels.select(&selection).unwrap();
        assert_eq!(selected, []);

        // invalid selection names
        let selection = Labels::empty(vec!["aaa"]);
        let err = labels.select(&selection).unwrap_err();
        assert_eq!(err.message,
            "invalid parameter: 'aaa' in selection is not part of these Labels"
        );
    }
}

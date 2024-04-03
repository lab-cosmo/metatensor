use std::ffi::CString;
use std::iter::FusedIterator;

use crate::block::TensorBlockRefMut;
use crate::c_api::{mts_tensormap_t, mts_labels_t};

use crate::errors::{check_status, check_ptr};
use crate::{Error, TensorBlock, TensorBlockRef, Labels, LabelValue};

/// [`TensorMap`] is the main user-facing struct of this library, and can
/// store any kind of data used in atomistic machine learning.
///
/// A tensor map contains a list of `TensorBlock`s, each one associated with a
/// key in the form of a single `Labels` entry.
///
/// It provides functions to merge blocks together by moving some of these keys
/// to the samples or properties labels of the blocks, transforming the sparse
/// representation of the data to a dense one.
pub struct TensorMap {
    pub(crate) ptr: *mut mts_tensormap_t,
    /// cache for the keys labels
    keys: Labels,
}

// SAFETY: Send is fine since we can free a TensorMap from any thread
unsafe impl Send for TensorMap {}
// SAFETY: Sync is fine since there is no internal mutability in TensorMap
unsafe impl Sync for TensorMap {}

impl std::fmt::Debug for TensorMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::labels::pretty_print_labels;
        writeln!(f, "Tensormap @ {:p} {{", self.ptr)?;

        write!(f, "    keys: ")?;
        pretty_print_labels(self.keys(), "    ", f)?;
        writeln!(f, "}}")
    }
}

impl std::ops::Drop for TensorMap {
    #[allow(unused_must_use)]
    fn drop(&mut self) {
        unsafe {
            crate::c_api::mts_tensormap_free(self.ptr);
        }
    }
}

impl TensorMap {
    /// Create a new `TensorMap` with the given keys and blocks.
    ///
    /// The number of keys must match the number of blocks, and all the blocks
    /// must contain the same kind of data (same labels names, same gradients
    /// defined on all blocks).
    #[allow(clippy::needless_pass_by_value)]
    #[inline]
    pub fn new(keys: Labels, mut blocks: Vec<TensorBlock>) -> Result<TensorMap, Error> {
        let ptr = unsafe {
            crate::c_api::mts_tensormap(
                keys.as_mts_labels_t(),
                // this cast is fine because TensorBlock is `repr(transparent)`
                // to a `*mut mts_block_t` (through `TensorBlockRefMut`, and
                // `TensorBlockRef`).
                blocks.as_mut_ptr().cast::<*mut crate::c_api::mts_block_t>(),
                blocks.len()
            )
        };

        for block in blocks {
            // we give ownership of the blocks to the new tensormap, so we
            // should not free them again from Rust
            std::mem::forget(block);
        }

        check_ptr(ptr)?;

        return Ok(unsafe { TensorMap::from_raw(ptr) });
    }

    /// Create a new `TensorMap` from a raw pointer.
    ///
    /// This function takes ownership of the pointer, and will call
    /// `mts_tensormap_free` on it when the `TensorMap` goes out of scope.
    ///
    /// # Safety
    ///
    /// The pointer must be non-null and created by `mts_tensormap` or
    /// `TensorMap::into_raw`.
    pub unsafe fn from_raw(ptr: *mut mts_tensormap_t) -> TensorMap {
        assert!(!ptr.is_null());

        let mut keys = mts_labels_t::null();
        check_status(crate::c_api::mts_tensormap_keys(
            ptr,
            &mut keys
        )).expect("failed to get the keys");

        let keys = Labels::from_raw(keys);

        return TensorMap {
            ptr,
            keys
        };
    }

    /// Extract the underlying raw pointer.
    ///
    /// The pointer should be passed back to `TensorMap::from_raw` or
    /// `mts_tensormap_free` to release the memory corresponding to this
    /// `TensorMap`.
    pub fn into_raw(mut map: TensorMap) -> *mut mts_tensormap_t {
        let ptr = map.ptr;
        map.ptr = std::ptr::null_mut();
        return ptr;
    }

    /// Clone this `TensorMap`, cloning all the data and metadata contained inside.
    ///
    /// This can fail if the external data held inside an `mts_array_t` can not
    /// be cloned.
    #[inline]
    pub fn try_clone(&self) -> Result<TensorMap, Error> {
        let ptr = unsafe {
            crate::c_api::mts_tensormap_copy(self.ptr)
        };
        crate::errors::check_ptr(ptr)?;

        return Ok(unsafe { TensorMap::from_raw(ptr) });
    }

    /// Load a `TensorMap` from the file at `path`
    ///
    /// This is a convenience function calling [`crate::io::load`]
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<TensorMap, Error> {
        return crate::io::load(path);
    }

    /// Load a `TensorMap` from an in-memory buffer
    ///
    /// This is a convenience function calling [`crate::io::load_buffer`]
    pub fn load_buffer(buffer: &[u8]) -> Result<TensorMap, Error> {
        return crate::io::load_buffer(buffer);
    }

    /// Save the given tensor to the file at `path`
    ///
    /// This is a convenience function calling [`crate::io::save`]
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        return crate::io::save(path, self);
    }

    /// Save the given tensor to an in-memory buffer
    ///
    /// This is a convenience function calling [`crate::io::save_buffer`]
    pub fn save_buffer(&self, buffer: &mut Vec<u8>) -> Result<(), Error> {
        return crate::io::save_buffer(self, buffer);
    }

    /// Get the keys defined in this `TensorMap`
    #[inline]
    pub fn keys(&self) -> &Labels {
        &self.keys
    }

    /// Get a reference to the block at the given `index` in this `TensorMap`
    ///
    /// # Panics
    ///
    /// If the index is out of bounds
    #[inline]
    pub fn block_by_id(&self, index: usize) -> TensorBlockRef<'_> {

        let mut block = std::ptr::null_mut();
        unsafe {
            check_status(crate::c_api::mts_tensormap_block_by_id(
                self.ptr,
                &mut block,
                index,
            )).expect("failed to get a block");
        }

        return unsafe { TensorBlockRef::from_raw(block) }
    }

    /// Get a mutable reference to the block at the given `index` in this `TensorMap`
    ///
    /// # Panics
    ///
    /// If the index is out of bounds
    #[inline]
    pub fn block_mut_by_id(&mut self, index: usize) -> TensorBlockRefMut<'_> {
        return unsafe { TensorMap::raw_block_mut_by_id(self.ptr, index) };
    }

    /// Implementation of `block_mut_by_id` which does not borrow the
    /// `mts_tensormap_t` pointer.
    ///
    /// This is used to provide references to multiple blocks at the same time
    /// in the iterators.
    ///
    /// # Safety
    ///
    /// This should be called with a valid `mts_tensormap_t`, and the lifetime
    /// `'a` should be properly constrained to the lifetime of the owner of
    /// `ptr`.
    #[inline]
    unsafe fn raw_block_mut_by_id<'a>(ptr: *mut mts_tensormap_t, index: usize) -> TensorBlockRefMut<'a> {
        let mut block = std::ptr::null_mut();

        check_status(crate::c_api::mts_tensormap_block_by_id(
            ptr,
            &mut block,
            index,
        )).expect("failed to get a block");

        return TensorBlockRefMut::from_raw(block);
    }

    /// Get the index of blocks matching the given selection.
    ///
    /// The selection must contains a single entry, defining the requested key
    /// or keys. If the selection contains only a subset of the dimensions of the
    /// keys, there can be multiple matching blocks.
    #[inline]
    pub fn blocks_matching(&self, selection: &Labels) -> Result<Vec<usize>, Error> {
        let mut indexes = vec![0; self.keys().count()];
        let mut matching = indexes.len();
        unsafe {
            check_status(crate::c_api::mts_tensormap_blocks_matching(
                self.ptr,
                indexes.as_mut_ptr(),
                &mut matching,
                selection.as_mts_labels_t(),
            ))?;
        }
        indexes.resize(matching, 0);

        return Ok(indexes);
    }

    /// Get the index of the single block matching the given selection.
    ///
    /// This function is similar to [`TensorMap::blocks_matching`], but also
    /// returns an error if more than one block matches the selection.
    #[inline]
    pub fn block_matching(&self, selection: &Labels) -> Result<usize, Error> {
        let matching = self.blocks_matching(selection)?;
        if matching.len() != 1 {
            let selection_str = selection.names()
                .iter().zip(&selection[0])
                .map(|(name, value)| format!("{} = {}", name, value))
                .collect::<Vec<_>>()
                .join(", ");


            if matching.is_empty() {
                return Err(Error {
                    code: None,
                    message: format!(
                        "no blocks matched the selection ({})",
                        selection_str
                    ),
                });
            } else {
                return Err(Error {
                    code: None,
                    message: format!(
                        "{} blocks matched the selection ({}), expected only one",
                        matching.len(),
                        selection_str
                    ),
                });
            }
        }

        return Ok(matching[0])
    }

    /// Get a reference to the block matching the given selection.
    ///
    /// This function uses [`TensorMap::blocks_matching`] under the hood to find
    /// the matching block.
    #[inline]
    pub fn block(&self, selection: &Labels) -> Result<TensorBlockRef<'_>, Error> {
        let id = self.block_matching(selection)?;
        return Ok(self.block_by_id(id));
    }

    /// Get a reference to every blocks in this `TensorMap`
    #[inline]
    pub fn blocks(&self) -> Vec<TensorBlockRef<'_>> {
        let mut blocks = Vec::new();
        for i in 0..self.keys().count() {
            blocks.push(self.block_by_id(i));
        }
        return blocks;
    }

    /// Get a mutable reference to every blocks in this `TensorMap`
    #[inline]
    pub fn blocks_mut(&mut self) -> Vec<TensorBlockRefMut<'_>> {
        let mut blocks = Vec::new();
        for i in 0..self.keys().count() {
            blocks.push(unsafe { TensorMap::raw_block_mut_by_id(self.ptr, i) });
        }
        return blocks;
    }

    /// Merge blocks with the same value for selected keys dimensions along the
    /// samples axis.
    ///
    /// The dimensions (names) of `keys_to_move` will be moved from the keys to
    /// the sample labels, and blocks with the same remaining keys dimensions
    /// will be merged together along the sample axis.
    ///
    /// `keys_to_move` must be empty (`keys_to_move.count() == 0`), and the new
    /// sample labels will contain entries corresponding to the merged blocks'
    /// keys.
    ///
    /// The new sample labels will contains all of the merged blocks sample
    /// labels. The order of the samples is controlled by `sort_samples`. If
    /// `sort_samples` is true, samples are re-ordered to keep them
    /// lexicographically sorted. Otherwise they are kept in the order in which
    /// they appear in the blocks.
    ///
    /// This function is only implemented if all merged block have the same
    /// property labels.
    #[inline]
    pub fn keys_to_samples(&self, keys_to_move: &Labels, sort_samples: bool) -> Result<TensorMap, Error> {
        let ptr = unsafe {
            crate::c_api::mts_tensormap_keys_to_samples(
                self.ptr,
                keys_to_move.as_mts_labels_t(),
                sort_samples,
            )
        };

        check_ptr(ptr)?;
        return Ok(unsafe { TensorMap::from_raw(ptr) });
    }

    /// Merge blocks with the same value for selected keys dimensions along the
    /// property axis.
    ///
    /// The dimensions (names) of `keys_to_move` will be moved from the keys to
    /// the property labels, and blocks with the same remaining keys dimensions
    /// will be merged together along the property axis.
    ///
    /// If `keys_to_move` does not contains any entries (`keys_to_move.count()
    /// == 0`), then the new property labels will contain entries corresponding
    /// to the merged blocks only. For example, merging a block with key `a=0`
    /// and properties `p=1, 2` with a block with key `a=2` and properties `p=1,
    /// 3` will produce a block with properties `a, p = (0, 1), (0, 2), (2, 1),
    /// (2, 3)`.
    ///
    /// If `keys_to_move` contains entries, then the property labels must be the
    /// same for all the merged blocks. In that case, the merged property labels
    /// will contains each of the entries of `keys_to_move` and then the current
    /// property labels. For example, using `a=2, 3` in `keys_to_move`, and
    /// blocks with properties `p=1, 2` will result in `a, p = (2, 1), (2, 2),
    /// (3, 1), (3, 2)`.
    ///
    /// The new sample labels will contains all of the merged blocks sample
    /// labels. The order of the samples is controlled by `sort_samples`. If
    /// `sort_samples` is true, samples are re-ordered to keep them
    /// lexicographically sorted. Otherwise they are kept in the order in which
    /// they appear in the blocks.
    #[inline]
    pub fn keys_to_properties(&self, keys_to_move: &Labels, sort_samples: bool) -> Result<TensorMap, Error> {
        let ptr = unsafe {
            crate::c_api::mts_tensormap_keys_to_properties(
                self.ptr,
                keys_to_move.as_mts_labels_t(),
                sort_samples,
            )
        };

        check_ptr(ptr)?;
        return Ok(unsafe { TensorMap::from_raw(ptr) });
    }

    /// Move the given dimensions from the component labels to the property
    /// labels for each block in this `TensorMap`.
    #[inline]
    pub fn components_to_properties(&self, dimensions: &[&str]) -> Result<TensorMap, Error> {
        let dimensions_c = dimensions.iter()
            .map(|&v| CString::new(v).expect("unexpected NULL byte"))
            .collect::<Vec<_>>();

        let dimensions_ptr = dimensions_c.iter()
            .map(|v| v.as_ptr())
            .collect::<Vec<_>>();


        let ptr = unsafe {
            crate::c_api::mts_tensormap_components_to_properties(
                self.ptr,
                dimensions_ptr.as_ptr(),
                dimensions.len(),
            )
        };

        check_ptr(ptr)?;
        return Ok(unsafe { TensorMap::from_raw(ptr) });
    }

    /// Get an iterator over the keys and associated blocks
    #[inline]
    pub fn iter(&self) -> TensorMapIter<'_> {
        return TensorMapIter {
            inner: self.keys().iter().zip(self.blocks())
        };
    }

    /// Get an iterator over the keys and associated blocks, with read-write
    /// access to the blocks
    #[inline]
    pub fn iter_mut(&mut self) -> TensorMapIterMut<'_> {
        // we can not use `self.blocks_mut()` here, since it would
        // double-borrow self
        let mut blocks = Vec::new();
        for i in 0..self.keys().count() {
            blocks.push(unsafe { TensorMap::raw_block_mut_by_id(self.ptr, i) });
        }

        return TensorMapIterMut {
            inner: self.keys().into_iter().zip(blocks)
        };
    }

    /// Get a parallel iterator over the keys and associated blocks
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_iter(&self) -> TensorMapParIter {
        use rayon::prelude::*;
        TensorMapParIter {
            inner: self.keys().par_iter().zip_eq(self.blocks().into_par_iter())
        }
    }

    /// Get a parallel iterator over the keys and associated blocks, with
    /// read-write access to the blocks
    #[cfg(feature = "rayon")]
    #[inline]
    pub fn par_iter_mut(&mut self) -> TensorMapParIterMut {
        use rayon::prelude::*;

        // we can not use `self.blocks_mut()` here, since it would
        // double-borrow self
        let mut blocks = Vec::new();
        for i in 0..self.keys().count() {
            blocks.push(unsafe { TensorMap::raw_block_mut_by_id(self.ptr, i) });
        }

        TensorMapParIterMut {
            inner: self.keys().par_iter().zip_eq(blocks)
        }
    }
}

/******************************************************************************/

/// Iterator over key/block pairs in a [`TensorMap`]
pub struct TensorMapIter<'a> {
    inner: std::iter::Zip<crate::labels::LabelsIter<'a>, std::vec::IntoIter<TensorBlockRef<'a>>>
}

impl<'a> Iterator for TensorMapIter<'a> {
    type Item = (&'a [LabelValue], TensorBlockRef<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> ExactSizeIterator for TensorMapIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a> FusedIterator for TensorMapIter<'a> {}

impl<'a> IntoIterator for &'a TensorMap {
    type Item = (&'a [LabelValue], TensorBlockRef<'a>);

    type IntoIter = TensorMapIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/******************************************************************************/

/// Iterator over key/block pairs in a [`TensorMap`], with mutable access to the
/// blocks
pub struct TensorMapIterMut<'a> {
    inner: std::iter::Zip<crate::labels::LabelsIter<'a>, std::vec::IntoIter<TensorBlockRefMut<'a>>>
}

impl<'a> Iterator for TensorMapIterMut<'a> {
    type Item = (&'a [LabelValue], TensorBlockRefMut<'a>);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a> ExactSizeIterator for TensorMapIterMut<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<'a> FusedIterator for TensorMapIterMut<'a> {}

impl<'a> IntoIterator for &'a mut TensorMap {
    type Item = (&'a [LabelValue], TensorBlockRefMut<'a>);

    type IntoIter = TensorMapIterMut<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}


/******************************************************************************/

/// Parallel iterator over key/block pairs in a [`TensorMap`]
#[cfg(feature = "rayon")]
pub struct TensorMapParIter<'a> {
    inner: rayon::iter::ZipEq<crate::labels::LabelsParIter<'a>, rayon::vec::IntoIter<TensorBlockRef<'a>>>,
}

#[cfg(feature = "rayon")]
impl<'a> rayon::iter::ParallelIterator for TensorMapParIter<'a> {
    type Item = (&'a [LabelValue], TensorBlockRef<'a>);

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item> {
        self.inner.drive_unindexed(consumer)
    }
}

#[cfg(feature = "rayon")]
impl<'a> rayon::iter::IndexedParallelIterator for TensorMapParIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        self.inner.drive(consumer)
    }

    #[inline]
    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        self.inner.with_producer(callback)
    }
}

/******************************************************************************/

/// Parallel iterator over key/block pairs in a [`TensorMap`], with mutable
/// access to the blocks
#[cfg(feature = "rayon")]
pub struct TensorMapParIterMut<'a> {
    inner: rayon::iter::ZipEq<crate::labels::LabelsParIter<'a>, rayon::vec::IntoIter<TensorBlockRefMut<'a>>>,
}

#[cfg(feature = "rayon")]
impl<'a> rayon::iter::ParallelIterator for TensorMapParIterMut<'a> {
    type Item = (&'a [LabelValue], TensorBlockRefMut<'a>);

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item> {
        self.inner.drive_unindexed(consumer)
    }
}

#[cfg(feature = "rayon")]
impl<'a> rayon::iter::IndexedParallelIterator for TensorMapParIterMut<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }

    #[inline]
    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        self.inner.drive(consumer)
    }

    #[inline]
    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        self.inner.with_producer(callback)
    }
}

/******************************************************************************/

#[cfg(test)]
mod tests {
    use crate::{Labels, TensorBlock, TensorMap};

    #[test]
    #[allow(clippy::cast_lossless, clippy::float_cmp)]
    fn iter() {
        let block_1 = TensorBlock::new(
            ndarray::ArrayD::from_elem(vec![2, 3], 1.0),
            &Labels::new(["samples"], &[[0], [1]]),
            &[],
            &Labels::new(["properties"], &[[-2], [0], [1]]),
        ).unwrap();

        let block_2 = TensorBlock::new(
            ndarray::ArrayD::from_elem(vec![1, 1], 3.0),
            &Labels::new(["samples"], &[[1]]),
            &[],
            &Labels::new(["properties"], &[[1]]),
        ).unwrap();

        let block_3 = TensorBlock::new(
            ndarray::ArrayD::from_elem(vec![3, 2], -4.0),
            &Labels::new(["samples"], &[[0], [1], [3]]),
            &[],
            &Labels::new(["properties"], &[[-2], [1]]),
        ).unwrap();

        let mut tensor = TensorMap::new(
            Labels::new(["key"], &[[1], [3], [-4]]),
            vec![block_1, block_2, block_3],
        ).unwrap();

        // iterate over keys & blocks
        for (key, block) in &tensor {
            assert_eq!(block.values().to_array()[[0, 0]], key[0].i32() as f64);
        }

        // iterate over keys & blocks mutably
        for (key, mut block) in &mut tensor {
            let array = block.values_mut().to_array_mut();
            *array *= 2.0;
            assert_eq!(array[[0, 0]], 2.0 * (key[0].i32() as f64));
        }
    }
}

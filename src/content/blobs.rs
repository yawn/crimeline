use std::io::{BufWriter, Cursor};
use std::sync::{Arc, LazyLock};

use anyhow::{Context, Result};
use arrow::{
    array::{BinaryArray, RecordBatch, UInt64Array},
    datatypes::{DataType, Field, Schema},
    ipc::{reader::FileReader, writer::FileWriter},
};
use arrow_buffer::Buffer;
use bytes::Bytes;
use memmap2::Mmap;
use tempfile::NamedTempFile;

use tracing::trace;

use crate::usage::{ReportUsage, Usage};

use super::Cid;

static ARROW_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("cid", DataType::UInt64, false),
        Field::new("blob", DataType::Binary, false),
    ]))
});

fn schema() -> Arc<Schema> {
    ARROW_SCHEMA.clone()
}

pub struct BlobStoreBuilder {
    writer: FileWriter<BufWriter<NamedTempFile>>,
}

pub struct BlobStore {
    _buffer: Buffer,
    _file: NamedTempFile,
    blobs: BinaryArray,
    cids: UInt64Array,
}

impl BlobStoreBuilder {
    pub fn new() -> Result<Self> {
        let file = NamedTempFile::with_prefix("crimeline-hot-arena")
            .context("create tempfile for blob arena")?;

        let file = BufWriter::new(file);

        let writer = FileWriter::try_new(file, &schema()).context("create arrow ipc writer")?;

        Ok(Self { writer })
    }

    pub fn append<T: AsRef<[u8]>>(&mut self, cids: &[Cid], blobs: &[T]) -> Result<()> {
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(UInt64Array::from_iter_values(cids.iter().copied())),
                Arc::new(BinaryArray::from_iter_values(blobs)),
            ],
        )
        .context("create blob batch")?;

        self.writer.write(&batch).context("write blob batch")?;

        trace!(len = cids.len(), "appended to blob store");

        Ok(())
    }

    pub fn build_and_sort(self, perm: &[usize]) -> Result<BlobStore> {
        let buffered = self.writer.into_inner().context("finish arrow writer")?;

        let old_file = buffered.into_inner().context("flush buffered writer")?;

        if perm.is_empty() {
            let empty: Vec<&[u8]> = vec![];
            return BlobStore::new(&[], &empty);
        }

        let mmap = unsafe { Mmap::map(old_file.as_file()).context("mmap blob file")? };

        let buffer = Buffer::from(Bytes::from_owner(mmap));

        let reader = FileReader::try_new(Cursor::new(buffer.as_slice()), None)
            .context("open arrow reader")?;

        let mut batches = Vec::new();
        let mut offsets = vec![0usize];

        for batch_result in reader {
            let batch = batch_result.context("read blob batch")?;

            offsets.push(
                offsets
                    .last()
                    .copied()
                    .context("offset vec unexpectedly empty")?
                    + batch.num_rows(),
            );

            batches.push(batch);
        }

        let typed: Vec<(&UInt64Array, &BinaryArray)> = batches
            .iter()
            .map(|b| -> Result<_> {
                Ok((
                    b.column(0)
                        .as_any()
                        .downcast_ref::<UInt64Array>()
                        .context("downcast cid column")?,
                    b.column(1)
                        .as_any()
                        .downcast_ref::<BinaryArray>()
                        .context("downcast blob column")?,
                ))
            })
            .collect::<Result<_>>()?;

        // collect sorted rows via perm — blob refs point into the mmap
        let n = perm.len();
        let mut blobs: Vec<&[u8]> = Vec::with_capacity(n);
        let mut cids: Vec<Cid> = Vec::with_capacity(n);

        for &physical in perm {
            let batch_idx = offsets.partition_point(|&o| o <= physical) - 1;
            let row = physical - offsets[batch_idx];
            let (cid_col, blob_col) = typed[batch_idx];

            blobs.push(blob_col.value(row));
            cids.push(cid_col.value(row));
        }

        trace!(len = cids.len(), "built and sorted blob store");

        BlobStore::new(&cids, &blobs)
    }

    pub fn build_presorted(self) -> Result<BlobStore> {
        let buffered = self.writer.into_inner().context("finish arrow writer")?;

        let old_file = buffered.into_inner().context("flush buffered writer")?;

        let mmap = unsafe { Mmap::map(old_file.as_file()).context("mmap blob file")? };

        let buffer = Buffer::from(Bytes::from_owner(mmap));

        let reader = FileReader::try_new(Cursor::new(buffer.as_slice()), None)
            .context("open arrow reader")?;

        let mut batches = Vec::new();

        for batch_result in reader {
            batches.push(batch_result.context("read blob batch")?);
        }

        // break if already single batch
        if batches.len() == 1 {
            drop(batches);
            drop(buffer);
            return BlobStore::from_file(old_file);
        }

        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

        let mut blobs: Vec<&[u8]> = Vec::with_capacity(total_rows);
        let mut cids: Vec<Cid> = Vec::with_capacity(total_rows);

        for batch in &batches {
            let cid_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .context("downcast cid column")?;

            let blob_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .context("downcast blob column")?;

            for i in 0..batch.num_rows() {
                blobs.push(blob_col.value(i));
                cids.push(cid_col.value(i));
            }
        }

        trace!(len = total_rows, "built and presorted blob store");

        BlobStore::new(&cids, &blobs)
    }
}

impl BlobStore {
    /// Create a blob store from parallel cid/blob slices.
    pub fn new<T: AsRef<[u8]>>(cids: &[Cid], blobs: &[T]) -> Result<Self> {
        let file =
            NamedTempFile::with_prefix("crimeline-cold-arena").context("create blob tempfile")?;

        {
            let batch = RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(UInt64Array::from_iter_values(cids.iter().copied())),
                    Arc::new(BinaryArray::from_iter_values(blobs)),
                ],
            )
            .context("create blob batch")?;

            let writer = BufWriter::new(file.as_file());

            let mut writer =
                FileWriter::try_new(writer, &schema()).context("create arrow writer")?;

            writer.write(&batch).context("write blob batch")?;

            writer.finish().context("finish arrow writer")?;
        }

        Self::from_file(file)
    }

    fn from_file(file: NamedTempFile) -> Result<Self> {
        let mmap = unsafe { Mmap::map(file.as_file()).context("mmap blob file")? };

        let buffer = Buffer::from(Bytes::from_owner(mmap));

        let mut reader = FileReader::try_new(Cursor::new(buffer.as_slice()), None)
            .context("open arrow reader")?;

        let batch = reader
            .next()
            .context("empty blob file")?
            .context("read blob batch")?;

        let cids = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .context("downcast cid column")?
            .clone();

        let blobs = batch
            .column(1)
            .as_any()
            .downcast_ref::<BinaryArray>()
            .context("downcast blob column")?
            .clone();

        Ok(Self {
            _file: file,
            _buffer: buffer,
            cids,
            blobs,
        })
    }

    pub fn resolve(&self, idx: usize) -> (Cid, &[u8]) {
        (self.cids.value(idx), self.blobs.value(idx))
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn is_empty(&self) -> bool {
        self.cids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.cids.len()
    }
}

impl ReportUsage for BlobStoreBuilder {
    fn usage(&self) -> Usage {
        let mut u = Usage::default();
        let disk = self
            .writer
            .get_ref()
            .get_ref()
            .as_file()
            .metadata()
            .map(|m| m.len())
            .unwrap_or(0);
        u.add_disk_usage(disk);
        u
    }
}

impl ReportUsage for BlobStore {
    fn usage(&self) -> Usage {
        let mut u = Usage::default();
        let disk = self
            ._file
            .as_file()
            .metadata()
            .map(|m| m.len())
            .unwrap_or(0);
        u.add_disk_usage(disk);
        u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_store() -> Result<()> {
        let empty: Vec<&[u8]> = vec![];
        let store = BlobStore::new(&[], &empty)?;
        assert_eq!(store.len(), 0);
        assert!(store.usage().disk > 0, "empty IPC file has header");
        Ok(())
    }

    #[test]
    fn roundtrip_resolve() -> Result<()> {
        let cids = [100u64, 200, 300];
        let blobs: Vec<&[u8]> = vec![b"aaa", b"bbb", b"ccc"];

        let store = BlobStore::new(&cids, &blobs)?;
        assert_eq!(store.len(), 3);

        let (cid, blob) = store.resolve(0);
        assert_eq!(cid, 100);
        assert_eq!(blob, b"aaa");

        let (cid, blob) = store.resolve(2);
        assert_eq!(cid, 300);
        assert_eq!(blob, b"ccc");
        Ok(())
    }

    #[test]
    fn builder_empty() -> Result<()> {
        let builder = BlobStoreBuilder::new()?;
        let store = builder.build_and_sort(&[])?;
        assert_eq!(store.len(), 0);
        Ok(())
    }

    #[test]
    fn builder_sorted_build() -> Result<()> {
        let mut builder = BlobStoreBuilder::new()?;
        builder.append(&[100, 200], &[b"aaa".as_slice(), b"bbb"])?;
        builder.append(&[300], &[b"ccc".as_slice()])?;

        // Reverse order: sorted[0]=physical[2], sorted[1]=physical[1], sorted[2]=physical[0]
        let store = builder.build_and_sort(&[2, 1, 0])?;
        assert_eq!(store.len(), 3);

        let (cid, blob) = store.resolve(0);
        assert_eq!(cid, 300);
        assert_eq!(blob, b"ccc");

        let (cid, blob) = store.resolve(2);
        assert_eq!(cid, 100);
        assert_eq!(blob, b"aaa");
        Ok(())
    }

    #[test]
    fn builder_disk_usage() -> Result<()> {
        let mut builder = BlobStoreBuilder::new()?;
        builder.append(&[1], &[b"x".as_slice()])?;
        assert!(builder.usage().disk > 0);
        Ok(())
    }

    #[test]
    fn builder_identity() -> Result<()> {
        let mut builder = BlobStoreBuilder::new()?;
        builder.append(&[100, 200], &[b"aaa".as_slice(), b"bbb"])?;
        builder.append(&[300], &[b"ccc".as_slice()])?;

        let store = builder.build_presorted()?;
        assert_eq!(store.len(), 3);

        let (cid, blob) = store.resolve(0);
        assert_eq!(cid, 100);
        assert_eq!(blob, b"aaa");

        let (cid, blob) = store.resolve(2);
        assert_eq!(cid, 300);
        assert_eq!(blob, b"ccc");
        Ok(())
    }

    #[test]
    fn builder_identity_empty() -> Result<()> {
        let builder = BlobStoreBuilder::new()?;
        let store = builder.build_presorted()?;
        assert_eq!(store.len(), 0);
        Ok(())
    }
}

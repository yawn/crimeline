use std::collections::HashSet;
use std::io::Write;
use std::mem::size_of;
use std::sync::{Arc, LazyLock};

use anyhow::{Context, Result};
use arrow::array::{BinaryArray, RecordBatch, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema};
use bytes::Bytes;
use itertools::Itertools;
use parquet::{
    arrow::{ArrowWriter, arrow_reader::ParquetRecordBatchReaderBuilder},
    basic::{Compression, ZstdLevel},
    file::{metadata::KeyValue, properties::WriterProperties},
};
use tracing::trace;

use crate::usage::{ReportUsage, Usage};
use crate::users::Uid;

use super::blobs::{BlobStore, BlobStoreBuilder};
use super::{Cid, Order, Timestamp, Window};

/// Max blobs held in memory before flushing to the blob store.
const BLOB_BATCH: usize = 256;

static PARQUET_SCHEMA: LazyLock<Arc<Schema>> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("uid", DataType::UInt32, false),
        Field::new("cid", DataType::UInt64, false),
        Field::new("timestamp", DataType::UInt64, false),
        Field::new("blob", DataType::Binary, false),
    ]))
});

fn schema() -> Arc<Schema> {
    PARQUET_SCHEMA.clone()
}

pub struct Cold {
    blobs: BlobStore,
    pub span: Window,
    pub(super) timestamps: Box<[u32]>,
    uids: Box<[Uid]>,
}

pub struct Entry<'a> {
    cold: &'a Cold,
    idx: usize,
    pub uid: Uid,
}

pub struct Hot {
    blobs: BlobStoreBuilder,
    cid_set: HashSet<Cid>,
    cids: Vec<Cid>,
    span: Window,
    pub timestamps: Vec<u32>,
    uids: Vec<Uid>,
}

impl Hot {
    pub fn new(span: Window) -> Result<Self> {
        Ok(Self {
            blobs: BlobStoreBuilder::new()?,
            cid_set: HashSet::new(),
            cids: Vec::new(),
            span,
            timestamps: Vec::new(),
            uids: Vec::new(),
        })
    }

    pub fn add(&mut self, uid: Uid, cid: Cid, ts: Timestamp, blob: &[u8]) -> Result<()> {
        if !self.cid_set.insert(cid) {
            return Ok(());
        }

        self.blobs.append(&[cid], &[blob])?;
        self.cids.push(cid);
        self.timestamps.push(self.span.convert_to_relative(ts));
        self.uids.push(uid);

        trace!(cid, "added to hot arena");

        Ok(())
    }

    pub fn add_bulk<T, B>(&mut self, entries: T) -> Result<()>
    where
        B: AsRef<[u8]>,
        T: IntoIterator<Item = (Uid, Cid, Timestamp, B)>,
    {
        let mut cids: Vec<Cid> = Vec::with_capacity(BLOB_BATCH);
        let mut blobs: Vec<B> = Vec::with_capacity(BLOB_BATCH);

        for chunk in &entries.into_iter().chunks(BLOB_BATCH) {
            cids.clear();
            blobs.clear();

            for (uid, cid, ts, blob) in chunk {
                if !self.cid_set.insert(cid) {
                    continue;
                }

                self.cids.push(cid);
                self.timestamps.push(self.span.convert_to_relative(ts));
                self.uids.push(uid);

                cids.push(cid);
                blobs.push(blob);
            }

            if !cids.is_empty() {
                self.blobs.append(&cids, &blobs)?;
                trace!(len = cids.len(), "added chunk to arena");
            }
        }

        trace!(len = self.cids.len(), "finished bulk add");

        Ok(())
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn is_empty(&self) -> bool {
        self.cids.is_empty()
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn len(&self) -> usize {
        self.cids.len()
    }
}

impl Entry<'_> {
    pub(super) fn new(cold: &Cold, idx: usize) -> Entry<'_> {
        Entry {
            uid: cold.uids[idx],
            cold,
            idx,
        }
    }

    pub fn resolve(&self) -> (Cid, &[u8]) {
        self.cold.blobs.resolve(self.idx)
    }

    pub fn timestamp(&self) -> Timestamp {
        self.cold
            .span
            .convert_to_absolute(self.cold.timestamps[self.idx])
    }
}

impl ReportUsage for Hot {
    fn usage(&self) -> Usage {
        let mut u = Usage::default();

        // HashSet stores (hash, value) per bucket.
        let bucket_bytes = size_of::<Cid>() + size_of::<u64>();
        u.add_heap_usage(self.cid_set.capacity() * bucket_bytes);
        u.add_heap_waste((self.cid_set.capacity() - self.cid_set.len()) * bucket_bytes);
        u.add_vec(&self.cids);
        u.add_vec(&self.timestamps);
        u.add_vec(&self.uids);
        u += self.blobs.usage();

        u
    }
}

impl Cold {
    pub fn export<T: Write + Send>(&self, writer: T) -> Result<()> {
        let compression = Compression::ZSTD(ZstdLevel::try_new(3)?);

        let metadata = vec![
            KeyValue::new("crimeline.epoch".into(), Some(self.span.epoch.to_string())),
            KeyValue::new(
                "crimeline.duration".into(),
                Some(self.span.duration.to_string()),
            ),
        ];

        let props = WriterProperties::builder()
            .set_compression(compression)
            .set_key_value_metadata(Some(metadata))
            .build();

        let mut pq =
            ArrowWriter::try_new(writer, schema(), Some(props)).context("create parquet writer")?;

        let n = self.uids.len();

        let mut blobs: Vec<&[u8]> = Vec::with_capacity(BLOB_BATCH);
        let mut cids: Vec<Cid> = Vec::with_capacity(BLOB_BATCH);

        for start in (0..n).step_by(BLOB_BATCH) {
            let end = (start + BLOB_BATCH).min(n);

            blobs.clear();
            cids.clear();

            for i in start..end {
                let (cid, blob) = self.blobs.resolve(i);

                blobs.push(blob);
                cids.push(cid);
            }

            let batch = RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(UInt32Array::from_iter_values(
                        self.uids[start..end].iter().copied(),
                    )),
                    Arc::new(UInt64Array::from_iter_values(cids.iter().copied())),
                    Arc::new(UInt64Array::from_iter_values(
                        self.timestamps[start..end]
                            .iter()
                            .map(|&t| self.span.convert_to_absolute(t)),
                    )),
                    Arc::new(BinaryArray::from_iter_values(&blobs)),
                ],
            )
            .context("create export batch")?;

            pq.write(&batch).context("write parquet batch")?;

            trace!(len = end - start, "exported chunk");
        }

        pq.close().context("close parquet writer")?;

        trace!(len = n, "exported arena");

        Ok(())
    }

    pub fn import(data: Bytes) -> Result<Arc<Self>> {
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(data).context("open parquet reader")?;

        let metadata = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .context("missing parquet metadata")?;

        let span: Window;

        {
            let epoch: u64 = metadata
                .iter()
                .find(|e| e.key == "crimeline.epoch")
                .and_then(|e| e.value.as_ref())
                .context("missing crimeline.epoch")?
                .parse()
                .context("parse epoch")?;

            let duration: u32 = metadata
                .iter()
                .find(|e| e.key == "crimeline.duration")
                .and_then(|e| e.value.as_ref())
                .context("missing crimeline.duration")?
                .parse()
                .context("parse duration")?;

            span = Window::new(epoch, duration);
        }

        let num_rows = builder.metadata().file_metadata().num_rows() as usize;

        let reader = builder.build().context("build parquet reader")?;

        let mut blob_builder = BlobStoreBuilder::new()?;
        let mut timestamps: Vec<u32> = Vec::with_capacity(num_rows);
        let mut uids: Vec<Uid> = Vec::with_capacity(num_rows);

        for batch_result in reader {
            let batch = batch_result.context("read parquet batch")?;

            let uid_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("downcast uid column")?;

            let cid_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .context("downcast cid column")?;

            let ts_col = batch
                .column(2)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .context("downcast timestamp column")?;

            let blob_col = batch
                .column(3)
                .as_any()
                .downcast_ref::<BinaryArray>()
                .context("downcast blob column")?;

            let n = batch.num_rows();

            let mut cids: Vec<Cid> = Vec::with_capacity(n);
            let mut blobs: Vec<&[u8]> = Vec::with_capacity(n);

            for i in 0..n {
                blobs.push(blob_col.value(i));
                cids.push(cid_col.value(i));
                timestamps.push(span.convert_to_relative(ts_col.value(i)));
                uids.push(uid_col.value(i));
            }

            blob_builder.append(&cids, &blobs)?;

            trace!(len = n, "imported chunk");
        }

        let blobs = blob_builder.build_presorted()?;

        trace!(len = uids.len(), "imported arena");

        Ok(Arc::new(Cold {
            blobs,
            span,
            timestamps: timestamps.into_boxed_slice(),
            uids: uids.into_boxed_slice(),
        }))
    }

    pub fn iter(&self, order: Order, start: Timestamp) -> impl Iterator<Item = Entry<'_>> {
        let skip_to = if start <= self.span.epoch {
            0
        } else {
            let rel_start = (start - self.span.epoch) as u32;
            self.timestamps.partition_point(|&ts| ts < rel_start)
        };

        let len = self.len();

        order
            .range(skip_to..len)
            .map(move |idx| Entry::new(self, idx))
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn is_empty(&self) -> bool {
        self.uids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.uids.len()
    }
}

impl ReportUsage for Cold {
    fn usage(&self) -> Usage {
        let mut u = Usage::default();
        u.add_boxed_slice(&self.timestamps);
        u.add_boxed_slice(&self.uids);
        u += self.blobs.usage();
        u
    }
}

impl TryInto<Arc<Cold>> for Hot {
    type Error = anyhow::Error;

    fn try_into(self) -> Result<Arc<Cold>> {
        let Self {
            cid_set: _,
            cids,
            timestamps,
            uids,
            span,
            blobs,
        } = self;

        let n = cids.len();

        let mut perm: Vec<usize> = (0..n).collect();
        perm.sort_unstable_by_key(|&i| (timestamps[i], cids[i]));

        let sorted_timestamps: Vec<u32> = perm.iter().map(|&i| timestamps[i]).collect();

        let sorted_uids: Vec<Uid> = perm.iter().map(|&i| uids[i]).collect();

        trace!(len = n, "froze arena");

        Ok(Arc::new(Cold {
            blobs: blobs.build_and_sort(&perm)?,
            span,
            timestamps: sorted_timestamps.into_boxed_slice(),
            uids: sorted_uids.into_boxed_slice(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_bulk_dedup_across_individual_and_bulk() -> Result<()> {
        let mut hot = Hot::new(Window::new(1000, 100))?;
        hot.add(1, 100, 1010, b"individual")?;

        let entries: Vec<(Uid, Cid, Timestamp, &[u8])> = vec![
            (2, 100, 1020, b"bulk_dup"),
            (3, 200, 1030, b"bulk_new"),
        ];
        hot.add_bulk(entries)?;

        let cold: Arc<Cold> = hot.try_into()?;
        assert_eq!(cold.len(), 2, "CID 100 from bulk should be deduped");
        Ok(())
    }

    #[test]
    fn add_bulk_deduplicates() -> Result<()> {
        let entries: Vec<(Uid, Cid, Timestamp, &[u8])> = vec![
            (1, 100, 1010, b"first"),
            (2, 100, 1020, b"second"),
            (3, 200, 1030, b"third"),
        ];

        let mut hot = Hot::new(Window::new(1000, 100))?;
        hot.add_bulk(entries)?;

        let cold: Arc<Cold> = hot.try_into()?;
        assert_eq!(cold.len(), 2, "duplicate CID should be skipped");

        let entries: Vec<_> = cold.iter(Order::Asc, 0).collect();
        let (cid0, blob0) = entries[0].resolve();
        assert_eq!(cid0, 100);
        assert_eq!(blob0, b"first", "first occurrence wins");
        Ok(())
    }

    #[test]
    fn add_bulk_empty() -> Result<()> {
        let mut hot = Hot::new(Window::new(1000, 100))?;
        let entries: Vec<(Uid, Cid, Timestamp, &[u8])> = vec![];
        hot.add_bulk(entries)?;

        let cold: Arc<Cold> = hot.try_into()?;
        assert_eq!(cold.len(), 0);
        Ok(())
    }

    #[test]
    fn add_bulk_equivalent_to_individual() -> Result<()> {
        let entries: Vec<(Uid, Cid, Timestamp, &[u8])> = vec![
            (1, 100, 1010, b"a"),
            (2, 200, 1020, b"b"),
            (3, 300, 1005, b"c"),
        ];

        let mut bulk = Hot::new(Window::new(1000, 100))?;
        bulk.add_bulk(entries.clone())?;

        let mut individual = Hot::new(Window::new(1000, 100))?;
        for (uid, cid, ts, blob) in &entries {
            individual.add(*uid, *cid, *ts, *blob)?;
        }

        let cold_bulk: Arc<Cold> = bulk.try_into()?;
        let cold_ind: Arc<Cold> = individual.try_into()?;

        assert_eq!(cold_bulk.len(), cold_ind.len());

        let b: Vec<_> = cold_bulk.iter(Order::Asc, 0).collect();
        let i: Vec<_> = cold_ind.iter(Order::Asc, 0).collect();

        for (be, ie) in b.iter().zip(i.iter()) {
            assert_eq!(be.uid, ie.uid, "uid mismatch");
            assert_eq!(be.timestamp(), ie.timestamp(), "timestamp mismatch");
            assert_eq!(be.resolve(), ie.resolve(), "resolve mismatch");
        }
        Ok(())
    }

    #[test]
    fn add_bulk_multi_chunk() -> Result<()> {
        let n = BLOB_BATCH + 50; // forces at least two chunks
        let entries: Vec<(Uid, Cid, Timestamp, Vec<u8>)> = (0..n)
            .map(|i| {
                (
                    i as Uid,
                    (1000 + i) as Cid,
                    1000 + i as Timestamp,
                    format!("blob_{i}").into_bytes(),
                )
            })
            .collect();

        let mut hot = Hot::new(Window::new(0, 10000))?;
        hot.add_bulk(entries)?;

        let cold: Arc<Cold> = hot.try_into()?;
        assert_eq!(cold.len(), n);

        // spot-check first and last
        let all: Vec<_> = cold.iter(Order::Asc, 0).collect();
        let (cid_first, _) = all[0].resolve();
        let (cid_last, _) = all[n - 1].resolve();
        assert_eq!(cid_first, 1000);
        assert_eq!(cid_last, (1000 + n - 1) as Cid);
        Ok(())
    }

    #[test]
    fn cold_empty() -> Result<()> {
        let hot = Hot::new(Window::new(1000, 100))?;
        let cold: Arc<Cold> = hot.try_into()?;
        assert_eq!(cold.len(), 0);
        assert_eq!(cold.iter(Order::Asc, 0).count(), 0);
        Ok(())
    }

    #[test]
    fn cold_iter_desc() -> Result<()> {
        let mut hot = Hot::new(Window::new(1000, 100))?;

        hot.add(1, 100, 1010, b"a")?;
        hot.add(2, 200, 1020, b"b")?;
        hot.add(3, 300, 1005, b"c")?;

        let cold: Arc<Cold> = hot.try_into()?;

        // Desc: 200@1020, 100@1010, 300@1005
        let entries: Vec<_> = cold.iter(Order::Desc, 0).collect();
        assert_eq!(entries[0].uid, 2);
        assert_eq!(entries[0].timestamp(), 1020);
        assert_eq!(entries[1].uid, 1);
        assert_eq!(entries[1].timestamp(), 1010);
        assert_eq!(entries[2].uid, 3);
        assert_eq!(entries[2].timestamp(), 1005);
        Ok(())
    }

    #[test]
    fn cold_iter_start_skips_entries() -> Result<()> {
        let mut hot = Hot::new(Window::new(1000, 100))?;
        hot.add(1, 100, 1010, b"a")?;
        hot.add(2, 200, 1020, b"b")?;
        hot.add(3, 300, 1050, b"c")?;
        hot.add(4, 400, 1080, b"d")?;

        let cold: Arc<Cold> = hot.try_into()?;

        // start=1020 should skip the entry at 1010
        let entries: Vec<_> = cold.iter(Order::Asc, 1020).collect();
        assert_eq!(entries.len(), 3, "asc: should skip 1 entry before start");
        assert_eq!(entries[0].timestamp(), 1020);
        assert_eq!(entries[1].timestamp(), 1050);
        assert_eq!(entries[2].timestamp(), 1080);

        // Desc with start=1020 should yield 1080, 1050, 1020
        let entries: Vec<_> = cold.iter(Order::Desc, 1020).collect();
        assert_eq!(entries.len(), 3, "desc: should skip 1 entry before start");
        assert_eq!(entries[0].timestamp(), 1080);
        assert_eq!(entries[1].timestamp(), 1050);
        assert_eq!(entries[2].timestamp(), 1020);
        Ok(())
    }

    #[test]
    fn cold_resolve_deduplicated() -> Result<()> {
        let mut hot = Hot::new(Window::new(1000, 100))?;

        // Add duplicate CIDs — only first blob should be kept
        hot.add(1, 100, 1010, b"first")?;
        hot.add(2, 100, 1020, b"second")?;

        let cold: Arc<Cold> = hot.try_into()?;
        assert_eq!(cold.len(), 1);

        let entries: Vec<_> = cold.iter(Order::Asc, 0).collect();
        let (cid, blob) = entries[0].resolve();
        assert_eq!(cid, 100);
        assert_eq!(blob, b"first");
        Ok(())
    }

    #[test]
    fn cold_resolve_retrieves_blobs() -> Result<()> {
        let mut hot = Hot::new(Window::new(1000, 100))?;

        hot.add(1, 100, 1010, b"blob_100")?;
        hot.add(2, 200, 1020, b"blob_200")?;
        hot.add(3, 300, 1005, b"blob_300")?;

        let cold: Arc<Cold> = hot.try_into()?;

        // Sorted by (timestamp, cid): 300@1005, 100@1010, 200@1020
        let entries: Vec<_> = cold.iter(Order::Asc, 0).collect();
        assert_eq!(entries.len(), 3);

        let (cid0, blob0) = entries[0].resolve();
        assert_eq!(cid0, 300);
        assert_eq!(blob0, b"blob_300");

        let (cid1, blob1) = entries[1].resolve();
        assert_eq!(cid1, 100);
        assert_eq!(blob1, b"blob_100");

        let (cid2, blob2) = entries[2].resolve();
        assert_eq!(cid2, 200);
        assert_eq!(blob2, b"blob_200");
        Ok(())
    }

    #[test]
    fn cold_usage_traits() -> Result<()> {
        let mut hot = Hot::new(Window::new(1000, 100))?;
        hot.add(1, 100, 1010, b"test1")?;
        hot.add(2, 200, 1020, b"test2")?;

        let cold: Arc<Cold> = hot.try_into()?;

        let u = cold.usage();
        assert!(u.heap > 0, "cold arena should have heap usage");
        assert!(u.disk > 0, "cold arena should have disk usage");
        Ok(())
    }

    #[test]
    fn export_import_empty() -> Result<()> {
        let hot = Hot::new(Window::new(5000, 200))?;
        let cold: Arc<Cold> = hot.try_into()?;

        let mut buf = Vec::new();
        cold.export(&mut buf)?;

        let imported = Cold::import(Bytes::from(buf))?;

        assert_eq!(imported.len(), 0);
        assert_eq!(imported.span, Window::new(5000, 200));
        Ok(())
    }

    #[test]
    fn export_import_preserves_order() -> Result<()> {
        let mut hot = Hot::new(Window::new(0, 10000))?;
        hot.add(5, 50, 5000, b"e")?;
        hot.add(1, 10, 1000, b"a")?;
        hot.add(3, 30, 3000, b"c")?;
        hot.add(2, 20, 2000, b"b")?;
        hot.add(4, 40, 4000, b"d")?;

        let cold: Arc<Cold> = hot.try_into()?;

        let mut buf = Vec::new();
        cold.export(&mut buf)?;

        let imported = Cold::import(Bytes::from(buf))?;

        let entries: Vec<_> = imported.iter(Order::Asc, 0).collect();
        for w in entries.windows(2) {
            assert!(
                w[0].timestamp() <= w[1].timestamp(),
                "out of order: {} > {}",
                w[0].timestamp(),
                w[1].timestamp(),
            );
        }
        Ok(())
    }

    #[test]
    fn export_import_roundtrip() -> Result<()> {
        let mut hot = Hot::new(Window::new(1000, 100))?;
        hot.add(1, 100, 1010, b"blob_100")?;
        hot.add(2, 200, 1020, b"blob_200")?;
        hot.add(3, 300, 1005, b"blob_300")?;

        let cold: Arc<Cold> = hot.try_into()?;

        let mut buf = Vec::new();
        cold.export(&mut buf)?;

        let imported = Cold::import(Bytes::from(buf))?;

        assert_eq!(imported.span, cold.span);
        assert_eq!(imported.span.end_exclusive(), cold.span.end_exclusive());
        assert_eq!(imported.len(), cold.len());

        let orig: Vec<_> = cold.iter(Order::Asc, 0).collect();
        let imp: Vec<_> = imported.iter(Order::Asc, 0).collect();

        for (o, i) in orig.iter().zip(imp.iter()) {
            assert_eq!(o.timestamp(), i.timestamp(), "timestamp mismatch");
            assert_eq!(o.uid, i.uid, "uid mismatch");
            let (o_cid, o_blob) = o.resolve();
            let (i_cid, i_blob) = i.resolve();
            assert_eq!(o_cid, i_cid, "cid mismatch");
            assert_eq!(o_blob, i_blob, "blob mismatch");
        }
        Ok(())
    }

    #[test]
    fn hot_usage_traits() -> Result<()> {
        let mut hot = Hot::new(Window::new(1000, 100))?;
        hot.add(1, 100, 1010, b"test")?;

        let u = hot.usage();
        assert!(u.heap > 0, "hot arena should have heap usage");
        assert!(u.disk > 0, "hot arena should have disk usage");
        Ok(())
    }
}

# `crimeline`

> "Your design is fan-out-on-read with time-indexed storage, which is essentially the worst combination for a read-heavy social workload." - Claude

Please note: this is just an experiment.

Timeline builder for ATProto that does not rely on precomputation. Instead of maintaining per-user feeds at write time, content is stored in time-windowed arenas and fan-out happens at read time — filtering by user relationships on the fly.

The appview database (Postgres) holds content in time-partitioned tables. Each row gets a monotonic `u64` content id and a `u32` author id. A background process exports time slices as Parquet files. Timeline servers import these into in-memory cold arenas for serving. User relationships (follows, blocks) are loaded into sharded adjacency maps. At read time the server iterates arenas in time order, checking the social graph per entry to decide inclusion — no precomputed feeds, no fan-out-on-write.

Coordination of cold arena export/import is not part of this crate but could be built on top of Postgres job-style tables and notifications.

A future option is including full CIDs in exports so clients can be served with zero database interaction. Moving the apis to async might also be a good idea. Also there is no API (to avoid extra overhead) to random access blobs in the store by cid (which might be needed) - since the heap requirements of content _should_ be small that should likely be added as well.

See `examples/` for runnable demos.

## Content Arenas

### Why Arrow AND Parquet

**Arrow IPC** is the runtime format. A sorted `RecordBatch` is written to a tempfile and mmap-ed back — `resolve(idx)` is a pointer offset into the mapped region, not a deserialization. Blobs never touch the heap. **Parquet** (zstd-3) is the exchange format for durable storage and transfer. On import, Parquet batches are re-materialized into an mmap-backed Arrow IPC file. Parquet for persistence, Arrow IPC + mmap for free random access at runtime.

### Hot Arena (write path)

Accumulates incoming content. Deduplicates on `Cid` via `HashSet`. Blobs stream to disk through an Arrow IPC writer during ingestion. `add`: O(1) amortized. `add_bulk`: O(k).

```
Hot { cid_set: HashSet<Cid>, cids: Vec<Cid>, timestamps: Vec<u32>, uids: Vec<Uid>, span: Window, blobs: BlobStoreBuilder }
```

### Hot → Cold compaction

Sorts by `(timestamp, cid)` via permutation index — O(n log n). Writes a single sorted Arrow IPC batch to a new mmap-backed tempfile. Absolute u64 timestamps compress to u32 relative offsets within the arena's `Window [epoch, epoch+duration)`, saving 4 B/entry.

### Cold Arena (read path)

Read-only. 8 bytes heap per entry (u32 uid + u32 relative timestamp). Content resolved on demand from the mmap — O(1) per entry. Iteration is O(n) sequential scan, asc or desc. Export/import via Parquet is O(n).

```
Cold { uids: Box<[u32]>, timestamps: Box<[u32]>, span: Window, blobs: BlobStore(mmap) }
```

Parquet schema: `{uid: u32, cid: u64, timestamp: u64, blob: Binary}`. Metadata keys: `crimeline.epoch`, `crimeline.duration`. Pre-sorted — import uses identity permutation.

### Timeline

Concurrent collection of cold arenas via `ArcSwap<Vec<Arc<Cold>>>`. Reads are **lock-free**: `iter()` atomically snapshots the arena list via `load_full()`. Writers use RCU (`rcu()`): clone, modify, atomically swap. In-flight iterators hold `Arc` refs — removed arenas stay alive until all readers finish. `add`/`remove`: O(a). `iter(start, order)`: O(a) filter + O(n) scan.

## User Relationships

### UserMap

Sharded adjacency map. Each uid is split via bitmask into shard index (low bits) and backbone index (high bits). Each shard holds a `Vec<Vec<Uid>>` — a dense backbone of sorted adjacency lists.

```
UserMap { shards: Box<[RwLock<Shard>]> }    # 2..4096 shards (Sharding enum)
Shard(Vec<Vec<Uid>>)                        # backbone[idx] → sorted target list
```

| Operation | Complexity |
|-----------|-----------|
| `contains(p, t)` | O(log t) binary search, read lock |
| `add(p, t)` | O(log t) search + O(t) shift, write lock |
| `add_bulk(p, targets)` | O(k log k) sort + O(t+k) merge |
| `remove(p, t)` | O(log t) search + O(t) shift |

Memory per edge: 4 B. Backbone overhead per uid slot: 24 B (Vec header). Empty slots from sparse uid spaces are the main source of waste.

### Concurrency

Each shard is wrapped in `parking_lot::RwLock` — readers never block readers, writers lock only their shard. Chosen over std for no poisoning, smaller lock size, and faster uncontended path. The shard count (2–4096) trades contention against memory overhead. The `Timeline` uses `ArcSwap` instead of locks entirely — reads are wait-free atomic loads.

### Relationships

Two `UserMap` instances (follows, blocks). `is_followed_by(p, t)` = `follows.contains(t, p)`. `is_blocked_by(p, t)` = `blocks.contains(t, p)`. `is_mutual(p, t)` = `blocks.contains(p, t) && follows.contains(t, p)`. All O(log t), read locks only.

## Examples

- **`examples/footprint.rs`** — memory footprint estimates from Bluesky-current to Twitter-scale
- **`examples/timeline.rs`** — concurrent timeline demo with follows/blocks filtering

use std::sync::Arc;

use arc_swap::ArcSwap;
use tracing::trace;

use super::arena::{Cold, Entry};
use super::{Order, Timestamp};

pub struct Slice {
    arena_pos: usize,
    entry_pos: usize,
    first: usize,
    n_relevant: usize,
    order: Order,
    skip: usize,
    snapshot: Arc<Vec<Arc<Cold>>>,
    start: Timestamp,
}

pub struct Timeline {
    arenas: ArcSwap<Vec<Arc<Cold>>>,
}

impl Timeline {
    pub fn new(mut arenas: Vec<Arc<Cold>>) -> Self {
        arenas.sort_unstable_by_key(|a| a.span.epoch);
        Timeline {
            arenas: ArcSwap::new(Arc::new(arenas)),
        }
    }

    pub fn add(&self, arena: Arc<Cold>) {
        self.arenas.rcu(|current| {
            let mut next = (**current).clone();
            let pos = next
                .binary_search_by_key(&arena.span.epoch, |a| a.span.epoch)
                .unwrap_or_else(|i| i);
            next.insert(pos, Arc::clone(&arena));
            next
        });

        trace!(len = self.len(), "added arena to timeline");
    }

    pub fn remove(&self, epoch: Timestamp) {
        self.arenas.rcu(|current| {
            let mut next = (**current).clone();
            next.retain(|a| a.span.epoch != epoch);
            next
        });

        trace!(epoch, len = self.len(), "removed arena from timeline");
    }

    pub fn iter(&self, start: Timestamp, order: Order) -> Slice {
        let snapshot = self.arenas.load_full();

        // Arenas are kept sorted by epoch. Find first with entries past `start`.
        let first = (0..snapshot.len())
            .find(|&i| snapshot[i].span.end_exclusive() > start)
            .unwrap_or(snapshot.len());
        let n_relevant = snapshot.len() - first;

        let skip = if n_relevant > 0 {
            let initial = match order {
                Order::Asc => first,
                Order::Desc => snapshot.len() - 1,
            };
            Self::compute_skip(&snapshot[initial], start)
        } else {
            0
        };

        Slice {
            snapshot,
            first,
            n_relevant,
            arena_pos: 0,
            entry_pos: 0,
            order,
            skip,
            start,
        }
    }

    #[cfg_attr(coverage_nightly, coverage(off))]
    pub fn is_empty(&self) -> bool {
        self.arenas.load().is_empty()
    }

    pub fn len(&self) -> usize {
        self.arenas.load().len()
    }

    fn compute_skip(arena: &Cold, start: Timestamp) -> usize {
        if start <= arena.span.epoch {
            0
        } else {
            let rel = (start - arena.span.epoch) as u32;
            arena.timestamps.partition_point(|&ts| ts < rel)
        }
    }
}

impl Slice {
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<Entry<'_>> {
        loop {
            if self.arena_pos >= self.n_relevant {
                return None;
            }

            let arena_idx = match self.order {
                Order::Asc => self.first + self.arena_pos,
                Order::Desc => self.snapshot.len() - 1 - self.arena_pos,
            };
            let arena = &*self.snapshot[arena_idx];
            let skip = self.skip;
            let len = arena.len();
            let effective = len - skip;

            if self.entry_pos >= effective {
                self.arena_pos += 1;
                self.entry_pos = 0;

                if self.arena_pos < self.n_relevant {
                    let next = match self.order {
                        Order::Asc => self.first + self.arena_pos,
                        Order::Desc => self.snapshot.len() - 1 - self.arena_pos,
                    };
                    self.skip = Timeline::compute_skip(&self.snapshot[next], self.start);
                }

                continue;
            }

            let idx = match self.order {
                Order::Asc => skip + self.entry_pos,
                Order::Desc => len - 1 - self.entry_pos,
            };

            self.entry_pos += 1;

            return Some(Entry::new(arena, idx));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::content::Window;
    use crate::content::arena::Hot;
    use proptest::prelude::*;

    fn make_arena(epoch: u64, duration: u32, n: usize) -> Arc<Cold> {
        let mut hot = Hot::new(Window::new(epoch, duration)).unwrap();
        for i in 0..n {
            let ts = epoch + (i as u64 % duration as u64);
            hot.add(i as u32, epoch * 10000 + i as u64, ts, b"x")
                .unwrap();
        }
        hot.try_into().unwrap()
    }

    fn drain(slice: &mut Slice) -> Vec<Timestamp> {
        let mut out = Vec::new();
        while let Some(e) = slice.next() {
            out.push(e.timestamp());
        }
        out
    }

    proptest! {
        #[test]
        fn fuzz_iter(
            num_arenas in 1..6usize,
            base in 0..10_000u64,
            duration in 10..200u32,
            counts in prop::collection::vec(0..30usize, 1..6),
            start_offset in 0..1200u64,
        ) {
            let n = num_arenas.min(counts.len());
            let arenas: Vec<Arc<Cold>> = (0..n)
                .map(|i| {
                    let epoch = base + (i as u64) * duration as u64;
                    make_arena(epoch, duration, counts[i])
                })
                .collect();

            let timeline = Timeline::new(arenas.clone());
            let start = base + start_offset;

            // Reference: collect all absolute timestamps >= start
            let mut expected: Vec<u64> = Vec::new();
            for arena in &arenas {
                if arena.span.end_exclusive() <= start { continue; }
                for &ts in arena.timestamps.iter() {
                    let abs = arena.span.convert_to_absolute(ts);
                    if abs >= start {
                        expected.push(abs);
                    }
                }
            }
            expected.sort();

            // Asc
            let mut slice = timeline.iter(start, Order::Asc);
            let asc = drain(&mut slice);
            for &ts in &asc {
                prop_assert!(ts >= start, "asc: {ts} < start {start}");
            }

            // Desc
            let mut slice = timeline.iter(start, Order::Desc);
            let desc = drain(&mut slice);
            for &ts in &desc {
                prop_assert!(ts >= start, "desc: {ts} < start {start}");
            }

            // Both orders yield the same multiset
            let mut asc_sorted = asc;
            asc_sorted.sort();
            let mut desc_sorted = desc;
            desc_sorted.sort();

            prop_assert_eq!(&asc_sorted, &expected, "asc mismatch");
            prop_assert_eq!(&desc_sorted, &expected, "desc mismatch");
        }

        #[test]
        fn fuzz_add_remove(
            initial in 1..4usize,
            base in 0..10_000u64,
            duration in 10..200u32,
        ) {
            let arenas: Vec<Arc<Cold>> = (0..initial)
                .map(|i| make_arena(base + (i as u64) * duration as u64, duration, 5))
                .collect();

            let timeline = Timeline::new(arenas);
            prop_assert_eq!(timeline.len(), initial);

            // add
            let new_arena = make_arena(base + (initial as u64) * duration as u64, duration, 3);
            timeline.add(new_arena);
            prop_assert_eq!(timeline.len(), initial + 1);

            // remove first arena by epoch
            timeline.remove(base);
            prop_assert_eq!(timeline.len(), initial);

            // idempotent remove
            timeline.remove(base);
            prop_assert_eq!(timeline.len(), initial);

            // no entries from removed epoch survive
            let min_epoch = base + duration as u64;
            let mut slice = timeline.iter(0, Order::Asc);
            while let Some(e) = slice.next() {
                prop_assert!(
                    e.timestamp() >= min_epoch,
                    "entry at {} from removed arena (epoch={})", e.timestamp(), base,
                );
            }
        }
    }
}

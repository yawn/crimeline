use std::cmp::Ordering;

use crate::stats::{HeapUsage, HeapWaste};
use crate::users::Uid;

/// Inner storage for one shard's adjacency lists.
///
/// Uses `Vec<Uid>` over `Box<[Uid]>` which trades a simpler algorithm,
/// (and associated cycles) for an 8-bytes (capacity word) of memory
/// overhead.
pub(crate) struct Shard(Vec<Vec<Uid>>);

impl Shard {
    pub fn new() -> Self {
        Shard(Vec::new())
    }

    pub fn delete(&mut self, index: usize, target: Uid) -> bool {
        let Some(list) = self.0.get_mut(index) else {
            return false;
        };

        match list.binary_search(&target) {
            Ok(pos) => {
                list.remove(pos);
                true
            }
            Err(_) => false,
        }
    }

    pub fn entry(&mut self, index: usize) -> &mut Vec<Uid> {
        let len = self.0.len();

        if len <= index {
            self.0.reserve_exact(index + 1 - len);

            for _ in len..=index {
                self.0.push(Vec::new());
            }
        }

        &mut self.0[index]
    }

    pub fn get(&self, index: usize) -> Option<&[Uid]> {
        self.0.get(index).map(|v| v.as_slice())
    }

    pub fn insert(&mut self, index: usize, target: Uid) -> bool {
        let list = self.entry(index);

        match list.binary_search(&target) {
            Ok(_) => false,
            Err(pos) => {
                list.reserve_exact(1);
                list.insert(pos, target);
                true
            }
        }
    }

    /// Merges a **sorted, deduplicated** slice of targets into the sorted list
    /// at `index`. Returns the number of newly added targets.
    pub fn merge(&mut self, index: usize, incoming: &[Uid]) -> usize {
        let list = self.entry(index);
        let old = std::mem::take(list);

        let size_before = old.len();

        let mut merged = Vec::with_capacity(size_before + incoming.len());

        {
            let (mut i, mut j) = (0, 0);

            while i < old.len() && j < incoming.len() {
                match old[i].cmp(&incoming[j]) {
                    Ordering::Less => {
                        merged.push(old[i]);
                        i += 1;
                    }
                    Ordering::Equal => {
                        merged.push(old[i]);
                        i += 1;
                        j += 1;
                    }
                    Ordering::Greater => {
                        merged.push(incoming[j]);
                        j += 1;
                    }
                }
            }

            merged.extend_from_slice(&old[i..]);
            merged.extend_from_slice(&incoming[j..]);
        }

        merged.shrink_to_fit();

        let size_after = merged.len() - size_before;

        *list = merged;

        size_after
    }
}

impl HeapUsage for Shard {
    fn heap_usage(&self) -> usize {
        let outer = self.0.capacity() * size_of::<Vec<Uid>>();
        let inner: usize = self.0.iter().map(|v| v.capacity() * size_of::<Uid>()).sum();
        outer + inner
    }
}

impl HeapWaste for Shard {
    fn heap_waste(&self) -> usize {
        let empty = self.0.iter().filter(|v| v.is_empty()).count();
        let excess = self.0.capacity() - self.0.len();
        (empty + excess) * size_of::<Vec<Uid>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UID_SIZE: usize = size_of::<Uid>();
    const VEC_SIZE: usize = size_of::<Vec<Uid>>();

    fn assert_stats(shard: &Shard, usage: usize, waste: usize, reason: &str) {
        assert_eq!(shard.heap_usage(), usage, "heap_usage: {reason}");
        assert_eq!(shard.heap_waste(), waste, "heap_waste: {reason}");
    }

    #[test]
    fn stats_empty() {
        assert_stats(&Shard::new(), 0, 0, "no allocations");
    }

    #[test]
    fn stats_entry_without_targets() {
        let mut s = Shard::new();
        s.entry(0);
        assert_stats(&s, VEC_SIZE, VEC_SIZE, "1 backbone slot, empty inner vec");
    }

    #[test]
    fn stats_single_insert() {
        let mut s = Shard::new();
        s.insert(0, 42);
        assert_stats(&s, VEC_SIZE + UID_SIZE, 0, "1 backbone slot + 1 target");
    }

    #[test]
    fn stats_sparse_index() {
        let mut s = Shard::new();
        s.insert(2, 42);
        assert_stats(
            &s,
            3 * VEC_SIZE + UID_SIZE,
            2 * VEC_SIZE,
            "3 backbone slots, slots 0 and 1 are empty padding",
        );
    }

    #[test]
    fn stats_dense_principals() {
        let mut s = Shard::new();
        s.insert(0, 1);
        s.insert(1, 2);
        assert_stats(
            &s,
            2 * VEC_SIZE + 2 * UID_SIZE,
            0,
            "2 dense backbone slots, no padding",
        );
    }

    #[test]
    fn stats_delete_leaves_empty_slot() {
        let mut s = Shard::new();
        s.insert(0, 1);
        s.delete(0, 1);
        assert_stats(
            &s,
            VEC_SIZE + UID_SIZE,
            VEC_SIZE,
            "inner vec retains cap=1 but is empty, counts as backbone waste",
        );
    }

    #[test]
    fn stats_merge_exact_capacity() {
        let mut s = Shard::new();
        s.merge(0, &[1, 2, 3]);
        assert_stats(
            &s,
            VEC_SIZE + 3 * UID_SIZE,
            0,
            "shrink_to_fit gives exact capacity",
        );
    }
}

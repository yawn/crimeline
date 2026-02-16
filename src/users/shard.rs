use std::cmp::Ordering;

use crate::usage::{ReportUsage, Usage};
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

    /// Inserts exactly one (1) target. This will perform bad if used
    /// for multiple uids (instead of using merge()).
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
    ///
    /// The slow path uses an in-place forward merge: old data is shifted to the
    /// end of the buffer and then merged forward with `incoming`, reusing the
    /// existing allocation instead of allocating a second `Vec`.
    pub fn merge(&mut self, index: usize, incoming: &[Uid]) -> usize {
        let list = self.entry(index);

        // Fast path: all incoming > all existing — just append.
        if incoming
            .first()
            .zip(list.last())
            .is_some_and(|(&first, &last)| first > last)
        {
            let before = list.len();
            list.extend_from_slice(incoming);
            return list.len() - before;
        }

        let old_len = list.len();

        // Extend with zeros and shift old data to the end of the buffer.
        list.resize(old_len + incoming.len(), 0);
        list.copy_within(0..old_len, incoming.len());

        // Forward merge: old data sits at list[incoming.len()..total],
        // incoming is the external slice.
        //
        // Invariant: write <= i. Proof sketch — write starts at 0 and i starts
        // at incoming.len(), so (write − i) = −incoming.len(). Only the
        // Greater arm (incoming wins) increments write without advancing i,
        // and that arm executes at most incoming.len() times. Hence write ≤ i
        // throughout, and we never overwrite unread old data.
        let total = old_len + incoming.len();
        let mut write = 0;
        let mut i = incoming.len(); // start of old data in list
        let mut j = 0; // index into incoming

        while i < total && j < incoming.len() {
            let old_val = list[i];
            let inc_val = incoming[j];
            match old_val.cmp(&inc_val) {
                Ordering::Less => {
                    list[write] = old_val;
                    write += 1;
                    i += 1;
                }
                Ordering::Equal => {
                    list[write] = old_val;
                    write += 1;
                    i += 1;
                    j += 1;
                }
                Ordering::Greater => {
                    list[write] = inc_val;
                    write += 1;
                    j += 1;
                }
            }
        }

        // Copy remaining old data (already in list, may need shifting).
        if i < total {
            list.copy_within(i..total, write);
            write += total - i;
        }

        // Copy remaining incoming data.
        while j < incoming.len() {
            list[write] = incoming[j];
            write += 1;
            j += 1;
        }

        list.truncate(write);
        list.shrink_to_fit();

        write - old_len
    }
}

impl ReportUsage for Shard {
    fn usage(&self) -> Usage {
        let mut u = Usage::default();

        // Outer backbone: Vec<Vec<Uid>>.
        u.add_heap_usage(self.0.capacity() * size_of::<Vec<Uid>>());

        let empty = self.0.iter().filter(|v| v.is_empty()).count();
        let excess = self.0.capacity() - self.0.len();
        u.add_heap_waste((empty + excess) * size_of::<Vec<Uid>>());

        // Inner adjacency lists.
        for v in &self.0 {
            u.add_vec(v);
        }

        u
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const UID_SIZE: usize = size_of::<Uid>();
    const VEC_SIZE: usize = size_of::<Vec<Uid>>();

    fn assert_stats(shard: &Shard, heap: usize, waste: usize, reason: &str) {
        let u = shard.usage();
        assert_eq!(u.heap, heap, "heap: {reason}");
        assert_eq!(u.waste, waste, "waste: {reason}");
    }

    #[test]
    fn merge_fast_path_all_incoming_greater() {
        let mut s = Shard::new();
        s.merge(0, &[1, 2, 3]);

        let added = s.merge(0, &[10, 20, 30]);
        assert_eq!(added, 3, "all incoming are new");
        assert_eq!(s.get(0).unwrap(), &[1, 2, 3, 10, 20, 30]);
    }

    #[test]
    fn merge_fast_path_empty_existing() {
        let mut s = Shard::new();
        s.entry(0); // create empty list

        // empty list has no last(), so is_some_and returns false => slow path
        let added = s.merge(0, &[5, 10]);
        assert_eq!(added, 2);
        assert_eq!(s.get(0).unwrap(), &[5, 10]);
    }

    #[test]
    fn merge_fast_path_single_existing_single_incoming() {
        let mut s = Shard::new();
        s.insert(0, 1);

        let added = s.merge(0, &[2]);
        assert_eq!(added, 1);
        assert_eq!(s.get(0).unwrap(), &[1, 2]);
    }

    #[test]
    fn merge_slow_path_interleaved() {
        let mut s = Shard::new();
        s.merge(0, &[1, 5, 10]);

        let added = s.merge(0, &[3, 5, 7, 12]);
        assert_eq!(added, 3, "5 is duplicate");
        assert_eq!(s.get(0).unwrap(), &[1, 3, 5, 7, 10, 12]);
    }

    #[test]
    fn stats_delete_leaves_empty_slot() {
        let mut s = Shard::new();
        s.insert(0, 1);
        s.delete(0, 1);
        assert_stats(
            &s,
            VEC_SIZE + UID_SIZE,
            VEC_SIZE + UID_SIZE,
            "empty inner vec: backbone waste + unused inner capacity",
        );
    }

    #[test]
    fn stats_dense_subjects() {
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
}

use std::cmp::Ordering;

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

    pub fn heap_size(&self) -> usize {
        let outer = self.0.capacity() * size_of::<Vec<Uid>>();
        let inner: usize = self.0.iter().map(|v| v.capacity() * size_of::<Uid>()).sum();
        outer + inner
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

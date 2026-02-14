use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;
use tracing::{debug, trace};

use crate::stats::Stats;
use crate::users::{Uid, shard::Shard, sharding::Sharding};

pub struct UserMap {
    len: AtomicUsize,
    shard_bits: u32,
    shard_mask: u32,
    shards: Box<[RwLock<Shard>]>,
}

#[cfg_attr(coverage_nightly, coverage(off))]
impl Default for UserMap {
    fn default() -> Self {
        Self::new(Sharding::S64)
    }
}

impl UserMap {
    pub fn new(sharding: Sharding) -> Self {
        let count = sharding.count();

        let shards: Vec<_> = (0..count).map(|_| RwLock::new(Shard::new())).collect();

        debug!(shards = count, "created user map with sharding");

        UserMap {
            len: AtomicUsize::new(0),
            shard_bits: sharding.bits(),
            shard_mask: sharding.mask(),
            shards: shards.into_boxed_slice(),
        }
    }

    pub fn add(&self, principal: Uid, target: Uid) {
        let (s, idx) = self.find(principal);
        let mut shard = self.shards[s].write();

        if shard.insert(idx, target) {
            self.len.fetch_add(1, Ordering::Relaxed);

            trace!(
                principal,
                target,
                len = self.len(),
                "added target to principal"
            );
        }
    }

    pub fn add_bulk(&self, principal: Uid, targets: impl IntoIterator<Item = Uid>) {
        let mut incoming: Vec<Uid> = targets.into_iter().collect();
        incoming.sort_unstable();
        incoming.dedup();

        let (s, idx) = self.find(principal);
        let mut shard = self.shards[s].write();

        let added = shard.merge(idx, &incoming);

        if added > 0 {
            self.len.fetch_add(added, Ordering::Relaxed);

            trace!(
                principal,
                targets = incoming.len(),
                len = self.len(),
                "added multiple targets to principal"
            );
        }
    }

    pub fn contains(&self, principal: Uid, target: Uid) -> bool {
        let (s, idx) = self.find(principal);
        let shard = self.shards[s].read();

        // TODO: potential optimization bloom-prefilter
        let matched = shard
            .get(idx)
            .map_or(false, |targets| targets.binary_search(&target).is_ok());

        trace!(
            principal,
            target, matched, "search for targets for principal"
        );

        matched
    }

    #[inline]
    fn find(&self, user: Uid) -> (usize, usize) {
        (
            (user & self.shard_mask) as usize,
            (user >> self.shard_bits) as usize,
        )
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    pub fn remove(&self, principal: Uid, target: Uid) {
        let (s, idx) = self.find(principal);
        let mut shard = self.shards[s].write();

        if shard.delete(idx, target) {
            self.len.fetch_sub(1, Ordering::Relaxed);

            trace!(
                principal,
                target,
                len = self.len(),
                "removed target from principal"
            );
        }
    }
}

impl fmt::Display for UserMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut stats = Stats::new("locks", self.shards.len() * size_of::<RwLock<Shard>>());

        for s in self.shards.iter() {
            stats.observe(&*s.read());
        }

        write!(f, "{stats}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::{BTreeSet, HashMap};

    const ALL_SHARDINGS: [Sharding; 12] = [
        Sharding::S2,
        Sharding::S4,
        Sharding::S8,
        Sharding::S16,
        Sharding::S32,
        Sharding::S64,
        Sharding::S128,
        Sharding::S256,
        Sharding::S512,
        Sharding::S1024,
        Sharding::S2048,
        Sharding::S4096,
    ];

    fn sharding_strategy() -> impl Strategy<Value = Sharding> {
        (0..ALL_SHARDINGS.len()).prop_map(|i| ALL_SHARDINGS[i])
    }

    #[derive(Clone, Debug)]
    enum Op {
        Add(Uid, Uid),
        Remove(Uid, Uid),
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            (0..200u32, 0..200u32).prop_map(|(p, t)| Op::Add(p, t)),
            (0..200u32, 0..200u32).prop_map(|(p, t)| Op::Remove(p, t)),
        ]
    }

    fn apply_and_track(map: &UserMap, reference: &mut HashMap<Uid, BTreeSet<Uid>>, op: &Op) {
        match *op {
            Op::Add(p, t) => {
                map.add(p, t);
                reference.entry(p).or_default().insert(t);
            }
            Op::Remove(p, t) => {
                map.remove(p, t);
                reference.entry(p).or_default().remove(&t);
            }
        }
    }

    fn assert_matches(
        map: &UserMap,
        reference: &HashMap<Uid, BTreeSet<Uid>>,
        principals: &BTreeSet<Uid>,
        targets: &BTreeSet<Uid>,
    ) {
        for &p in principals {
            for &t in targets {
                let expected = reference.get(&p).map_or(false, |s| s.contains(&t));
                assert_eq!(
                    map.contains(p, t),
                    expected,
                    "contains({}, {}): expected {}",
                    p,
                    t,
                    expected,
                );
            }
        }
        let expected_len: usize = reference.values().map(|s| s.len()).sum();
        assert_eq!(map.len(), expected_len, "len mismatch");
    }

    #[test]
    fn edge_cases() {
        let cases: Vec<(&str, Vec<Op>, Vec<(Uid, Uid, bool)>, usize)> = vec![
            ("empty map", vec![], vec![(0, 0, false), (1, 1, false)], 0),
            (
                "self-reference",
                vec![Op::Add(5, 5)],
                vec![(5, 5, true), (5, 0, false)],
                1,
            ),
            (
                "large user ids",
                vec![Op::Add(1_000_000, 2_000_000)],
                vec![
                    (1_000_000, 2_000_000, true),
                    (1_000_000, 0, false),
                    (0, 2_000_000, false),
                ],
                1,
            ),
            (
                "shard boundary ids",
                vec![Op::Add(127, 128), Op::Add(128, 127)],
                vec![
                    (127, 128, true),
                    (128, 127, true),
                    (127, 127, false),
                    (128, 128, false),
                ],
                2,
            ),
            (
                "remove from empty",
                vec![Op::Remove(0, 1)],
                vec![(0, 1, false)],
                0,
            ),
            (
                "add remove add",
                vec![Op::Add(0, 1), Op::Remove(0, 1), Op::Add(0, 1)],
                vec![(0, 1, true)],
                1,
            ),
        ];

        for (name, ops, checks, expected_len) in &cases {
            let map = UserMap::new(Sharding::S128);
            for op in ops {
                match *op {
                    Op::Add(p, t) => map.add(p, t),
                    Op::Remove(p, t) => map.remove(p, t),
                }
            }
            for &(p, t, expected) in checks {
                assert_eq!(
                    map.contains(p, t),
                    expected,
                    "case '{}': contains({}, {}) expected {}",
                    name,
                    p,
                    t,
                    expected,
                );
            }
            assert_eq!(map.len(), *expected_len, "case '{}': len mismatch", name);
        }
    }

    proptest! {
        #[test]
        fn fuzz_bulk_equivalence(
            sharding in sharding_strategy(),
            principal in 0..200u32,
            existing in prop::collection::vec(0..1000u32, 0..50),
            incoming in prop::collection::vec(0..1000u32, 0..50),
        ) {
            let bulk_map = UserMap::new(sharding);
            let individual_map = UserMap::new(sharding);

            for &t in &existing {
                bulk_map.add(principal, t);
                individual_map.add(principal, t);
            }

            bulk_map.add_bulk(principal, incoming.iter().copied());
            for &t in &incoming {
                individual_map.add(principal, t);
            }

            let all_targets: BTreeSet<Uid> = existing.iter().chain(&incoming).copied().collect();
            for &t in &all_targets {
                prop_assert_eq!(
                    bulk_map.contains(principal, t),
                    individual_map.contains(principal, t),
                    "mismatch at ({}, {})", principal, t,
                );
            }
            prop_assert_eq!(bulk_map.len(), individual_map.len());
        }

        #[test]
        fn fuzz_operations(
            sharding in sharding_strategy(),
            ops in prop::collection::vec(op_strategy(), 0..80),
        ) {
            let map = UserMap::new(sharding);
            let mut reference: HashMap<Uid, BTreeSet<Uid>> = HashMap::new();

            let mut principals = BTreeSet::new();
            let mut targets = BTreeSet::new();

            for op in &ops {
                apply_and_track(&map, &mut reference, op);
                match *op {
                    Op::Add(p, t) | Op::Remove(p, t) => {
                        principals.insert(p);
                        targets.insert(t);
                    }
                }
            }

            principals.insert(999);
            targets.insert(999);

            assert_matches(&map, &reference, &principals, &targets);
        }
    }
}

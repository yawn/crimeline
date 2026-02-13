use crate::{Sharding, Uid, UserMap};

pub struct Relationships {
    blocks: UserMap,
    follows: UserMap,
}

impl Relationships {
    pub fn new(sharding: Sharding) -> Self {
        Relationships {
            blocks: UserMap::new(sharding),
            follows: UserMap::new(sharding),
        }
    }

    pub fn is_blocked_by_principal(&self, principal: Uid, target: Uid) -> bool {
        self.blocks.contains(target, principal)
    }

    pub fn is_mutual_of_principal(&self, principal: Uid, target: Uid) -> bool {
        self.blocks.contains(principal, target) && self.follows.contains(target, principal)
    }

    pub fn is_followed_by_principal(&self, principal: Uid, target: Uid) -> bool {
        self.follows.contains(target, principal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::BTreeSet;

    struct Case {
        name: &'static str,
        blocks: Vec<(Uid, Uid)>,
        follows: Vec<(Uid, Uid)>,
        checks: Vec<(Uid, Uid, bool)>,
    }

    proptest! {
        #[test]
        fn fuzz_mutual(
            block_pairs in prop::collection::vec((0..100u32, 0..100u32), 0..40),
            follow_pairs in prop::collection::vec((0..100u32, 0..100u32), 0..40),
            queries in prop::collection::vec((0..100u32, 0..100u32), 1..20),
        ) {
            let rel = Relationships::new(Sharding::S64);

            let mut ref_blocks: BTreeSet<(Uid, Uid)> = BTreeSet::new();
            let mut ref_follows: BTreeSet<(Uid, Uid)> = BTreeSet::new();

            for &(p, t) in &block_pairs {
                rel.blocks.add(p, t);
                ref_blocks.insert((p, t));
            }
            for &(p, t) in &follow_pairs {
                rel.follows.add(p, t);
                ref_follows.insert((p, t));
            }

            for &(principal, target) in &queries {
                let expected =
                    ref_blocks.contains(&(principal, target))
                    && ref_follows.contains(&(target, principal));

                prop_assert_eq!(
                    rel.is_mutual_of_principal(principal, target),
                    expected,
                    "mutual({}, {}): expected {}",
                    principal,
                    target,
                    expected,
                );
            }
        }
    }

    #[test]
    fn mutual_edge_cases() {
        let cases = vec![
            Case {
                name: "empty relationships",
                blocks: vec![],
                follows: vec![],
                checks: vec![(0, 1, false), (1, 0, false)],
            },
            Case {
                name: "block only, no follow back",
                blocks: vec![(1, 2)],
                follows: vec![],
                checks: vec![(1, 2, false), (2, 1, false)],
            },
            Case {
                name: "follow only, no block",
                blocks: vec![],
                follows: vec![(2, 1)],
                checks: vec![(1, 2, false), (2, 1, false)],
            },
            Case {
                name: "block and follow back => mutual",
                blocks: vec![(1, 2)],
                follows: vec![(2, 1)],
                checks: vec![(1, 2, true), (2, 1, false)],
            },
            Case {
                name: "symmetric blocks and follows",
                blocks: vec![(1, 2), (2, 1)],
                follows: vec![(1, 2), (2, 1)],
                checks: vec![(1, 2, true), (2, 1, true)],
            },
            Case {
                name: "self-reference block and follow",
                blocks: vec![(5, 5)],
                follows: vec![(5, 5)],
                checks: vec![(5, 5, true)],
            },
            Case {
                name: "same direction block and follow — not mutual",
                blocks: vec![(2, 1)],
                follows: vec![(2, 1)],
                checks: vec![(1, 2, false), (2, 1, false)],
            },
            Case {
                name: "multiple principals, isolated",
                blocks: vec![(10, 20), (30, 40)],
                follows: vec![(20, 10), (40, 30)],
                checks: vec![
                    (10, 20, true),
                    (30, 40, true),
                    (10, 40, false),
                    (30, 20, false),
                ],
            },
            Case {
                name: "block removed breaks mutual",
                blocks: vec![],
                follows: vec![(2, 1)],
                checks: vec![(1, 2, false)],
            },
        ];

        for case in &cases {
            let rel = Relationships::new(Sharding::S64);

            for &(p, t) in &case.blocks {
                rel.blocks.add(p, t);
            }
            for &(p, t) in &case.follows {
                rel.follows.add(p, t);
            }

            for &(principal, target, expected) in &case.checks {
                assert_eq!(
                    rel.is_mutual_of_principal(principal, target),
                    expected,
                    "case '{}': mutual({}, {}) expected {}",
                    case.name,
                    principal,
                    target,
                    expected,
                );
            }
        }
    }
}

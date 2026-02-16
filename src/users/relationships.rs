use crate::{Sharding, Uid, UserMap};

pub struct Relationships {
    pub blocks: UserMap,
    pub follows: UserMap,
}

impl Relationships {
    pub fn new(sharding: Sharding) -> Self {
        Relationships {
            blocks: UserMap::new(sharding),
            follows: UserMap::new(sharding),
        }
    }

    pub fn is_blocked_by(&self, subject: Uid, target: Uid) -> bool {
        self.blocks.contains(target, subject)
    }

    pub fn is_mutual(&self, subject: Uid, target: Uid) -> bool {
        self.blocks.contains(subject, target) && self.follows.contains(target, subject)
    }

    pub fn is_followed_by(&self, subject: Uid, target: Uid) -> bool {
        self.follows.contains(target, subject)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::BTreeSet;

    #[derive(Clone, Debug)]
    enum Op {
        Block(Uid, Uid),
        Unblock(Uid, Uid),
        Follow(Uid, Uid),
        Unfollow(Uid, Uid),
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            (0..100u32, 0..100u32).prop_map(|(p, t)| Op::Block(p, t)),
            (0..100u32, 0..100u32).prop_map(|(p, t)| Op::Unblock(p, t)),
            (0..100u32, 0..100u32).prop_map(|(p, t)| Op::Follow(p, t)),
            (0..100u32, 0..100u32).prop_map(|(p, t)| Op::Unfollow(p, t)),
        ]
    }

    proptest! {
        #[test]
        fn fuzz_relationships(
            ops in prop::collection::vec(op_strategy(), 0..80),
            queries in prop::collection::vec((0..100u32, 0..100u32), 1..30),
        ) {
            let rel = Relationships::new(Sharding::S64);

            let mut ref_blocks: BTreeSet<(Uid, Uid)> = BTreeSet::new();
            let mut ref_follows: BTreeSet<(Uid, Uid)> = BTreeSet::new();

            for op in &ops {
                match *op {
                    Op::Block(p, t) => {
                        rel.blocks.add(p, t);
                        ref_blocks.insert((p, t));
                    }
                    Op::Unblock(p, t) => {
                        rel.blocks.remove(p, t);
                        ref_blocks.remove(&(p, t));
                    }
                    Op::Follow(p, t) => {
                        rel.follows.add(p, t);
                        ref_follows.insert((p, t));
                    }
                    Op::Unfollow(p, t) => {
                        rel.follows.remove(p, t);
                        ref_follows.remove(&(p, t));
                    }
                }
            }

            for &(subject, target) in &queries {
                let expected_blocked = ref_blocks.contains(&(target, subject));
                prop_assert_eq!(
                    rel.is_blocked_by(subject, target),
                    expected_blocked,
                    "is_blocked_by({}, {})", subject, target,
                );

                let expected_followed = ref_follows.contains(&(target, subject));
                prop_assert_eq!(
                    rel.is_followed_by(subject, target),
                    expected_followed,
                    "is_followed_by({}, {})", subject, target,
                );

                let expected_mutual =
                    ref_blocks.contains(&(subject, target))
                    && ref_follows.contains(&(target, subject));
                prop_assert_eq!(
                    rel.is_mutual(subject, target),
                    expected_mutual,
                    "is_mutual({}, {})", subject, target,
                );
            }
        }
    }
}

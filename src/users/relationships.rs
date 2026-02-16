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

    fn rel() -> Relationships {
        Relationships::new(Sharding::S64)
    }

    fn assert_blocked(
        rel: &Relationships,
        subject: Uid,
        target: Uid,
        expected: bool,
        label: &str,
    ) {
        assert_eq!(
            rel.is_blocked_by(subject, target),
            expected,
            "{label}: is_blocked_by({subject}, {target})",
        );
    }

    fn assert_followed(
        rel: &Relationships,
        subject: Uid,
        target: Uid,
        expected: bool,
        label: &str,
    ) {
        assert_eq!(
            rel.is_followed_by(subject, target),
            expected,
            "{label}: is_followed_by({subject}, {target})",
        );
    }

    fn assert_mutual(
        rel: &Relationships,
        subject: Uid,
        target: Uid,
        expected: bool,
        label: &str,
    ) {
        assert_eq!(
            rel.is_mutual(subject, target),
            expected,
            "{label}: is_mutual({subject}, {target})",
        );
    }

    #[test]
    fn blocked_asymmetric() {
        let r = rel();
        r.blocks.add(1, 2);
        assert_blocked(&r, 2, 1, true, "target blocked by subject");
        assert_blocked(&r, 1, 2, false, "subject is not blocked by target");
    }

    #[test]
    fn blocked_empty() {
        let r = rel();
        assert_blocked(&r, 0, 1, false, "no blocks");
    }

    #[test]
    fn blocked_self_reference() {
        let r = rel();
        r.blocks.add(5, 5);
        assert_blocked(&r, 5, 5, true, "self-block");
    }

    #[test]
    fn blocked_symmetric() {
        let r = rel();
        r.blocks.add(1, 2);
        r.blocks.add(2, 1);
        assert_blocked(&r, 1, 2, true, "1 blocked by 2");
        assert_blocked(&r, 2, 1, true, "2 blocked by 1");
    }

    #[test]
    fn followed_asymmetric() {
        let r = rel();
        r.follows.add(2, 1);
        assert_followed(&r, 1, 2, true, "target follows subject");
        assert_followed(&r, 2, 1, false, "subject does not follow target");
    }

    #[test]
    fn followed_empty() {
        let r = rel();
        assert_followed(&r, 0, 1, false, "no follows");
    }

    #[test]
    fn followed_self_reference() {
        let r = rel();
        r.follows.add(5, 5);
        assert_followed(&r, 5, 5, true, "self-follow");
    }

    #[test]
    fn followed_symmetric() {
        let r = rel();
        r.follows.add(1, 2);
        r.follows.add(2, 1);
        assert_followed(&r, 1, 2, true, "1 followed by 2");
        assert_followed(&r, 2, 1, true, "2 followed by 1");
    }

    #[test]
    fn mutual_block_and_follow_back() {
        let r = rel();
        r.blocks.add(1, 2);
        r.follows.add(2, 1);
        assert_mutual(&r, 1, 2, true, "block + follow back => mutual");
        assert_mutual(&r, 2, 1, false, "reverse direction is not mutual");
    }

    #[test]
    fn mutual_block_only() {
        let r = rel();
        r.blocks.add(1, 2);
        assert_mutual(&r, 1, 2, false, "block without follow");
        assert_mutual(&r, 2, 1, false, "reverse");
    }

    #[test]
    fn mutual_empty() {
        let r = rel();
        assert_mutual(&r, 0, 1, false, "no relationships");
        assert_mutual(&r, 1, 0, false, "reverse");
    }

    #[test]
    fn mutual_follow_only() {
        let r = rel();
        r.follows.add(2, 1);
        assert_mutual(&r, 1, 2, false, "follow without block");
        assert_mutual(&r, 2, 1, false, "reverse");
    }

    #[test]
    fn mutual_missing_block_breaks_mutual() {
        let r = rel();
        r.follows.add(2, 1);
        assert_mutual(&r, 1, 2, false, "follow present but no block");
    }

    #[test]
    fn mutual_multiple_subjects_isolated() {
        let r = rel();
        r.blocks.add(10, 20);
        r.blocks.add(30, 40);
        r.follows.add(20, 10);
        r.follows.add(40, 30);
        assert_mutual(&r, 10, 20, true, "pair 10-20");
        assert_mutual(&r, 30, 40, true, "pair 30-40");
        assert_mutual(&r, 10, 40, false, "cross-pair 10-40");
        assert_mutual(&r, 30, 20, false, "cross-pair 30-20");
    }

    #[test]
    fn mutual_same_direction_not_mutual() {
        let r = rel();
        r.blocks.add(2, 1);
        r.follows.add(2, 1);
        assert_mutual(&r, 1, 2, false, "same direction block+follow");
        assert_mutual(&r, 2, 1, false, "reverse");
    }

    #[test]
    fn mutual_self_reference() {
        let r = rel();
        r.blocks.add(5, 5);
        r.follows.add(5, 5);
        assert_mutual(&r, 5, 5, true, "self-block + self-follow");
    }

    #[test]
    fn mutual_symmetric() {
        let r = rel();
        r.blocks.add(1, 2);
        r.blocks.add(2, 1);
        r.follows.add(1, 2);
        r.follows.add(2, 1);
        assert_mutual(&r, 1, 2, true, "1 mutual with 2");
        assert_mutual(&r, 2, 1, true, "2 mutual with 1");
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

            for &(subject, target) in &queries {
                let expected =
                    ref_blocks.contains(&(subject, target))
                    && ref_follows.contains(&(target, subject));

                prop_assert_eq!(
                    rel.is_mutual(subject, target),
                    expected,
                    "mutual({}, {}): expected {}",
                    subject,
                    target,
                    expected,
                );
            }
        }
    }
}

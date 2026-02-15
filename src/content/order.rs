#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    Asc,
    Desc,
}

impl Order {
    pub fn sort_unstable_by_key<T, K: Ord>(&self, slice: &mut [T], mut f: impl FnMut(&T) -> K) {
        match self {
            Order::Asc => slice.sort_unstable_by(|a, b| f(a).cmp(&f(b))),
            Order::Desc => slice.sort_unstable_by(|a, b| f(b).cmp(&f(a))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn sort_unstable_by_key_orders_correctly(mut v: Vec<u64>) {
            let mut asc = v.clone();
            Order::Asc.sort_unstable_by_key(&mut asc, |x| *x);
            assert!(asc.windows(2).all(|w| w[0] <= w[1]), "asc not monotonic");

            Order::Desc.sort_unstable_by_key(&mut v, |x| *x);
            assert!(v.windows(2).all(|w| w[0] >= w[1]), "desc not monotonic");

            // same elements, opposite order
            v.reverse();
            assert_eq!(v, asc, "desc reversed should equal asc");
        }
    }
}

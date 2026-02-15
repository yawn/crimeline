use std::iter::Rev;
use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    Asc,
    Desc,
}

pub enum OrderedRange {
    Asc(Range<usize>),
    Desc(Rev<Range<usize>>),
}

impl Order {
    pub fn range(self, r: Range<usize>) -> OrderedRange {
        match self {
            Order::Asc => OrderedRange::Asc(r),
            Order::Desc => OrderedRange::Desc(r.rev()),
        }
    }

    pub fn sort_unstable_by_key<T, K: Ord>(&self, slice: &mut [T], mut f: impl FnMut(&T) -> K) {
        match self {
            Order::Asc => slice.sort_unstable_by(|a, b| f(a).cmp(&f(b))),
            Order::Desc => slice.sort_unstable_by(|a, b| f(b).cmp(&f(a))),
        }
    }
}

impl Iterator for OrderedRange {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        match self {
            OrderedRange::Asc(r) => r.next(),
            OrderedRange::Desc(r) => r.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            OrderedRange::Asc(r) => r.size_hint(),
            OrderedRange::Desc(r) => r.size_hint(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn range_asc() {
        let v: Vec<usize> = Order::Asc.range(2..5).collect();
        assert_eq!(v, vec![2, 3, 4]);
    }

    #[test]
    fn range_desc() {
        let v: Vec<usize> = Order::Desc.range(2..5).collect();
        assert_eq!(v, vec![4, 3, 2]);
    }

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

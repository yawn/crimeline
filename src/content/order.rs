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
            Order::Asc => slice.sort_unstable_by_key(|a| f(a)),
            Order::Desc => slice.sort_unstable_by_key(|b| std::cmp::Reverse(f(b))),
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

    proptest! {
        #[test]
        fn fuzz_sort_and_range(mut v: Vec<u64>, start in 0..50usize, len in 0..50usize) {
            // sort: asc and desc produce inverse orderings of the same elements
            let mut asc = v.clone();
            Order::Asc.sort_unstable_by_key(&mut asc, |x| *x);
            prop_assert!(asc.windows(2).all(|w| w[0] <= w[1]), "asc not monotonic");

            Order::Desc.sort_unstable_by_key(&mut v, |x| *x);
            prop_assert!(v.windows(2).all(|w| w[0] >= w[1]), "desc not monotonic");

            v.reverse();
            prop_assert_eq!(&v, &asc, "desc reversed should equal asc");

            // range: asc and desc yield same elements in opposite order
            let end = start + len;
            let fwd: Vec<usize> = Order::Asc.range(start..end).collect();
            let mut rev: Vec<usize> = Order::Desc.range(start..end).collect();
            rev.reverse();
            prop_assert_eq!(&fwd, &rev, "range elements should match");

            if !fwd.is_empty() {
                prop_assert_eq!(fwd[0], start);
                prop_assert_eq!(*fwd.last().unwrap(), end - 1);
            }
        }
    }
}

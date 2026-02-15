use std::fmt;
use std::ops::AddAssign;

use humansize::{BINARY, SizeFormatter};

#[derive(Default)]
pub struct Usage {
    active: usize,
    count: usize,
    label: &'static str,
    max: usize,
    min: usize,
    overhead: usize,
    pub disk: u64,
    pub heap: usize,
    pub waste: usize,
}

pub trait ReportUsage {
    fn usage(&self) -> Usage;
}

impl Usage {
    pub fn new(label: &'static str, overhead: usize) -> Self {
        Usage {
            label,
            overhead,
            ..Default::default()
        }
    }

    pub fn add_boxed_slice<T>(&mut self, s: &[T]) {
        self.heap += s.len() * size_of::<T>();
    }

    pub fn add_heap_usage(&mut self, bytes: usize) {
        self.heap += bytes;
    }

    pub fn add_heap_waste(&mut self, bytes: usize) {
        self.waste += bytes;
    }

    pub fn add_disk_usage(&mut self, bytes: u64) {
        self.disk += bytes;
    }

    #[allow(clippy::ptr_arg)]
    pub fn add_vec<T>(&mut self, v: &Vec<T>) {
        self.heap += v.capacity() * size_of::<T>();
        self.waste += (v.capacity() - v.len()) * size_of::<T>();
    }

    pub fn observe<T: ReportUsage>(&mut self, item: &T) {
        let u = item.usage();

        self.heap += u.heap;
        self.waste += u.waste;
        self.disk += u.disk;
        self.count += 1;

        if u.heap > 0 {
            self.active += 1;
        }

        if self.count == 1 {
            self.min = u.heap;
            self.max = u.heap;
        } else {
            self.min = self.min.min(u.heap);
            self.max = self.max.max(u.heap);
        }
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, rhs: Self) {
        self.heap += rhs.heap;
        self.waste += rhs.waste;
        self.disk += rhs.disk;
    }
}

impl fmt::Display for Usage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let min = if self.count == 0 { 0 } else { self.min };

        write!(
            f,
            "{} ({} {} + {} data, {} wasted",
            SizeFormatter::new(self.overhead + self.heap, BINARY),
            SizeFormatter::new(self.overhead, BINARY),
            self.label,
            SizeFormatter::new(self.heap, BINARY),
            SizeFormatter::new(self.waste, BINARY),
        )?;

        if self.disk > 0 {
            write!(f, ", {} on disk", SizeFormatter::new(self.disk, BINARY),)?;
        }

        write!(
            f,
            ") across {} ({} active, {}..{})",
            self.count,
            self.active,
            SizeFormatter::new(min, BINARY),
            SizeFormatter::new(self.max, BINARY),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeItem {
        heap: usize,
        waste: usize,
        disk: u64,
    }

    impl ReportUsage for FakeItem {
        fn usage(&self) -> Usage {
            let mut u = Usage::default();
            u.add_heap_usage(self.heap);
            u.add_heap_waste(self.waste);
            u.add_disk_usage(self.disk);
            u
        }
    }

    #[test]
    fn add_assign_nests_totals_only() {
        let mut parent = Usage::new("test", 0);
        parent.observe(&FakeItem {
            heap: 100,
            waste: 10,
            disk: 0,
        });

        let mut child = Usage::default();
        child.add_heap_usage(50);
        child.add_heap_waste(5);
        child.add_disk_usage(1000);

        parent += child;

        assert_eq!(parent.heap, 150);
        assert_eq!(parent.waste, 15);
        assert_eq!(parent.disk, 1000);
        // count/active/min/max unchanged by +=
        assert_eq!(parent.count, 1);
        assert_eq!(parent.active, 1);
        assert_eq!(parent.min, 100);
        assert_eq!(parent.max, 100);
    }

    #[test]
    fn add_boxed_slice_no_waste() {
        let s: Box<[u32]> = vec![1, 2, 3].into_boxed_slice();

        let mut u = Usage::default();
        u.add_boxed_slice(&s);

        assert_eq!(u.heap, 3 * size_of::<u32>());
        assert_eq!(u.waste, 0);
    }

    #[test]
    fn add_vec_accounts_for_capacity() {
        let mut v: Vec<u32> = Vec::with_capacity(10);
        v.extend_from_slice(&[1, 2, 3]);

        let mut u = Usage::default();
        u.add_vec(&v);

        assert_eq!(u.heap, 10 * size_of::<u32>());
        assert_eq!(u.waste, 7 * size_of::<u32>());
    }

    #[test]
    fn display_empty() {
        let usage = Usage::new("locks", 2048);
        assert_eq!(
            format!("{usage}"),
            "2 KiB (2 KiB locks + 0 B data, 0 B wasted) across 0 (0 active, 0 B..0 B)",
        );
    }

    #[test]
    fn display_mixed_items() {
        let mut usage = Usage::new("locks", 64);
        for item in &[
            FakeItem {
                heap: 100,
                waste: 24,
                disk: 0,
            },
            FakeItem {
                heap: 0,
                waste: 0,
                disk: 0,
            },
            FakeItem {
                heap: 300,
                waste: 48,
                disk: 0,
            },
        ] {
            usage.observe(item);
        }
        assert_eq!(
            format!("{usage}"),
            "464 B (64 B locks + 400 B data, 72 B wasted) across 3 (2 active, 0 B..300 B)",
        );
    }

    #[test]
    fn display_single_item() {
        let mut usage = Usage::new("locks", 64);
        usage.observe(&FakeItem {
            heap: 1024,
            waste: 0,
            disk: 0,
        });
        assert_eq!(
            format!("{usage}"),
            "1.06 KiB (64 B locks + 1 KiB data, 0 B wasted) across 1 (1 active, 1 KiB..1 KiB)",
        );
    }

    #[test]
    fn display_with_disk() {
        let mut usage = Usage::new("locks", 64);
        usage.observe(&FakeItem {
            heap: 512,
            waste: 32,
            disk: 4096,
        });
        assert_eq!(
            format!("{usage}"),
            "576 B (64 B locks + 512 B data, 32 B wasted, 4 KiB on disk) across 1 (1 active, 512 B..512 B)",
        );
    }

    #[test]
    fn display_with_waste() {
        let mut usage = Usage::new("locks", 64);
        usage.observe(&FakeItem {
            heap: 1024,
            waste: 128,
            disk: 0,
        });
        assert_eq!(
            format!("{usage}"),
            "1.06 KiB (64 B locks + 1 KiB data, 128 B wasted) across 1 (1 active, 1 KiB..1 KiB)",
        );
    }
}

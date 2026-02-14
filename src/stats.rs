use std::fmt;

use humansize::{BINARY, SizeFormatter};

pub(crate) trait HeapUsage {
    fn heap_usage(&self) -> usize;
}

pub(crate) trait HeapWaste: HeapUsage {
    fn heap_waste(&self) -> usize;
}

pub(crate) struct Stats {
    active: usize,
    count: usize,
    data: usize,
    label: &'static str,
    max: usize,
    min: usize,
    overhead: usize,
    wasted: usize,
}

impl Stats {
    pub fn new(label: &'static str, overhead: usize) -> Self {
        Stats {
            active: 0,
            count: 0,
            data: 0,
            label,
            max: 0,
            min: usize::MAX,
            overhead,
            wasted: 0,
        }
    }

    pub fn observe<T: HeapWaste>(&mut self, item: &T) {
        let usage = item.heap_usage();

        self.data += usage;
        self.wasted += item.heap_waste();
        self.count += 1;

        if usage > 0 {
            self.active += 1;
        }

        self.min = self.min.min(usage);
        self.max = self.max.max(usage);
    }
}

impl fmt::Display for Stats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let min = if self.count == 0 { 0 } else { self.min };

        write!(
            f,
            "{} ({} {} + {} data, {} wasted) across {} shards ({} active, shard size {}..{})",
            SizeFormatter::new(self.overhead + self.data, BINARY),
            SizeFormatter::new(self.overhead, BINARY),
            self.label,
            SizeFormatter::new(self.data, BINARY),
            SizeFormatter::new(self.wasted, BINARY),
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
        usage: usize,
        waste: usize,
    }

    impl HeapUsage for FakeItem {
        fn heap_usage(&self) -> usize {
            self.usage
        }
    }

    impl HeapWaste for FakeItem {
        fn heap_waste(&self) -> usize {
            self.waste
        }
    }

    #[test]
    fn display_empty() {
        let stats = Stats::new("locks", 2048);
        assert_eq!(
            format!("{stats}"),
            "2 KiB (2 KiB locks + 0 B data, 0 B wasted) across 0 shards (0 active, shard size 0 B..0 B)",
        );
    }

    #[test]
    fn display_single_shard() {
        let mut stats = Stats::new("locks", 64);
        stats.observe(&FakeItem {
            usage: 1024,
            waste: 0,
        });
        assert_eq!(
            format!("{stats}"),
            "1.06 KiB (64 B locks + 1 KiB data, 0 B wasted) across 1 shards (1 active, shard size 1 KiB..1 KiB)",
        );
    }

    #[test]
    fn display_mixed_shards() {
        let mut stats = Stats::new("locks", 64);
        stats.observe(&FakeItem {
            usage: 100,
            waste: 24,
        });
        stats.observe(&FakeItem { usage: 0, waste: 0 });
        stats.observe(&FakeItem {
            usage: 300,
            waste: 48,
        });
        assert_eq!(
            format!("{stats}"),
            "464 B (64 B locks + 400 B data, 72 B wasted) across 3 shards (2 active, shard size 0 B..300 B)",
        );
    }
}

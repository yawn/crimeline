use super::Timestamp;

#[derive(Debug, PartialEq, Eq)]
pub struct Window {
    pub duration: u32,
    pub epoch: Timestamp,
}

impl Window {
    pub fn new(epoch: Timestamp, duration: u32) -> Self {
        Self { epoch, duration }
    }

    pub fn contains(&self, ts: Timestamp) -> bool {
        ts >= self.epoch && ts < self.end_exclusive()
    }

    pub fn convert_to_absolute(&self, rel: u32) -> Timestamp {
        self.epoch + rel as u64
    }

    pub fn convert_to_relative(&self, ts: Timestamp) -> u32 {
        debug_assert!(
            self.contains(ts),
            "timestamp {ts} outside [{}, {})",
            self.epoch,
            self.end_exclusive(),
        );
        (ts - self.epoch) as u32
    }

    pub fn end_exclusive(&self) -> Timestamp {
        self.epoch + self.duration as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn fuzz_window(
            epoch in 0..u64::MAX / 2,
            duration in 1..100_000u32,
            offset in 0..100_000u64,
        ) {
            let span = Window::new(epoch, duration);

            // end_exclusive is always epoch + duration
            prop_assert_eq!(span.end_exclusive(), epoch + duration as u64);

            let ts = epoch + offset;

            // contains is [epoch, epoch+duration)
            let expected = ts >= epoch && ts < epoch + duration as u64;
            prop_assert_eq!(span.contains(ts), expected, "contains({})", ts);

            // roundtrip for timestamps inside the window
            if expected {
                let rel = span.convert_to_relative(ts);
                prop_assert_eq!(span.convert_to_absolute(rel), ts, "roundtrip({})", ts);
            }
        }
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn relative_outside_panics_in_debug() {
        let span = Window::new(1000, 100);
        span.convert_to_relative(1100);
    }

    #[test]
    fn zero_duration_contains_nothing() {
        let span = Window::new(1000, 0);
        assert!(!span.contains(1000));
        assert!(!span.contains(999));
        assert_eq!(span.end_exclusive(), 1000);
    }
}

use super::Timestamp;

#[derive(Debug, PartialEq, Eq)]
pub struct TimeSpan {
    pub duration: u32,
    pub epoch: Timestamp,
}

impl TimeSpan {
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

    #[test]
    fn absolute_roundtrip() {
        let span = TimeSpan::new(1000, 100);
        let ts: Timestamp = 1042;
        assert_eq!(span.convert_to_absolute(span.convert_to_relative(ts)), ts);
    }

    #[test]
    fn absolute_at_zero() {
        let span = TimeSpan::new(1000, 100);
        assert_eq!(span.convert_to_absolute(0), 1000);
    }

    #[test]
    fn contains_before_epoch() {
        let span = TimeSpan::new(1000, 100);
        assert!(!span.contains(999));
    }

    #[test]
    fn contains_start_inclusive() {
        let span = TimeSpan::new(1000, 100);
        assert!(span.contains(1000), "start should be inclusive");
    }

    #[test]
    fn contains_end_exclusive() {
        let span = TimeSpan::new(1000, 100);
        assert!(!span.contains(1100), "end should be exclusive");
    }

    #[test]
    fn contains_mid() {
        let span = TimeSpan::new(1000, 100);
        assert!(span.contains(1050));
    }

    #[test]
    fn end_equals_epoch_plus_duration() {
        let span = TimeSpan::new(500, 200);
        assert_eq!(span.end_exclusive(), 700);
    }

    #[test]
    fn relative_at_epoch() {
        let span = TimeSpan::new(1000, 100);
        assert_eq!(span.convert_to_relative(1000), 0);
    }

    #[test]
    fn relative_mid() {
        let span = TimeSpan::new(1000, 100);
        assert_eq!(span.convert_to_relative(1042), 42);
    }

    #[test]
    #[should_panic(expected = "outside")]
    fn relative_outside_panics_in_debug() {
        let span = TimeSpan::new(1000, 100);
        span.convert_to_relative(1100);
    }

    #[test]
    fn zero_duration_contains_nothing() {
        let span = TimeSpan::new(1000, 0);
        assert!(!span.contains(1000));
        assert!(!span.contains(999));
        assert_eq!(span.end_exclusive(), 1000);
    }
}

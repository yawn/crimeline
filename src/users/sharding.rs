#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Sharding {
    S2 = 1,
    S4 = 2,
    S8 = 3,
    S16 = 4,
    S32 = 5,
    S64 = 6,
    S128 = 7,
    S256 = 8,
    S512 = 9,
    S1024 = 10,
    S2048 = 11,
    S4096 = 12,
}

impl Sharding {
    #[inline]
    pub fn bits(self) -> u32 {
        self as u32
    }

    pub fn count(&self) -> usize {
        1usize << self.bits()
    }

    #[inline]
    pub fn mask(self) -> u32 {
        (1u32 << self.bits()) - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_count_mask() {
        let cases: Vec<(Sharding, u32, usize, u32)> = vec![
            //            bits  count  mask
            (Sharding::S2, 1, 2, 0b1),
            (Sharding::S4, 2, 4, 0b11),
            (Sharding::S8, 3, 8, 0b111),
            (Sharding::S16, 4, 16, 0b1111),
            (Sharding::S32, 5, 32, 0b1_1111),
            (Sharding::S64, 6, 64, 0b11_1111),
            (Sharding::S128, 7, 128, 0b111_1111),
            (Sharding::S256, 8, 256, 0b1111_1111),
            (Sharding::S512, 9, 512, 0b1_1111_1111),
            (Sharding::S1024, 10, 1024, 0b11_1111_1111),
            (Sharding::S2048, 11, 2048, 0b111_1111_1111),
            (Sharding::S4096, 12, 4096, 0b1111_1111_1111),
        ];

        for (sharding, expected_bits, expected_count, expected_mask) in &cases {
            assert_eq!(
                sharding.bits(),
                *expected_bits,
                "{:?}: bits",
                sharding,
            );
            assert_eq!(
                sharding.count(),
                *expected_count,
                "{:?}: count",
                sharding,
            );
            assert_eq!(
                sharding.mask(),
                *expected_mask,
                "{:?}: mask",
                sharding,
            );
        }
    }

    #[test]
    fn mask_is_count_minus_one() {
        let all = [
            Sharding::S2, Sharding::S4, Sharding::S8, Sharding::S16,
            Sharding::S32, Sharding::S64, Sharding::S128, Sharding::S256,
            Sharding::S512, Sharding::S1024, Sharding::S2048, Sharding::S4096,
        ];

        for s in &all {
            assert_eq!(
                s.mask() as usize,
                s.count() - 1,
                "{:?}: mask should equal count - 1",
                s,
            );
        }
    }

    #[test]
    fn variant_name_matches_count() {
        let cases: Vec<(Sharding, &str)> = vec![
            (Sharding::S2, "S2"),
            (Sharding::S4, "S4"),
            (Sharding::S8, "S8"),
            (Sharding::S16, "S16"),
            (Sharding::S32, "S32"),
            (Sharding::S64, "S64"),
            (Sharding::S128, "S128"),
            (Sharding::S256, "S256"),
            (Sharding::S512, "S512"),
            (Sharding::S1024, "S1024"),
            (Sharding::S2048, "S2048"),
            (Sharding::S4096, "S4096"),
        ];

        for (sharding, name) in &cases {
            let n: usize = name[1..].parse().unwrap();
            assert_eq!(
                sharding.count(),
                n,
                "{:?}: variant name implies count {}",
                sharding,
                n,
            );
        }
    }
}

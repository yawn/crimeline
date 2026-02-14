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

    const ALL: [Sharding; 12] = [
        Sharding::S2,
        Sharding::S4,
        Sharding::S8,
        Sharding::S16,
        Sharding::S32,
        Sharding::S64,
        Sharding::S128,
        Sharding::S256,
        Sharding::S512,
        Sharding::S1024,
        Sharding::S2048,
        Sharding::S4096,
    ];

    fn assert_sharding(s: Sharding, bits: u32, count: usize, mask: u32, label: &str) {
        assert_eq!(s.bits(), bits, "{label}: bits");
        assert_eq!(s.count(), count, "{label}: count");
        assert_eq!(s.mask(), mask, "{label}: mask");
    }

    #[test]
    fn bits_count_mask_s1024() {
        assert_sharding(Sharding::S1024, 10, 1024, 0b11_1111_1111, "S1024");
    }

    #[test]
    fn bits_count_mask_s128() {
        assert_sharding(Sharding::S128, 7, 128, 0b111_1111, "S128");
    }

    #[test]
    fn bits_count_mask_s16() {
        assert_sharding(Sharding::S16, 4, 16, 0b1111, "S16");
    }

    #[test]
    fn bits_count_mask_s2() {
        assert_sharding(Sharding::S2, 1, 2, 0b1, "S2");
    }

    #[test]
    fn bits_count_mask_s2048() {
        assert_sharding(Sharding::S2048, 11, 2048, 0b111_1111_1111, "S2048");
    }

    #[test]
    fn bits_count_mask_s256() {
        assert_sharding(Sharding::S256, 8, 256, 0b1111_1111, "S256");
    }

    #[test]
    fn bits_count_mask_s32() {
        assert_sharding(Sharding::S32, 5, 32, 0b1_1111, "S32");
    }

    #[test]
    fn bits_count_mask_s4() {
        assert_sharding(Sharding::S4, 2, 4, 0b11, "S4");
    }

    #[test]
    fn bits_count_mask_s4096() {
        assert_sharding(Sharding::S4096, 12, 4096, 0b1111_1111_1111, "S4096");
    }

    #[test]
    fn bits_count_mask_s512() {
        assert_sharding(Sharding::S512, 9, 512, 0b1_1111_1111, "S512");
    }

    #[test]
    fn bits_count_mask_s64() {
        assert_sharding(Sharding::S64, 6, 64, 0b11_1111, "S64");
    }

    #[test]
    fn bits_count_mask_s8() {
        assert_sharding(Sharding::S8, 3, 8, 0b111, "S8");
    }

    #[test]
    fn mask_equals_count_minus_one() {
        for s in &ALL {
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
        for s in &ALL {
            let name = format!("{s:?}");
            let n: usize = name[1..].parse().unwrap();
            assert_eq!(s.count(), n, "{name}: variant name implies count {n}");
        }
    }
}

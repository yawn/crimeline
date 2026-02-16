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
    use anyhow::Result;

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
    fn variant_name_matches_count() -> Result<()> {
        for s in &ALL {
            let name = format!("{s:?}");
            let n: usize = name[1..].parse()?;
            assert_eq!(s.count(), n, "{name}: variant name implies count {n}");
        }
        Ok(())
    }
}

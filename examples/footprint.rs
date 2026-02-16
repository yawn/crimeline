use std::mem::size_of;
use std::sync::Arc;

use anyhow::Result;
use crimeline::{
    Order, ReportUsage, Timeline, Uid, Usage, Window,
    arena::{Cold, Hot},
};

// ── Bluesky content statistics ───────────────────────────────────────────────
//
// Sources: Bluesky 2025 Transparency Report (Jan 2026), Jaz's bsky stats
//          (Aug 2025), bsky-users.theo.io (Feb 2026).
//
//   Posts:
//     ~2.3B  all-time (end of 2025)
//     1.41B  in 2025 alone (61% of all-time volume)
//     235M   media posts in 2025
//     ~500K  posts/day typical, 1.48M/day peak (Nov 2024)
//     ~35    avg posts per user
//
//   Likes:
//     ~6.6B  total (Aug 2025)
//
//   Growth:
//     ~17K   new users/day (late 2025, down from 466K/day peak Nov 2024)
//     Milestones: 1M (Sep 2023) → 10M (Sep 2024) → 42.5M (Feb 2026)

struct ContentScenario {
    avg_blob_bytes: usize,
    entries_per_window: u64,
    name: &'static str,
    window_secs: u32,
    windows: u64,
}

struct UserScenario {
    avg_blocks: u64,
    avg_follows: u64,
    max_uid: u64,
    name: &'static str,
    users: u64,
}

fn main() -> Result<()> {
    print_user_scenarios();
    print_content_scenarios()?;
    Ok(())
}

// ── user graph footprint (pure math) ─────────────────────────────────────────

fn print_user_scenarios() {
    let uid = size_of::<Uid>();
    let vec_uid = size_of::<Vec<Uid>>();

    // Bluesky relationship statistics (sources as above):
    //   42.5M registered users (Feb 2026)
    //   2.4B follow records / 41.4M users ~ 58 avg follows (Jaz, Aug 2025)
    //   Blocks: no published aggregate; estimated 3-6/user from Twitter ratios
    //   Follow distribution: power-law (PLOS One), median ~1, mean ~58
    //   ~3.5-4M DAU (late 2025)

    let scenarios = [
        UserScenario {
            name: "Bluesky current (42.5M users, dense IDs)",
            users: 42_500_000,
            max_uid: 42_500_000,
            avg_follows: 58,
            avg_blocks: 5,
        },
        UserScenario {
            name: "Bluesky current, sparse IDs (10% density)",
            users: 42_500_000,
            max_uid: 425_000_000,
            avg_follows: 58,
            avg_blocks: 5,
        },
        UserScenario {
            name: "DAU-only hot set (4M of 42.5M)",
            users: 4_000_000,
            max_uid: 42_500_000,
            avg_follows: 120, // active users skew higher
            avg_blocks: 10,
        },
        UserScenario {
            name: "Timeslice partition (1M authors, full ID space)",
            users: 1_000_000,
            max_uid: 42_500_000,
            avg_follows: 150,
            avg_blocks: 10,
        },
        UserScenario {
            name: "Bluesky 100M users (projected)",
            users: 100_000_000,
            max_uid: 100_000_000,
            avg_follows: 80, // network effect
            avg_blocks: 8,
        },
        UserScenario {
            name: "Twitter-scale (500M users)",
            users: 500_000_000,
            max_uid: 500_000_000,
            avg_follows: 200,
            avg_blocks: 15,
        },
    ];

    println!("relationship graph\n");

    for s in &scenarios {
        let density = 100.0 * s.users as f64 / s.max_uid as f64;
        let empty = (s.max_uid - s.users) as usize;

        println!("── {} ({:.1}% dense) ──", s.name, density);

        let follows = estimate_users(
            vec_uid,
            uid,
            s.max_uid as usize,
            s.users,
            s.avg_follows,
            empty,
        );
        println!("  follows: {follows}");

        let blocks = estimate_users(
            vec_uid,
            uid,
            s.max_uid as usize,
            s.users,
            s.avg_blocks,
            empty,
        );
        println!("  blocks:  {blocks}");

        let mut total = Usage::new("backbone", 2 * s.max_uid as usize * vec_uid);
        total.add_heap_usage((s.users * (s.avg_follows + s.avg_blocks)) as usize * uid);
        total.add_heap_waste(2 * empty * vec_uid);
        println!("  total:   {total}");
        println!();
    }
}

fn estimate_users(
    vec_uid: usize,
    uid: usize,
    max_uid: usize,
    users: u64,
    avg: u64,
    empty: usize,
) -> Usage {
    let mut u = Usage::new("backbone", max_uid * vec_uid);
    u.add_heap_usage((users * avg) as usize * uid);
    u.add_heap_waste(empty * vec_uid);
    u
}

// ── content layer footprint ──────────────────────────────────────────────────
//
// Builds a sample cold arena per scenario to measure per-entry costs via
// ReportUsage, then extrapolates to the full window count.

fn print_content_scenarios() -> Result<()> {
    // ~500K posts/day = ~20,833/hour for Bluesky current
    // ATProto post records are CBOR-encoded, typically 200-500 B
    let scenarios = [
        ContentScenario {
            name: "Bluesky 24h rolling (1h windows)",
            window_secs: 3600,
            windows: 24,
            entries_per_window: 20_833,
            avg_blob_bytes: 256,
        },
        ContentScenario {
            name: "Bluesky 7d rolling (1h windows)",
            window_secs: 3600,
            windows: 168,
            entries_per_window: 20_833,
            avg_blob_bytes: 256,
        },
        ContentScenario {
            name: "Bluesky 30d rolling (6h windows)",
            window_secs: 21_600,
            windows: 120,
            entries_per_window: 125_000,
            avg_blob_bytes: 256,
        },
        ContentScenario {
            name: "Bluesky peak day (1h windows, 1.48M/day)",
            window_secs: 3600,
            windows: 24,
            entries_per_window: 61_667,
            avg_blob_bytes: 256,
        },
        ContentScenario {
            name: "Twitter-scale 24h (1h windows, ~23M posts/day)",
            window_secs: 3600,
            windows: 24,
            entries_per_window: 958_333,
            avg_blob_bytes: 512,
        },
    ];

    println!("cold arenas + timeline\n");

    for s in &scenarios {
        let sample = freeze(0, s.window_secs, s.entries_per_window, s.avg_blob_bytes)?;
        let per_arena = sample.usage();

        let mut total = Usage::new("timeline", 0);
        total.add_heap_usage(per_arena.heap * s.windows as usize);
        total.add_disk_usage(per_arena.disk * s.windows);

        println!("── {} ──", s.name);
        println!(
            "  window:    {}s, {} entries/window",
            s.window_secs, s.entries_per_window,
        );
        println!("  per arena: {per_arena}");
        println!("  {} arenas: {total}", s.windows);

        let timeline = Timeline::new(vec![sample]);
        let mut iter = timeline.iter(0, Order::Asc);
        let mut count = 0;
        while iter.next().is_some() {
            count += 1;
        }
        println!("  verified:  {count} entries iterable");
        println!();
    }

    Ok(())
}

/// Build a cold arena with `n` synthetic entries.
fn freeze(epoch: u64, duration: u32, n: u64, blob_size: usize) -> Result<Arc<Cold>> {
    let mut hot = Hot::new(Window::new(epoch, duration))?;
    let blob = vec![0x42u8; blob_size];
    for i in 0..n {
        let ts = epoch + (i * duration as u64) / n;
        hot.add(
            (i % 10_000) as Uid,
            epoch * 1_000_000 + i, // unique cid
            ts,
            &blob,
        )?;
    }
    Ok(hot.try_into()?)
}

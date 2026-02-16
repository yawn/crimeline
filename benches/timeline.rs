use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use crimeline::arena::{Cold, Hot};
use crimeline::{Order, Timeline, Window};

const ARENA_COUNTS: &[usize] = &[1, 5, 10, 50];
const BLOB_SIZE: usize = 256;
const ENTRIES_PER_ARENA: usize = 1_000;

fn make_blob(i: usize) -> Vec<u8> {
    vec![(i & 0xff) as u8; BLOB_SIZE]
}

fn make_cold(epoch: u64, n: usize) -> Arc<Cold> {
    let duration = (n as u32 + 1) * 10;
    let mut hot = Hot::new(Window::new(epoch, duration)).unwrap();
    for i in 0..n {
        hot.add(
            i as u32,
            epoch + i as u64,
            epoch + (i as u64) * 10,
            &make_blob(i),
        )
        .unwrap();
    }
    hot.try_into().unwrap()
}

fn populated_timeline(n_arenas: usize) -> Timeline {
    let span = (ENTRIES_PER_ARENA as u64 + 1) * 10;
    let arenas: Vec<Arc<Cold>> = (0..n_arenas)
        .map(|i| make_cold(i as u64 * span, ENTRIES_PER_ARENA))
        .collect();
    Timeline::new(arenas)
}

/// Iterate across all arenas (no blob resolve)
fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("timeline/iter");

    for &n in ARENA_COUNTS {
        let tl = populated_timeline(n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut iter = tl.iter(0, Order::Asc);
                let mut count = 0usize;
                while iter.next().is_some() {
                    count += 1;
                }
                black_box(count);
            });
        });
    }

    group.finish();
}

/// Iterate and resolve every blob
fn bench_iter_resolve(c: &mut Criterion) {
    let mut group = c.benchmark_group("timeline/iter_resolve");

    for &n in ARENA_COUNTS {
        let tl = populated_timeline(n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut iter = tl.iter(0, Order::Asc);
                while let Some(entry) = iter.next() {
                    black_box(entry.resolve());
                }
            });
        });
    }

    group.finish();
}

/// Iterate with a start offset that skips ~half of entries
fn bench_iter_skip_half(c: &mut Criterion) {
    let mut group = c.benchmark_group("timeline/iter_skip_half");

    for &n in ARENA_COUNTS {
        let tl = populated_timeline(n);
        let span = (ENTRIES_PER_ARENA as u64 + 1) * 10;
        let total_span = n as u64 * span;
        let midpoint = total_span / 2;

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let mut iter = tl.iter(black_box(midpoint), Order::Asc);
                let mut count = 0usize;
                while iter.next().is_some() {
                    count += 1;
                }
                black_box(count);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_iter,
    bench_iter_resolve,
    bench_iter_skip_half
);

criterion_main!(benches);

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use crimeline::{Sharding, UserMap};

const SIZES: &[u32] = &[10, 100, 1000, 10_000, 100_000];

fn populated_map(n_targets: u32) -> UserMap {
    let map = UserMap::new(Sharding::S128);

    for t in 0..n_targets {
        map.add(0, t * 3);
    }

    map
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched_ref(
                || populated_map(size),
                |map| map.add(0, black_box(size * 3 + 1)),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_add_bulk(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_bulk");

    for &size in SIZES {
        let incoming: Vec<u32> = (0..size).map(|t| t * 3 + 1).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched_ref(
                || populated_map(size),
                |map| map.add_bulk(0, black_box(incoming.clone())),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_contains_hit(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_hit");

    for &size in SIZES {
        let map = populated_map(size);
        let target = (size / 2) * 3;
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(map.contains(0, black_box(target))));
        });
    }

    group.finish();
}

fn bench_contains_miss(c: &mut Criterion) {
    let mut group = c.benchmark_group("contains_miss");

    for &size in SIZES {
        let map = populated_map(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| black_box(map.contains(0, black_box(u32::MAX))));
        });
    }

    group.finish();
}

fn bench_remove(c: &mut Criterion) {
    let mut group = c.benchmark_group("remove");

    for &size in SIZES {
        let target = (size / 2) * 3;
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched_ref(
                || populated_map(size),
                |map| map.remove(0, black_box(target)),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_add,
    bench_add_bulk,
    bench_contains_hit,
    bench_contains_miss,
    bench_remove
);

criterion_main!(benches);

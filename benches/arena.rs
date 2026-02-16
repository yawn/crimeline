use std::hint::black_box;
use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use crimeline::arena::{Cold, Hot};
use crimeline::Window;

const SIZES: &[usize] = &[100, 1_000, 10_000, 100_000];
const BLOB_SIZE: usize = 256;

fn make_blob(i: usize) -> Vec<u8> {
    vec![(i & 0xff) as u8; BLOB_SIZE]
}

fn populated_hot(n: usize) -> Hot {
    let mut hot = Hot::new(Window::new(0, (n as u32 + 1) * 10)).unwrap();
    for i in 0..n {
        hot.add(i as u32, i as u64, (i as u64) * 10, &make_blob(i))
            .unwrap();
    }
    hot
}

fn populated_cold(n: usize) -> Arc<Cold> {
    populated_hot(n).try_into().unwrap()
}

fn exported_bytes(n: usize) -> bytes::Bytes {
    let cold = populated_cold(n);
    let mut buf = Vec::new();
    cold.export(&mut buf).unwrap();
    bytes::Bytes::from(buf)
}

/// Hot → Cold conversion (sort + blob rewrite)
fn bench_hot_to_cold(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena/hot_to_cold");

    for &size in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter_batched(
                || populated_hot(size),
                |hot| -> Arc<Cold> { black_box(hot.try_into().unwrap()) },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Import cold arena from parquet bytes
fn bench_import(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena/import");

    for &size in SIZES {
        let data = exported_bytes(size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                black_box(Cold::import(data.clone()).unwrap());
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_hot_to_cold, bench_import);

criterion_main!(benches);

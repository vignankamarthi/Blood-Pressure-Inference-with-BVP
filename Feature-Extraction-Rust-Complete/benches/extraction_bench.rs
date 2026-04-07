use criterion::{criterion_group, criterion_main, Criterion};
use feature_extraction_rust_complete::catch22::Catch22Features;
use feature_extraction_rust_complete::entropy::EntropyFeatures;
use feature_extraction_rust_complete::stats::StatFeatures;

fn bench_catch22(c: &mut Criterion) {
    let signal: Vec<f64> = (0..1250).map(|i| (i as f64 * 0.05).sin() + (i as f64 * 0.13).cos()).collect();
    c.bench_function("catch22_all", |b| b.iter(|| Catch22Features::compute(&signal)));
}

fn bench_entropy(c: &mut Criterion) {
    let signal: Vec<f64> = (0..1250).map(|i| (i as f64 * 0.05).sin() + (i as f64 * 0.13).cos()).collect();
    c.bench_function("entropy_d7_t2", |b| b.iter(|| EntropyFeatures::calculate(&signal, 7, 2)));
}

fn bench_stats(c: &mut Criterion) {
    let signal: Vec<f64> = (0..1250).map(|i| (i as f64 * 0.05).sin() + (i as f64 * 0.13).cos()).collect();
    c.bench_function("stats_all", |b| b.iter(|| StatFeatures::compute(&signal)));
}

criterion_group!(benches, bench_catch22, bench_entropy, bench_stats);
criterion_main!(benches);

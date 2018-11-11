extern crate rkm;
extern crate ndarray;
extern crate csv;
#[macro_use]
extern crate bencher;

use bencher::Bencher;
use ndarray::Array2;
use std::str::FromStr;

fn read_test_data(data_path: &str, dim: usize) -> Array2<f32> {;
    let mut data_reader = csv::Reader::from_path(data_path).unwrap();
    let mut data: Vec<f32> = Vec::new();
    for record in data_reader.records() {
        for field in record.unwrap().iter() {
            let value = f32::from_str(field);
            data.push(value.unwrap());
        }
    }
    Array2::from_shape_vec((data.len() / dim, dim), data).unwrap()
}

fn bench_iris(b: &mut Bencher) {
    let data = read_test_data("data/iris.data.csv", 2);
    b.iter(|| {
        rkm::kmeans_lloyd(&data.view(), 3)
    });
}

fn bench_s1(b: &mut Bencher) {
    let data = read_test_data("data/s1.data.csv", 2);
    b.iter(|| {
        rkm::kmeans_lloyd(&data.view(), 15)
    });
}

fn bench_birch3(b: &mut Bencher) {
    let data = read_test_data("data/birch3.data.csv", 2);
    b.iter(|| {
        rkm::kmeans_lloyd(&data.view(), 100)
    });
}

fn bench_dim128(b: &mut Bencher) {
    let data = read_test_data("data/dim128.data.csv", 128);
    b.iter(|| {
        rkm::kmeans_lloyd(&data.view(), 16)
    });
}

// TODO: birch3 doesn't converge?
benchmark_group!(benches, bench_iris, bench_s1, bench_birch3, bench_dim128);
benchmark_main!(benches);
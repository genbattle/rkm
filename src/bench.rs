extern crate rkm;
extern crate ndarray;
extern crate csv;
#[macro_use]
extern crate bencher;

use bencher::Bencher;
use ndarray::Array2;

fn read_test_data() -> Array2<f32> {;
    let mut data_reader = csv::Reader::from_file("data/iris.data.csv").unwrap();
    let mut data: Vec<f32> = Vec::new();
    for record in data_reader.decode() {
        let (sl, _, pl, _, _): (f32, f32, f32, f32, String) = record.unwrap();
        data.push(sl);
        data.push(pl);
    }
    Array2::from_shape_vec((data.len() / 2, 2), data).unwrap()
}

fn bench_rkm(b: &mut Bencher) {
    let data = read_test_data();
    b.iter(|| {
        rkm::kmeans_lloyd(&data.view(), 3)
    });
}

benchmark_group!(benches, bench_rkm);
benchmark_main!(benches);
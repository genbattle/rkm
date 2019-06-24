extern crate rkm;
extern crate ndarray;
extern crate csv;

use ndarray::Array2;
use std::str::FromStr;

fn read_test_data() -> Array2<f32> {;
    let mut data_reader = csv::Reader::from_path("data/iris.data.csv").unwrap();
    let mut data: Vec<f32> = Vec::new();
    for record in data_reader.records() {
        for field in record.unwrap().iter() {
            let value = f32::from_str(field);
            data.push(value.unwrap());
        }
    }
    Array2::from_shape_vec((data.len() / 2, 2), data).unwrap()
}


pub fn main() {
    let data = read_test_data();
    let (means, clusters) = rkm::kmeans_lloyd(&data.view(), 3, None);
    println!("data:\n{:?}\nmeans:\n{:?}\nclusters:\n{:?}", data, means, clusters);
}

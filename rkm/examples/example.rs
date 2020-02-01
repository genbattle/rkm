/// example.rs - a basic example that loads the simple iris data set.
///
/// The data set is fed into the k-means algorithm and then the means
/// and clusters are printed. TODO: a better output visualisation.
///
/// This example must be run from the crate root for the relative paths
/// for the example data to be correct; this program will panic if the
/// input file cannot be found.
///
/// You can run this program with the cargo run command:
/// `cargo run --example example`
extern crate csv;
extern crate ndarray;
extern crate rkm;

use ndarray::Array2;
use std::str::FromStr;

fn read_test_data() -> Array2<f32> {
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
    let (means, clusters) = rkm::kmeans_lloyd(&data.view(), 3);
    println!(
        "data:\n{:?}\nmeans:\n{:?}\nclusters:\n{:?}",
        data, means, clusters
    );
}

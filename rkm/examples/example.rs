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
use ndarray::{Array2, ArrayView2};
use std::str::FromStr;
use plotters::prelude::*;

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

fn plot_means_clusters(data: &ArrayView2<f32>, means: &ArrayView2<f32>, clusters: &[usize]) {
    let root_area = BitMapBackend::new("test.png", (800, 600)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut context = ChartBuilder::on(&root_area)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("K Means Demo", ("Arial", 40))
        .build_ranged(1.8f32..5.0f32, 0.0f32..3.0f32)
        .unwrap();
    
    context.configure_mesh().draw().unwrap();
    context.draw_series(data.outer_iter()
        .map(|coords| Circle::new((coords[0], coords[1]), 3, GREEN.filled())))
        .unwrap();
    context.draw_series(means.outer_iter()
        .map(|coords| Circle::new((coords[0], coords[1]), 3, RED.filled())))
        .unwrap();
}

pub fn main() {
    let data = read_test_data();
    let (means, clusters) = rkm::kmeans_lloyd(&data.view(), 3);
    println!(
        "data:\n{:?}\nmeans:\n{:?}\nclusters:\n{:?}",
        data, means, clusters
    );
    plot_means_clusters(&data.view(), &means.view(), &clusters);
}

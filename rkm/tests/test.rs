use ndarray::{arr2, Array2};
use rkm::{Config, kmeans_lloyd, kmeans_lloyd_with_config};
use assert_approx_eq::assert_approx_eq;

fn read_test_data(data_path: &str, dim: usize) -> Array2<f32> {
    use std::str::FromStr;
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

#[test]
#[should_panic(expected = "assertion failed")]
fn test_min_k() {
    let d = arr2(&[[1.0f32, 1.0f32], [2.0f32, 2.0f32], [3.0f32, 3.0f32]]);
    kmeans_lloyd(&d.view(), 1);
}

#[test]
fn test_small_kmeans() {
    let d = arr2(&[
        [1.0f32, 1.0f32],
        [2.0f32, 2.0f32],
        [3.0f32, 3.0f32],
        [1200.0f32, 1200.0f32],
        [956.0f32, 956.0f32],
        [1024.0f32, 1024.0f32],
        [1024.0f32, 17.0f32],
        [1171.0f32, 20.0f32],
    ]);
    let expected_means = arr2(&[
        [2.0f32, 2.0f32],
        [1097.5f32, 18.5f32],
        [1060.0f32, 1060.0f32],
    ]);
    let expected_clusters = vec![0, 0, 0, 2, 2, 2, 1, 1];
    let config = Config::from(Some((0 as u128).to_le_bytes()), None, None);
    let (means, clusters) = kmeans_lloyd_with_config(&d.view(), 3, &config);
    assert_eq!(clusters, expected_clusters);
    means.iter().zip(expected_means.iter())
        .for_each(|m|{
            assert_approx_eq!(m.0, m.1);
        });
}

#[test]
fn test_iris() {
    let data = read_test_data("data/iris.data.csv", 2);
    let expected_means = arr2(&[
        [2.7075472, 1.3094337],
        [3.0416667, 2.0520833],
        [3.439583, 0.24374998],
    ]);
    let config = Config::from(Some((100 as u128).to_le_bytes()), None, None);
    let (means, clusters) = kmeans_lloyd_with_config(&data.view(), 3, &config);
    // not checking actual cluster values because there are too many
    assert_eq!(clusters.len(), data.dim().0);
    means.iter().zip(expected_means.iter())
        .for_each(|m|{
            assert_approx_eq!(m.0, m.1);
        });
}
// TODO: tests for new termination conditions


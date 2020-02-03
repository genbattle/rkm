use assert_approx_eq::assert_approx_eq;
use ndarray::{arr2, Array2};
use rkm::{kmeans_lloyd, kmeans_lloyd_with_config, Config};

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
    means.iter().zip(expected_means.iter()).for_each(|m| {
        assert_approx_eq!(m.0, m.1);
    });
}

/// Test a simple real dataset
#[test]
fn test_iris() {
    let data = read_test_data("data/iris.data.csv", 2);
    let expected_means = arr2(&[
        [2.7075472f32, 1.3094337f32],
        [3.0416667f32, 2.0520833f32],
        [3.439583f32, 0.24374998f32],
    ]);
    let config = Config::from(Some((100 as u128).to_le_bytes()), None, None);
    let (means, clusters) = kmeans_lloyd_with_config(&data.view(), 3, &config);
    // not checking actual cluster values because there are too many
    assert_eq!(clusters.len(), data.dim().0);
    means.iter().zip(expected_means.iter()).for_each(|m| {
        assert_approx_eq!(m.0, m.1);
    });
}

/// Test a simple synthetic data set
#[test]
fn test_s1() {
    let data = read_test_data("data/s1.data.csv", 2);
    let expected_means = arr2(&[
        [654696.1f32, 405158.75f32],
        [565321.1f32, 604057.3f32],
        [672352.44f32, 589449.75f32],
        [802534.1f32, 320892.88f32],
        [600760.9f32, 525206.9f32],
        [326800.03f32, 818472.06f32],
        [604379.7f32, 574263.56f32],
        [669604.9f32, 862576.25f32],
        [823421.2f32, 731145.0f32],
        [859508.75f32, 545905.5f32],
        [597991.06f32, 392953.4f32],
        [414921.8f32, 168421.61f32],
        [852058.44f32, 157685.61f32],
        [367569.66f32, 481344.63f32],
        [153468.75f32, 455027.53f32],
    ]);
    let config = Config::from(Some((7 as u128).to_le_bytes()), None, None);
    let (means, clusters) = kmeans_lloyd_with_config(&data.view(), 15, &config);
    // not checking actual cluster values because there are too many
    assert_eq!(clusters.len(), data.dim().0);
    means.iter().zip(expected_means.iter()).for_each(|m| {
        assert_approx_eq!(m.0, m.1, 1.0f32);
    });
}

/// Test a simple synthetic data set with an early termination condition.
/// Only checks that the means are as would be expected from an early
/// termination because the public API doesn't expose the iteration count
/// (and doing so would have little point other than for testing)
#[test]
fn test_iteration_limit() {
    let data = read_test_data("data/s1.data.csv", 2);
    let expected_means = arr2(&[
        [681714.6f32, 437062.0f32],
        [566276.6f32, 605332.6f32],
        [670336.25f32, 591615.5f32],
        [801869.94f32, 320939.72f32],
        [586849.06f32, 531964.25f32],
        [326800.03f32, 818472.06f32],
        [607580.44f32, 573471.44f32],
        [669604.9f32, 862576.25f32],
        [823421.2f32, 731145.0f32],
        [859883.25f32, 546356.06f32],
        [610730.56f32, 394746.63f32],
        [415181.7f32, 168482.83f32],
        [852058.44f32, 157685.61f32],
        [368004.5f32, 480908.22f32],
        [153468.75f32, 455027.53f32],
    ]);
    let config = Config::from(Some((7 as u128).to_le_bytes()), Some(5), None);
    let (means, clusters) = kmeans_lloyd_with_config(&data.view(), 15, &config);
    // not checking actual cluster values because there are too many
    assert_eq!(clusters.len(), data.dim().0);
    means.iter().zip(expected_means.iter()).for_each(|m| {
        assert_approx_eq!(m.0, m.1, 1.0f32);
    });
}

/// Test a simple synthetic data set with a delta termination condition.
/// Only checks that the means are as would be expected from an early
/// termination because the public API doesn't expose the delta.
#[test]
fn test_delta_limit() {
    let data = read_test_data("data/s1.data.csv", 2);
    let expected_means = arr2(&[
        [659394.4f32, 408869.38f32],
        [566225.2f32, 605209.94f32],
        [672352.44f32, 589449.75f32],
        [802534.1f32, 320892.88f32],
        [597145.94f32, 528207.56f32],
        [326800.03f32, 818472.06f32],
        [605312.5f32, 574865.9f32],
        [669604.9f32, 862576.25f32],
        [823421.2f32, 731145.0f32],
        [859508.75f32, 545905.5f32],
        [600865.44f32, 392542.34f32],
        [414921.8f32, 168421.61f32],
        [852058.44f32, 157685.61f32],
        [367720.16f32, 481216.72f32],
        [153468.75f32, 455027.53f32],
    ]);
    let config = Config::from(Some((7 as u128).to_le_bytes()), None, Some(1500.0f32));
    let (means, clusters) = kmeans_lloyd_with_config(&data.view(), 15, &config);
    // not checking actual cluster values because there are too many
    assert_eq!(clusters.len(), data.dim().0);
    means.iter().zip(expected_means.iter()).for_each(|m| {
        assert_approx_eq!(m.0, m.1, 1.0f32);
    });
}

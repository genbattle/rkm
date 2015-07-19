/*
This create contains a simple implementation of the 
[k-means clustering algorithm](http://en.wikipedia.org/wiki/K-means_clustering).
*/

extern crate rand;

// finds the point in the list of means that is closest to the given point and returns the index
// of that mean.
fn find_closest(point: &[f32; 2], means: &[[f32; 2]]) -> usize {
    // find the mean that is closest
    let mut distances = means.iter().map(|&m|{
        let d0 = m[0] - point[0];
        let d1 = m[1] - point[1];
        d0 * d0 + d1 * d1
    });
    let mut min = distances.next().unwrap();
    let mut index = 0;
    // TODO: should be able to convert this to a fold or something?
    for v in distances.enumerate() {
        if v.1 < min {
            min = v.1;
            index = v.0 + 1;
        }
    }
    return index;
}

/*
The `kmeans` function provides a simple implementation of the standard k-means algorithm, as
described [here](http://en.wikipedia.org/wiki/K-means_clustering#Standard_algorithm).
*/
pub fn kmeans(data: &[[f32; 2]], k: usize) -> Vec<[f32; 2]>  {
    assert!(k > 1); // this algorithm won't work with k < 2 at the moment
    assert!(data.len() > 1); // won't work with at least one data point
    // randomly select initial means from data set
    let mut rng = rand::thread_rng();
    let mut means: Vec<[f32; 2]> = rand::sample(&mut rng, data.iter(), k as usize).iter().map(|v|{*v.clone()}).collect();
    let mut converged = false;
    // loop until convergence is reached
    while !converged {
        let mut means_new: Vec<[f32; 2]> = vec![[0.0, 0.0]; k];
        {
            // Assignment step
            let clusters = data.iter().map(|&d|{
                find_closest(&d, &means)
            }).zip(data);
            // Update step
            let mut means_count: Vec<usize> = vec![0; k as usize];
            for v in clusters {
                means_new[v.0][0] += v.1[0];
                means_new[v.0][1] += v.1[1];
                means_count[v.0] += 1;
            }
            means_new = means_new.iter().zip(means_count.iter()).map(|v| {
                [v.0[0] / *v.1 as f32, v.0[1] / *v.1 as f32]
            }).collect();
        }
        if means_new.iter().zip(means.iter()).all(|v| {
            v.0[0] == v.1[0] && v.0[1] == v.1[1]
        }) {
            converged = true;
        } else {
            means = means_new;
        }
    }
    return means;
}

#[cfg(test)]
mod tests {
    extern crate csv;
    use super::kmeans;
    
    #[cfg(test)]
    fn read_test_data() -> Vec<[f32; 2]> {
        let mut data_reader = csv::Reader::from_file("data/iris.data").unwrap();
        let mut data: Vec<[f32; 2]> = Vec::new();
        for record in data_reader.decode() {
            let (sl, _, pl, _, _): (f32, f32, f32, f32, String) = record.unwrap();
            data.push([sl, pl]);
        }
        println!("Read external data:");
        println!("{:?}", data);
        return data;
    }

    /*
    Test the kmeans method with a basic f32 dataset loaded from a CSV file (in this case iris.data).
    
    NOTE: This test just checks that the algorithm runs successfully; I haven't figured out a way
    to check the validity of the data yet as this dataset has more than one local minima for this
    algorithm. That doesn't mean one result is wrong, it's just the way the algorithm works.
    */
    #[test]
    fn test_kmeans() {
        let data = read_test_data();
        let means = kmeans(&data[..], 3);
        println!("Got means {:?}", means);
    }

    /*
    Test that the algorithm panics when k < 2 is given.
    */
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_min_k() {
        let data = read_test_data();
        let means = kmeans(&data[..], 1);
        println!("Got means {:?}", means);
        // Should panic at this point
        println!("test_min_k failed, should have panicked");
    }
    
    /*
    Test that the algorithm panics when no data is provided.
    TODO: improve this behaviour?
    */
    #[test]
    #[should_panic(expected = "assertion failed")]
    fn test_min_data() {
        let data: Vec<[f32; 2]> = Vec::new();
        let means = kmeans(&data, 3);
        println!("Got means {:?}", means);
        // Should panic at this point
        println!("test_min_data failed, should have panicked");
    }
}

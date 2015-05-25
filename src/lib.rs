/*
This create contains a simple implementation of the 
[k-means clustering algorithm](http://en.wikipedia.org/wiki/K-means_clustering).
*/

extern crate rand;

/*
The `kmeans` function provides a simple implementation of the standard k-means algorithm, as
described [here](http://en.wikipedia.org/wiki/K-means_clustering#Standard_algorithm).
*/
pub fn kmeans(data: &[[f32; 2]], k: u32) -> Vec<[f32; 2]> {
    assert!(k >= 2); // this algorithm won't work with k < 2 at the moment
    // randomly select initial means from data set
    let mut rng = rand::thread_rng();
    let mut means: Vec<[f32; 2]> = rand::sample(&mut rng, data.iter(), k as usize).iter().map(|v|{*v.clone()}).collect();
    let mut converged = false;
    let mut iters = 0;
    // loop until convergence is reached
    while !converged && iters < 40 { // TODO: sort out convergence criteria
        let mut means_new: Vec<[f32; 2]> = vec![[0.0, 0.0]; k as usize];
        {
            // Assignment step
            let clusters = data.iter().map(|&d|{
                // find the mean that is closest
                let mut distances = means.iter().map(|&m|{
                    (m[0] - d[0]).powf(2.0) + (m[1] - d[1]).powf(2.0)
                });
                let mut min = distances.next().unwrap();
                let mut index = 0;
                for v in distances.enumerate() {
                    if v.1 < min {
                        min = v.1;
                        index = v.0;
                    }
                }
                return index;
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
            println!("got means {:?}", means);
        }
        iters += 1;
    }
    return means;
}

extern crate csv;
#[test]
fn test_kmeans() {
    let mut data_reader = csv::Reader::from_file("data/iris.data").unwrap();
    let mut data: Vec<[f32; 2]> = Vec::new();
    for record in data_reader.decode() {
        let (sl, _, pl, _, _): (f32, f32, f32, f32, String) = record.unwrap();
        data.push([sl, pl]);
    }
    println!("Read external data:");
    println!("{:?}", data);
    let means = kmeans(&data[..], 3);
    println!("Got means {:?}", means);
    assert!(false);
}

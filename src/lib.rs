/*
This create contains a simple implementation of the 
[k-means clustering algorithm](http://en.wikipedia.org/wiki/K-means_clustering).
*/

extern crate rand;

use std::ops::{Add, Sub, Mul, Div, Index};
use std::cmp::{PartialOrd, PartialEq};
use std::fmt::Debug;

/*
Numeric value trait, defines the types that can be used for the value of each dimension in a
data point.
*/
trait Num: Add + Sub + Mul + Div + PartialOrd + PartialEq + Copy + Debug {}
impl<T> Num for T where T: Add + Sub + Mul + Div + PartialOrd + PartialEq + Copy + Debug {}

// kmeans data, for accessing 
trait Data<T> where T: Index<usize> {
    type V: Num = T;
    
    fn len() -> usize;
}

/*
Find the point in the list of means that is closest to the given point and return the index
of that mean in the means slice.
*/
fn find_closest<T, U>(point: &T, means: &[T]) -> usize
where
    T: Index<usize, Output=U>,
    U: Add<Output=U> + Clone + Mul<Output=U> + PartialOrd + Sub<Output=U>
{
    // find the mean that is closest
    let mut distances = means.iter().map(|&m|{ // TODO: make this part of the data trait? At the very least T will need to be Iterator
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
pub fn kmeans<T, U>(data: &[T], k: usize) -> Vec<T>
where
    T: Index<usize, Output=U>,
    U: Add<Output=U> + Sub<Output=U> + Mul<Output=U> + Div<Output=U> + PartialOrd + PartialEq + Copy + Debug {
    assert!(k > 1); // this algorithm won't work with k < 2 at the moment
    assert!(data.len() > 1); // won't work with at least one data point
    // randomly select initial means from data set
    let mut rng = rand::thread_rng();
    let mut means: Vec<T> = rand::sample(&mut rng, data.iter(), k).iter().map(|v|{*v.clone()}).collect();
    let mut converged = false;
    // loop until convergence is reached
    while !converged {
        let mut means_new: Vec<T>;
        {
            // TODO: put some of this code in the data trait?
            // Assignment step
            let clusters = data.iter().map(|&d|{
                find_closest(&d, &means)
            }).zip(data);
            // Update step
            let mut means_count: Vec<usize> = vec![0; k];
            for _ in [0..]
            for v in clusters {
                means_new[v.0][0] += v.1[0];
                means_new[v.0][1] += v.1[1];
                means_count[v.0] += 1;
            }
            means_new = means_new.iter().zip(means_count.iter()).map(|v| {
                [v.0[0] / *v.1, v.0[1] / *v.1]
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

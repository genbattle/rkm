/*
This create contains a simple implementation of the 
[k-means clustering algorithm](http://en.wikipedia.org/wiki/K-means_clustering).
*/

extern crate rand;
extern crate num;

use std::ops::{Add, Sub, Mul, Div, Index};
use std::cmp::{PartialOrd, PartialEq};
use std::fmt::Debug;

/*
Numeric value trait, defines the types that can be used for the value of each dimension in a
data point.
*/
trait Value: num::Signed + From<usize> + PartialOrd + Copy + Debug {}
impl<T> Value for T where T: num::Signed + From<usize> + PartialOrd + Copy + Debug {}

/*
// kmeans data, for accessing 
trait Data<T> where T: Num {
    fn len(&self) -> usize;
    fn get(&self, index: usize) -> T;
    fn set(&mut self, index: usize, value: T);
    fn calc_mean(total: T, count: usize) -> T;
}*/

/*
Data implementation for [f32; 2]

The reason get and set are used instead of the `Index`/`IndexMut` traits is because built-in traits 
cannot be implemented by a third party such as myself for built-in types; you must be the author
of either the type or trait in a type->trait relationship.

`calc_mean()`` is here because of the scalar cast from f32 to usize, which can't be done with a 
generic type T.
*/
/*impl<T> Data<T> for [T; 2] where T: Num {
    fn len(&self) -> usize {
        2
    }
    
    fn get(&self, index: usize) -> T {
        self[index]
    }
    
    fn set(&mut self, index: usize, value: T) {
        self[index] = value;
    }
    
    fn calc_mean(total: T, count: usize) -> T {
        total / T::from(count)
    }
}*/

/*
Find the point in the list of means that is closest to the given point and return the index
of that mean in the means slice.
*/
fn find_closest<T, U>(point: &T, means: &[T]) -> usize
where
    T: IntoIterator<Item=U>,
    U: Value
{
    // find the mean that is closest
    let mut distances = means.iter().map(|&m|{
        point.into_iter().zip(m.into_iter()).fold(num::Zero::zero(), |total, v| {
            let delta: U = v.0 - v.1;
            total + (delta * delta)
        })
    });
    let mut min: U = distances.next().unwrap();
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
    T: IntoIterator<Item=U>,
    U: Value
{
    assert!(k > 1); // this algorithm won't work with k < 2 at the moment
    assert!(data.len() > 1); // won't work without at least one data point
    // randomly select initial means from data set
    let mut rng = rand::thread_rng();
    let mut means: Vec<T> = rand::sample(&mut rng, data.iter(), k).iter().map(|v|{*v.clone()}).collect();
    let mut converged = false;
    // loop until convergence is reached
    while !converged {
        let mut means_new: Vec<T>;
        {
            // Assignment step
            let clusters = data.iter().map(|&d|{
                find_closest(&d, &means)
            }).zip(data);
            // Update step
            let mut means_count: Vec<usize> = vec![0; k];
            // TODO: get the totals
            means_new = (0..k).map(|i| {
                T means
            }).collect();
            // get the counts
            means_count = (0..k).map(|i| {
                clusters.filter(|v| v.0 == i).count()
                //let mut current = clusters.filter(|v| v.0 == i);
                //clusters.filter(|v| v.0 == i).fold(0, |sum, v| sum + v.1) / T::from(clusters.filter(|v| v.0 == i).count())
                //clusters.filter(|v| v.0 == i)
                //    .fold(0, |sum, v| sum.into_iter().zip(v.1.into_iter()).map(|j| j.0 + j.1).collect())
                //    / U::from(clusters.filter(|v| v.0 == i).count())
            }).collect();
            /*for v in clusters {
                // TODO: calculate total for each dimension of each mean
                means_new[v.0].set(0, means_new[v.0].get(0) + v.1.get(0));
                means_new[v.0].set(1, means_new[v.0].get(1) + v.1.get(1));
                means_count[v.0] += 1;
            }
            means_new = means_new.iter().zip(means_count.iter()).map(|v| {
                // TODO: average for each mean
                v.0.iter().map(|x| x / T::from(v.1))
                //[Data::calc_mean(v.0.get(0), *v.1), Data::calc_mean(v.0.get(1), *v.1)]
            }).collect();*/
        }
        if means_new.iter().zip(means.iter()).all(|v| {
            v.0.get(0) == v.1.get(0) && v.0.get(1) == v.1.get(1)
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

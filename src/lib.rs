/*
This create contains a simple implementation of the 
[k-means clustering algorithm](http://en.wikipedia.org/wiki/K-means_clustering).
*/

/*
The `kmeans` function provides a simple implementation of the standard k-means algorithm, as
described [here](http://en.wikipedia.org/wiki/K-means_clustering#Standard_algorithm).
*/
pub fn kmeans(data: &[[f32; 2]], k: u32) -> Vec<[f32; 2]> {
    return Vec::new();
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

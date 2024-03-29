use crate::*;
use ndarray::ArcArray1;
use ndarray::ArcArray2;
use ndarray::{rcarr1, rcarr2};
pub type S = f64;
pub type V = ArcArray1<S>;
pub type M = ArcArray2<S>;
use rand::thread_rng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::io::Read;
use std::iter;
/// Utility Functions for the Conjugate Gradient Method

/// A linear system, ax-b=0, to be solved iteratively, with an optional initial solution.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct LinearSystem {
    pub a: M,
    pub b: V,
    pub x0: Option<V>,
}

pub fn make_3x3_pd_system_1() -> LinearSystem {
    make_3x3_psd_system(
        rcarr2(&[[1., 2., -1.], [0., 1., 0.], [0., 0., 1.]]),
        rcarr1(&[0., 1., 0.]),
    )
}

pub fn make_3x3_pd_system_2() -> LinearSystem {
    make_3x3_psd_system(
        rcarr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]),
        rcarr1(&[0., 1., 0.]),
    )
}

pub fn make_3x3_psd_system(m: M, b: V) -> LinearSystem {
    let a = (m.t().dot(&m)).into_shared();
    LinearSystem { a, b, x0: None }
}

/// Utility Functions for Reservoir Sampling

/// Produce a stream like [a, ..., a, b, ..., b] with `num_of_initial_values` copies of "a"
/// (aka `initial_value`s) and `stream_length` total values.
/// Utility function used in examples showing convergence of reservoir mean to stream mean.
pub fn generate_step_stream(
    stream_length: usize,
    num_of_initial_values: usize,
    initial_value: i64,
    final_value: i64,
) -> impl StreamingIterator<Item = i64> {
    // Create capacity of items with initial weight.
    let initial_iter = iter::repeat(initial_value).take(num_of_initial_values);
    if num_of_initial_values > stream_length {
        panic!("Number of initiral values must be less than or equal to stream length.");
    }
    let final_iter = iter::repeat(final_value).take(stream_length - num_of_initial_values);
    let stream = initial_iter.chain(final_iter);
    convert(stream)
}

/// A stream of 100 weighted datum is generated using generate_step_stream. The first
/// 50 items have value 0 and the remaining 50 have value 1. All weights are set to 1.
/// The mean of a reservoir sample with capacity equal to half the stream length is computed
/// and stored. This is repeated 50 times and the mean of the means is returned.
///
/// This is used in wrs_mean_test and wrs_mean_test_looped. The latter is used to
/// determine the error rate of the former.
pub fn mean_of_means_of_step_stream() -> f64 {
    let stream_length = 100usize;
    let num_of_initial_values = stream_length / 2;
    let initial_value = 0i64;
    let final_value = 1i64;
    let capacity = num_of_initial_values;
    let num_runs = 50usize;
    let mut means: Vec<f64> = Vec::with_capacity(capacity);

    for _i in 0..num_runs {
        let stream = generate_step_stream(
            stream_length,
            num_of_initial_values,
            initial_value,
            final_value,
        );
        let stream = wd_iterable(stream, |_x| 1f64);
        let stream = weighted_reservoir_sample(stream, capacity, None);
        let res = last(stream).unwrap();
        let res: Vec<i64> = res.iter().map(|x| x.value).collect();
        let mean = res.iter().sum::<i64>() as f64 / capacity as f64;
        means.push(mean);
    }
    let mean_mean = means.iter().sum::<f64>() / num_runs as f64;
    mean_mean
}

/// Produce an enumerated stream like [(0,a), (1,a),..., (capacity, b), ..., (stream_length-1, b)] with `capacity` copies of "a"
/// (aka `initial_value`s) and `stream_length` total values.
/// Utility function used in examples showing convergence of reservoir mean to stream mean.
pub fn generate_enumerated_step_stream(
    stream_length: usize,
    capacity: usize,
    initial_value: i64,
    final_value: i64,
) -> impl StreamingIterator<Item = Numbered<i64>> {
    // Create capacity of items with initial weight.
    let initial_iter = iter::repeat(initial_value).take(capacity);
    if capacity > stream_length {
        panic!("Capacity must be less than or equal to stream length.");
    }
    let final_iter = iter::repeat(final_value).take(stream_length - capacity);
    let stream = initial_iter.chain(final_iter);
    let stream = convert(stream);
    enumerate(stream)
}

// Produce a stream from a normal distribution.
// Utility function used in examples of reservoir sampling
// histogram animation.
#[allow(clippy::needless_collect)]
pub fn generate_stream_from_normal_distribution(
    stream_length: usize,
    mean: f64,
    sigma: f64,
) -> impl Iterator<Item = f64> {
    let normal = Normal::new(mean, sigma).unwrap();
    let stream: Vec<f64> = normal
        .sample_iter(&mut thread_rng())
        .take(stream_length)
        .collect();
    stream.into_iter()
}

/// Utility Functions for Weighted Reservoir Sampling

/// utility function for testing ReservoirSample
pub fn generate_stream_with_constant_probability(
    stream_length: usize,
    capacity: usize,
    probability: f64,
    initial_weight: f64,
    initial_value: usize,
    final_value: usize,
) -> impl Iterator<Item = WeightedDatum<usize>> {
    // Create capacity of items with initial weight.
    let initial_iter = iter::repeat(new_datum(initial_value, initial_weight)).take(capacity);
    if capacity > stream_length {
        panic!("Capacity must be less than or equal to stream length.");
    }
    let final_iter =
        iter::repeat(new_datum(final_value, initial_weight)).take(stream_length - capacity);
    let mut power = 0i32;
    let mapped = final_iter.map(move |wd| {
        power += 1;
        new_datum(
            wd.value,
            initial_weight
                * (probability / capacity as f64)
                * (capacity as f64 / (capacity as f64 - probability)).powi(power),
        )
    });
    initial_iter.chain(mapped)
}

pub fn expose_w(count: &f64) -> f64 {
    count * count
}

/// Utility functions for writing data to yaml, including for visualizations
///
/// The order of the parameters is not controlled.
pub fn write_parameters_to_yaml<T>(
    params: HashMap<String, T>,
    file_path: &str,
) -> std::io::Result<()>
where
    T: std::string::ToString,
{
    let mut file = File::create(file_path)?;
    for (key, value) in params.iter() {
        let line: String = [&key.to_string(), ":", " ", &value.to_string(), "\n"].join("");
        file.write_all(line.as_bytes())?;
    }
    Ok(())
}

pub fn read_yaml_to_string(file_path: &str) -> Result<std::string::String, std::io::Error> {
    let mut read_file =
        File::open(file_path).expect("Could not open file with test data to asserteq.");
    let mut contents = String::new();
    read_file
        .read_to_string(&mut contents)
        .expect("Could not read data from file.");
    // println!("Contents: \n \n {:#?}", contents);
    std::fs::remove_file(file_path).expect("Could not remove data file for test.");
    Ok(contents)
}

/// A simple Counter iterator to use in demos and tests.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Default)]
pub struct Counter {
    count: f64,
}

impl Counter {
    pub fn new() -> Self {
        Counter { count: 0. }
    }
}

impl StreamingIterator for Counter {
    type Item = f64;
    fn advance(&mut self) {
        self.count += 1.;
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(&self.count)
    }
}

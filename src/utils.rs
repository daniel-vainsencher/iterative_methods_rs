use crate::algorithms::cg_method::*;
use crate::*;
use ndarray::{rcarr1, rcarr2};
use std::collections::HashMap;
use std::io::Read;
use std::iter;

/// Utility Functions for the Conjugate Gradient Method

/// A linear system, ax-b=0, to be solved iteratively, with an optional initial solution.
#[derive(Clone, Debug)]
pub struct LinearSystem {
    pub a: M,
    pub b: V,
    pub x0: Option<V>,
}

pub fn make_3x3_psd_system_1() -> LinearSystem {
    make_3x3_psd_system(
        rcarr2(&[[1., 2., -1.], [0., 1., 0.], [0., 0., 1.]]),
        rcarr1(&[0., 1., 0.]),
    )
}

pub fn make_3x3_psd_system_2() -> LinearSystem {
    make_3x3_psd_system(
        rcarr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]),
        rcarr1(&[0., 1., 0.]),
    )
}

pub fn make_3x3_psd_system(m: M, b: V) -> LinearSystem {
    let a = (m.t().dot(&m)).into_shared();
    LinearSystem {
        a: a,
        b: b,
        x0: None,
    }
}

/// Utility Functions for Weighted Reservoir Sampling

/// utility function for testing ReservoirIterable
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
            initial_weight * probability / (1.0 - probability).powi(power),
        )
    });
    initial_iter.chain(mapped)
}

/// Produce a stream like [a, ..., a, b, ..., b] with `capacity` copies of "a"
/// (aka `initial_value`s) and `stream_length` total values.
/// Utility function used in examples showing convergence of reservoir mean to stream mean.
pub fn generate_step_stream(
    stream_length: usize,
    capacity: usize,
    initial_value: i64,
    final_value: i64,
) -> impl Iterator<Item = i64> {
    // Create capacity of items with initial weight.
    let initial_iter = iter::repeat(initial_value).take(capacity);
    if capacity > stream_length {
        panic!("Capacity must be less than or equal to stream length.");
    }
    let final_iter = iter::repeat(final_value).take(stream_length - capacity);
    let stream = initial_iter.chain(final_iter);
    stream
}

pub fn expose_w(count: &f64) -> f64 {
    count * count
}

/// Utility functions for visualizations
///
/// The order of the parameters is not controlled.
pub fn write_parameters_to_yaml<T>(params: HashMap<&str, T>, file_path: &str) -> std::io::Result<()>
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

// pub fn write_vec_to_yaml<T>(avec: &Vec<T>, file_name: &str) -> std::io::Result<()>
pub fn write_vec_to_yaml<T>(avec: &[T], file_name: &str) -> std::io::Result<()>
where
    T: std::fmt::Display,
{
    let mut file = File::create(file_name)?;
    file.set_len(0)?;
    for val in avec {
        let mut val = val.to_string();
        val = ["-", &val, "\n"].join(" ");
        file.write_all(val.as_bytes())?;
    }
    file.flush()?;
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

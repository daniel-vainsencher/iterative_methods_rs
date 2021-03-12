use crate::algorithms::cg_method::*;
use crate::*;
use ndarray::{rcarr1, rcarr2};
use rand::distributions::Uniform;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
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

/// Utility Functions for (Weighted) Reservoir Sampling

/// Generates a random sample from the uniform distribution on (0,1) returned as a Vec.
/// Utility function to test the moments of a WRS.
pub fn uniform_stream_as_vec(num: usize) -> Vec<f64> {
    let range = Uniform::from(0.0..1.0);
    let stream_vec: Vec<f64> = rand::thread_rng().sample_iter(&range).take(num).collect();
    stream_vec
}

pub fn write_population_to_file<T>(a_stream_vec: &Vec<T>, file_name: &str) -> std::io::Result<()>
where
    T: std::fmt::Display,
{
    let mut file = File::create(file_name)?;
    for val in a_stream_vec {
        let mut val = val.to_string();
        val = ["-", &val, "\n"].join(" ");
        file.write_all(val.as_bytes())?;
    }
    file.flush()?;
    Ok(())
}

pub fn clear_file(file_name: &str) -> std::io::Result<()> {
    let file = File::create(file_name)?;
    file.set_len(0)?;
    Ok(())
}

pub fn write_res_to_file<T>(res_vec_values: &Vec<T>, file_name: &str) -> std::io::Result<()>
where
    T: std::fmt::Display,
{
    let mut file = OpenOptions::new().append(true).create(true).open(file_name);
    if let Ok(ref mut f) = file {
        for val in res_vec_values {
            let mut val = val.to_string();
            val = ["-", &val, "\n"].join(" ");
            f.write_all(val.as_bytes())?;
        }
        f.write_all(b"--- # new reservoir \n")?;
        f.flush()?;
    };
    Ok(())
}

/// Utility function for testing WeightedReservoirIterable
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
    let stream = initial_iter.chain(mapped);
    stream
}

/// Utility function used in examples showing convergence of reservoir mean to stream mean.
pub fn generate_step_stream(
    stream_length: usize,
    capacity: usize,
    weight: f64,
    initial_value: usize,
    final_value: usize,
) -> impl Iterator<Item = WeightedDatum<usize>> {
    // Create capacity of items with initial weight and value.
    let initial_iter = iter::repeat(new_datum(initial_value, weight)).take(capacity);
    if capacity > stream_length {
        panic!("Capacity must be less than or equal to stream length.");
    }
    let final_iter = iter::repeat(new_datum(final_value, weight)).take(stream_length - capacity);
    let stream = initial_iter.chain(final_iter);
    stream
}

pub fn expose_w(count: &f64) -> f64 {
    count * count
}

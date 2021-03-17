use crate::algorithms::cg_method::*;
use crate::*;
use ndarray::{rcarr1, rcarr2};
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
    let stream = initial_iter.chain(mapped);
    stream
}

pub fn expose_w(count: &f64) -> f64 {
    count * count
}

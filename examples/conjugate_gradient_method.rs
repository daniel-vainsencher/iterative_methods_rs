extern crate eigenvalues;
extern crate nalgebra as na;
use streaming_iterator::*;

use iterative_methods::conjugate_gradient::{conjugate_gradient, ConjugateGradient};
use iterative_methods::utils::make_3x3_pd_system_2;
use iterative_methods::*;

use ndarray::{rcarr1, ArcArray1};
pub type V = ArcArray1<f64>;

/// The usual euclidean length of the residual
fn residual_l2(result: &ConjugateGradient) -> f64 {
    let res = result.a.dot(&result.solution) - &result.b;
    res.dot(&res).sqrt()
}

/// The euclidean distance induced by A, between current solution and
/// a user provided point, which is interesting when it is the true
/// optimum.
fn a_distance(result: &ConjugateGradient, optimum: V) -> f64 {
    let error = &result.solution - &optimum;
    error.dot(&result.a.dot(&error)).sqrt()
}

/// The l-infinity norm of the residual
fn residual_linf(result: &ConjugateGradient) -> f64 {
    (result.a.dot(&result.solution) - &result.b).fold(0.0, |m, e| m.max(e.abs()))
}

/// Example demonstrating different uses of adaptors.
fn cg_demo() {
    // Set up a problem for which we happen to know the solution
    let p = make_3x3_pd_system_2();
    let optimum = rcarr1(&[-4.0, 6., -4.]);

    // Initialize the conjugate gradient solver on this problem
    let cg_iter = conjugate_gradient(&p);

    // Cap the number of iterations.
    let cg_iter = cg_iter.take(80);

    // Time each iteration, only of preceding steps (the method)
    // excluding downstream evaluation and I/O (tracking overhead), as
    // well as elapsed clocktime (combining both).
    let cg_iter = time(cg_iter);

    // Record multiple measures of quality
    let cg_iter = assess(cg_iter, |TimedResult { result, .. }| {
        (
            residual_l2(result),
            residual_linf(result),
            a_distance(result, optimum.clone()),
        )
    });

    // Stop if converged by both residual criteria. We could use the
    // distance from optimum, but that would be cheating. Luckily, we
    // know that:
    //
    // 1. The method converges, so distance to optimum tends to zero.
    // 2. The solution tends to the optimum together with the residual
    // tending to zero.
    //
    // Hence conditions on the residual also work.
    fn small_residual((euc, linf, _): &(f64, f64, f64)) -> bool {
        euc < &1e-3 && linf < &1e-3
    }

    // Take_until implements a stopping condition: will return all
    // initial elements until the first that satisfies a predictate,
    // inclusive.
    let mut cg_iter = take_until(cg_iter, |ar| small_residual(&ar.annotation));

    // Actually run, and output progress
    while let Some(AnnotatedResult {
        annotation: (euc, linf, a_dist),
        result:
            TimedResult {
                result,
                start_time,
                duration,
            },
    }) = cg_iter.next()
    {
        println!(
            "{:8} : {:6} | ||Ax - b||_2 = {:.3}, ||Ax - b||_inf = {:.3}, ||x-x*||_A = {:.3}, for x = {:.4}",
            start_time.as_nanos(), duration.as_nanos(),
            euc, linf, a_dist, result.solution
        );
    }
}

fn main() {
    cg_demo();
}

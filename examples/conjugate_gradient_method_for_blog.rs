extern crate eigenvalues;
extern crate nalgebra as na;
use streaming_iterator::*;

use iterative_methods::algorithms::cg_method::CGIterable;
use iterative_methods::utils::make_3x3_pd_system_2;
use iterative_methods::*;

use ndarray::{rcarr1, ArcArray1};
pub type V = ArcArray1<f64>;

/// Negative example: no reuse, ignoring adaptors, corresponding to
/// https://daniel-vainsencher.github.io/book/iterative_methods_part_1.html
/// see cg_demo_pt2_2 below for a better example.
fn cg_demo_pt1() {
    // First we generate a problem, which consists of the pair (A,b).
    let p = make_3x3_pd_system_2();

    // Next convert it into an iterator
    let mut cg_iter = CGIterable::conjugate_gradient(p);

    // and loop over intermediate solutions.
    // Note `next` is provided by the StreamingIterator trait using
    // `advance` then `get`.
    while let Some(result) = cg_iter.next() {
        // We want to find x such that a.dot(x) = b
        // then the difference between the two sides (called the residual),
        // is a good measure of the error in a solution.
        let res = result.a.dot(&result.x) - &result.b;

        // The (squared) length of the residual is a cost, a number
        // summarizing how bad a solution is. When working on iterative
        // methods, we want to see these numbers decrease quickly.
        let res_squared_length = res.dot(&res);

        // || ... ||_2 is notation for euclidean length of what
        // lies between the vertical lines.
        println!(
            "||Ax - b||_2 = {:.5}, for x = {:.4}, residual = {:.7}",
            res_squared_length.sqrt(),
            result.x,
            res
        );
        if res_squared_length < 1e-3 {
            break;
        }
    }
}

/// The usual euclidean length of the residual
fn residual_l2(result: &CGIterable) -> f64 {
    let res = result.a.dot(&result.x) - &result.b;
    res.dot(&res).sqrt()
}

/// Early example using adaptors from
/// https://daniel-vainsencher.github.io/book/iterative_methods_part_2.html
fn cg_demo_pt2_1() {
    let p = make_3x3_pd_system_2();
    let cg_iter = CGIterable::conjugate_gradient(p);

    // Annotate each approximate solution with its cost
    let cg_iter = assess(cg_iter, residual_l2);
    // and use it in stopping condition
    let mut cg_iter = cg_iter.take_while(|ar| ar.annotation > 1e-3);
    // Deconstruct to break out the result and cost
    while let Some(AnnotatedResult {
        result: cgi,
        annotation: euc,
    }) = cg_iter.next()
    {
        // Now the loop body is I/O only as it should be!
        println!("||Ax - b||_2 = {:.5}, for x = {:.4}", euc, cgi.x);
    }
}

/// The euclidean distance induced by A, between current solution and
/// a user provided point, which is interesting when it is the true
/// optimum.
fn a_distance(result: &CGIterable, optimum: V) -> f64 {
    let error = &result.x - &optimum;
    error.dot(&result.a.dot(&error)).sqrt()
}

/// The l-infinity norm of the residual
fn residual_linf(result: &CGIterable) -> f64 {
    (result.a.dot(&result.x) - &result.b).fold(0.0, |m, e| m.max(e.abs()))
}

/// Example using adaptors from late in
/// https://daniel-vainsencher.github.io/book/iterative_methods_part_2.html
fn cg_demo_pt2_2() {
    // Set up a problem for which we happen to know the solution
    let p = make_3x3_pd_system_2();
    let optimum = rcarr1(&[-4.0, 6., -4.]);

    let cg_iter = CGIterable::conjugate_gradient(p);
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
    // Stop if converged by both criteria
    fn small_residual((euc, linf, _): &(f64, f64, f64)) -> bool {
        euc < &1e-3 && linf < &1e-3
    }
    let mut cg_iter = cg_iter.take_while(|ar| !small_residual(&ar.annotation));
    // Output progress
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
            "{:8} : {:8} | ||Ax - b||_2 = {:.5}, ||Ax - b||_inf = {:.5}, ||x-x*||_A = {:.5}, for x = {:.4}, residual = {:.7}",
            start_time.as_nanos(), duration.as_nanos(),
            euc, linf, a_dist, result.x, result.r
        );
    }
}

fn main() {
    println!("Part 1 demo: method in iterator, logic in loop.");
    cg_demo_pt1();
    println!("Part 2 demo 1: method in iterator, assessment in adaptor, rest in loop.");
    cg_demo_pt2_1();
    println!("Part 2 demo 2: logic via adaptors, I/O in loop.");
    cg_demo_pt2_2();
}

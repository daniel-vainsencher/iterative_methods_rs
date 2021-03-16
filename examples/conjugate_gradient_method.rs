extern crate eigenvalues;
extern crate nalgebra as na;
use streaming_iterator::*;

use iterative_methods::algorithms::cg_method::CGIterable;
use iterative_methods::utils::make_3x3_psd_system_2;
use iterative_methods::*;

/// Demonstrate usage and convergence of conjugate gradient as a streaming-iterator.
fn cg_demo() {
    let p = make_3x3_psd_system_2();
    println!("a: \n{}", &p.a);
    let cg_iter = CGIterable::conjugate_gradient(p);
    // Upper bound the number of iterations
    let cg_iter = cg_iter.take(20);
    // Apply a quality based stopping condition; this relies on
    // algorithm internals, requiring all state to be exposed and
    // not just the result.
    let cg_iter = cg_iter.take_while(|cgi| cgi.rsprev.sqrt() > 1e-6);

    // Note the side effect of tee is applied exactly to every x
    // produced above, the sequence of which is not affected at
    // all. This is just like applying a side effect inside the while
    // loop, except we can compose multiple tee, each with its own
    // effect.
    let cg_iter = step_by(cg_iter, 2);
    let cg_iter = time(cg_iter);

    // We are assessing after timing, which means that computing this
    // function is excluded from the duration measurements, which is
    // generally the right way to do it, though not important here.
    fn score(TimedResult { result, .. }: &TimedResult<CGIterable>) -> f64 {
        result.rs
    }

    let cg_iter = assess(cg_iter, score);
    let mut cg_print_iter = tee(
        cg_iter,
        |AnnotatedResult {
             result:
                 TimedResult {
                     result,
                     start_time,
                     duration,
                 },
             annotation: cost,
         }| {
            let res = result.a.dot(&result.x) - &result.b;
            println!(
            "||Ax - b ||_2^2 = {:.5}, for x = {:.4}, and Ax - b = {:.5}; iteration start {}μs, duration {}μs",
            cost,
            result.x,
            res,
            start_time.as_nanos(),
            duration.as_nanos(),
        );
        },
    );
    while let Some(_cgi) = cg_print_iter.next() {}
}

fn main() {
    cg_demo();
}

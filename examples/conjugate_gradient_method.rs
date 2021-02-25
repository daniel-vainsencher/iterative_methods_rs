extern crate eigenvalues;
extern crate nalgebra as na;
use ndarray::*;
use streaming_iterator::*;

use iterative_methods::algorithms::cg_method::*;
use iterative_methods::*;

pub fn make_3x3_psd_system_1() -> LinearSystem {
    make_3x3_psd_system(
        rcarr2(&[[1., 2., -1.], [0., 1., 0.], [0., 0., 1.]]),
        rcarr1(&[0., 1., 0.]),
    )
}

fn make_3x3_psd_system_2() -> LinearSystem {
    make_3x3_psd_system(
        rcarr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]),
        rcarr1(&[0., 1., 0.]),
    )
}

fn make_3x3_psd_system(m: M, b: V) -> LinearSystem {
    let a = (m.t().dot(&m)).into_shared();
    LinearSystem {
        a: a,
        b: b,
        x0: None,
    }
}

/// Demonstrate usage and convergence of conjugate gradient as a streaming-iterator.
fn cg_demo() {
    let p = make_3x3_psd_system_2();
    println!("a: \n{}", &p.a);
    let cg_iter = CGIterable::conjugate_gradient(p)
        // Upper bound the number of iterations
        .take(20)
        // Apply a quality based stopping condition; this relies on
        // algorithm internals, requiring all state to be exposed and
        // not just the result.
        .take_while(|cgi| cgi.rsprev.sqrt() > 1e-6);
    // Because time, tee are not part of the StreamingIterator trait,
    // they cannot be chained syntactically as in the above.

    // TODO can this be fixed? see iterutils crate.

    //Note the side effect of tee is applied exactly to every x
    // produced above, the sequence of which is not affected at
    // all. This is just like applying a side effect inside the while
    // loop, except we can compose multiple tee, each with its own
    // effect.
    let step_by_cg_iter = step_by(cg_iter, 2);
    let timed_cg_iter = time(step_by_cg_iter);
    // We are assessing after timing, which means that computing this
    // function is excluded from the duration measurements, which can
    // be important in other cases.
    let ct_cg_iter = assess(timed_cg_iter, |TimedResult { result, .. }| {
        let res = result.a.dot(&result.x) - &result.b;
        res.dot(&res)
    });
    let mut cg_print_iter = tee(
        ct_cg_iter,
        |CostResult {
             result:
                 TimedResult {
                     result,
                     start_time,
                     duration,
                 },
             cost,
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

extern crate eigenvalues;
extern crate nalgebra as na;
use streaming_iterator::*;

use iterative_methods::algorithms::cg_method::CGIterable;
use iterative_methods::utils::{make_3x3_pd_system_1, make_3x3_pd_system_2};
use iterative_methods::*;

/// Demonstrate usage and convergence of conjugate gradient as a streaming-iterator.
fn cg_demo() {
    let p = make_3x3_pd_system_2();
    println!("a: \n{}", &p.a);
    let cg_iter = CGIterable::conjugate_gradient(p);
    // Upper bound the number of iterations
    let cg_iter = cg_iter.take(20);
    // Apply a quality based stopping condition; this relies on
    // algorithm internals, requiring all state to be exposed and
    // not just the result.
    let cg_iter = cg_iter.take_while(|cgi| cgi.rsprev.sqrt() > 1e-6);

    // Note the side effect of inspect is applied exactly to every x
    // produced above, the sequence of which is not affected at
    // all. This is just like applying a side effect inside the while
    // loop, except we can compose multiple inspect, each with its own
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
    let mut cg_print_iter = inspect(
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
        // methods, we want to see these number decrease quickly.
        let res_squared_length = res.dot(&res);

        // || ... ||_2 is notation for euclidean length of what
        // lies between the vertical lines.
        println!(
            "||Ax - b||_2 = {:.5}, for x = {:.4}, residual = {:.7}",
            res_squared_length.sqrt(),
            result.x,
            res
        );
        if res_squared_length < 1e-10 {
            break;
        }
    }
}

fn residual_l2(result: &CGIterable) -> f64 {
    let res = result.a.dot(&result.x) - &result.b;
    res.dot(&res).sqrt()
}

fn cg_demo_pt2_1() {
    let p = make_3x3_pd_system_2();
    let cg_iter = CGIterable::conjugate_gradient(p);

    // Annotate each approximate solution with its cost
    let mut cg_iter = assess(cg_iter, residual_l2);
    // Deconstruct to break back out the result from the cost 
    while let Some(AnnotatedResult {
        result: cgi,
        annotation: euc,
        }) = cg_iter.next()
    {
        // Now the loop body is I/O only as it should be!
        println!(
            "||Ax - b||_2 = {:.5}, for x = {:.4}", euc, cgi.x
        );
    }
}

fn residual_linf(result: &CGIterable) -> f64 {
    (result.a.dot(&result.x) - &result.b).fold(0.0, |a, b| a.max(b.abs()))
}

fn cg_demo_pt2_2() {
    let p = make_3x3_pd_system_2();
    let cg_iter = CGIterable::conjugate_gradient(p);
    // Capping the number of iterations is good for a demo...
    // and surprisingly common elsewhere too.
    let cg_iter = cg_iter.take(20);
    
    // 
    let cg_iter = step_by(cg_iter, 1);
    let cg_iter = time(cg_iter);
    let cg_iter = assess(cg_iter, | TimedResult { result, .. }| {
        (residual_l2(result), residual_linf(result))
    });
    fn small_residual((euc, linf): &(f64, f64)) -> bool {
        euc < &1e-6 && linf < &1e-6
    }
    let mut cg_iter = cg_iter.take_while(
        | AnnotatedResult {
            annotation: metrics,
            ..
        }| !small_residual(metrics));

    while let Some(AnnotatedResult {
        result: TimedResult {
            result: cgi,
            start_time,
            duration,
        },
        annotation: (euc, linf),
    }) = cg_iter.next()
    {
        println!(
            "{:8} : {:8} | ||Ax - b||_2 = {:.5}, ||Ax - b||_inf = {:.5}, for x = {:.4}, residual = {:.7}", start_time.as_nanos(), duration.as_nanos(),
            euc, linf, cgi.x, cgi.r
        );
    }
}

fn main() {
    cg_demo();
    cg_demo_pt1();
    cg_demo_pt2_1();
    cg_demo_pt2_2();
}

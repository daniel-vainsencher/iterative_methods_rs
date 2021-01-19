//! # iterative-methods
//! A demonstration of the use of StreamingIterators and their adapters to implement iterative algorithms.
#[cfg(test)]
extern crate quickcheck;

use ndarray::*;
use std::time::{Duration, Instant};
use streaming_iterator::*;

/// State of Fibonacci iterator.
struct FibonnacciIterable<T> {
    s0: T,
    s1: T,
}

impl FibonnacciIterable<f64> {
    fn start(first: f64, second: f64) -> FibonnacciIterable<f64> {
        FibonnacciIterable::<f64> {
            s0: first,
            s1: second,
        }
    }
}

impl Iterator for FibonnacciIterable<f64> {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        let out = self.s0;
        self.s0 = self.s1;
        self.s1 = self.s0 + out;
        Some(out)
    }
}

/// Demonstrate usage of fibonacci sequence as an Iterator.
fn fib_demo() {
    let fib = FibonnacciIterable::start(0.0, 1.0);

    // enumerate is a simple iterator adaptor annotating the results
    // with their place in the sequence.
    for (i, n) in fib.enumerate().take(10) {
        println!("fib {} is {}", i, n)
    }
}

type S = f64;
type M = ArcArray2<S>;
type V = ArcArray1<S>;

/// A linear system, ax-b=0, to be solved iteratively, with an optional initial solution.
#[derive(Clone, Debug)]
struct LinearSystem {
    a: M,
    b: V,
    x0: Option<V>,
}

/// The state of a conjugate gradient algorithm.
#[derive(Clone)]
struct CGIterable {
    a: M,
    b: V,
    x: V,
    r: V,
    rs: S,
    rsprev: S,
    p: V,
    ap: Option<V>,
}

impl CGIterable {
    /// Convert a LinearSystem problem into a StreamingIterator of conjugate gradient solutions.
    pub fn conjugate_gradient(problem: LinearSystem) -> CGIterable {
        let x = match problem.x0 {
            None => ArrayBase::zeros(problem.a.shape()[1]),
            Some(init_x) => init_x,
        };
        let r = problem.b.clone() - problem.a.dot(&x).view();
        let rs = r.dot(&r);
        let cgi_p = r.clone();

        CGIterable {
            a: problem.a,
            b: problem.b,
            x,
            r,
            rs,
            rsprev: 1.,
            p: cgi_p,
            ap: None,
        }
    }
}

impl StreamingIterator for CGIterable {
    type Item = CGIterable;
    /// Implementation of conjugate gradient iteration
    fn advance(&mut self) {
        let ap = self.a.dot(&self.p).into_shared();
        let alpha = self.rs / self.p.dot(&ap);
        self.x += &(alpha * &self.p);
        self.r -= &(alpha * &ap);
        self.rsprev = self.rs;
        self.rs = self.r.dot(&self.r);
        self.p = (&self.r + &(&self.rs / self.rsprev * &self.p)).into_shared();
        self.ap = Some(ap);
    }
    fn get(&self) -> Option<&Self::Item> {
        if (!self.rsprev.is_normal()) || self.rsprev.abs() <= std::f64::MIN * 10. {
            None
        } else {
            Some(self)
        }
    }
}

/// Pass the values from the streaming iterator through, running a
/// function on each for side effects.
struct Tee<I, F> {
    it: I,
    f: F,
}

/*
// TODO: For ideal convenience, this should be implemented inside the StreamingIterator trait.
impl StreamingIterator<Item=T> {
    fn tee<F>(self, f: F) -> Tee<Self, F>
    where
        Self: Sized,
        F: Fn(&Self::Item)
    {
        Tee {
            it: self,
            f: f
        }
    }
} */

fn tee<I, F, T>(it: I, f: F) -> Tee<I, F>
where
    I: Sized + StreamingIterator<Item = T>,
    F: Fn(&T),
{
    Tee { it: it, f: f }
}

impl<I, F> StreamingIterator for Tee<I, F>
where
    I: StreamingIterator,
    F: FnMut(&I::Item),
{
    type Item = I::Item;

    #[inline]
    fn advance(&mut self) {
        // The side effect happens exactly once for each new value
        // generated.
        self.it.advance();
        if let Some(x) = self.it.get() {
            (self.f)(x);
        }
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        self.it.get()
    }
}

fn solve_approximately(p: LinearSystem) -> V {
    let mut solution = CGIterable::conjugate_gradient(p).take(200);
    last(solution.map(|s| s.x.clone()))
}

fn show_progress(p: LinearSystem) {
    let cg_iter = CGIterable::conjugate_gradient(p).take(50);
    //.take_while(|cgi| cgi.rsprev.sqrt() > 1e-6);
    let mut cg_print_iter = tee(cg_iter, |result| {
        let res = result.a.dot(&result.x) - &result.b;
        let res_norm = res.dot(&res);
        println!(
            "rs = {:.10}, ||Ax - b ||_2 = {:.5}, for x = {:.4}, and Ax - b = {:.5}",
            result.rs,
            res_norm,
            result.x,
            result.a.dot(&result.x) - &result.b,
        );
    });
    while let Some(_cgi) = cg_print_iter.next() {}
}

fn make_3x3_psd_system_1() -> LinearSystem {
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
    println!("a: {}", &p.a);
    let cg_iter = CGIterable::conjugate_gradient(p)
        // Upper bound the number of iterations
        .take(20)
        // Apply a quality based stopping condition; this relies on
        // algorithm internals, requiring all state to be exposed and
        // not just the result.
        .take_while(|cgi| cgi.rsprev.sqrt() > 1e-6);
    // Because time, tee are not part of the StreamingIterator trait,
    // they cannot be chained as in the above. Note the side effect of
    // tee is applied exactly to every x produced above, the sequence
    // of which is not affected at all. This is just like applying a
    // side effect inside the while loop, except we can compose
    // multiple tee, each with its own effect.
    // TODO can this be fixed? see iterutils crate.
    let step_by_cg_iter = step_by(cg_iter, 2);
    let timed_cg_iter = time(step_by_cg_iter);
    let mut cg_print_iter = tee(timed_cg_iter, |TimedResult { result, duration }| {
        let res = result.a.dot(&result.x) - &result.b;
        let res_norm = res.dot(&res);
        println!(
            "||Ax - b ||_2 = {:.5}, for x = {:.4}, and Ax - b = {:.5}; iteration duration {}μs",
            res_norm,
            result.x,
            result.a.dot(&result.x) - &result.b,
            duration.as_nanos(),
        );
    });
    while let Some(_cgi) = cg_print_iter.next() {}
}

/// Time every call to `advance` on the underlying
/// StreamingIterator. The goal is that our get returns a pair that is
/// approximately (Duration, &I::Item), but the types are not lining
/// up just yet.
struct TimedIterable<I, T>
where
    I: StreamingIterator<Item = T>,
{
    it: I,
    last: Option<TimedResult<T>>,
}

struct TimedResult<T> {
    result: T,
    duration: Duration,
}

fn last<I, T>(it: I) -> T
where
    I: StreamingIterator<Item = T>,
    T: Sized + Clone,
{
    let last_some = it.fold(None, |acc, i| { Some((*i).clone())} );
    let last_item = last_some.expect("StreamingIterator last expects at least one non-None element.").clone();
    last_item
}


fn time<I, T>(it: I) -> TimedIterable<I, T>
where
    I: Sized + StreamingIterator<Item = T>,
    T: Sized,
{
    TimedIterable { it: it, last: None }
}

impl<I, T> StreamingIterator for TimedIterable<I, T>
where
    I: StreamingIterator<Item = T>,
    T: Sized + Clone,
{
    type Item = TimedResult<T>;

    fn advance(&mut self) {
        let before = Instant::now();
        self.it.advance();
        self.last = match self.it.get() {
            Some(n) => Some(TimedResult {
                duration: before.elapsed(),
                result: n.clone(),
            }),
            None => None,
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        match &self.last {
            Some(tr) => Some(&tr),
            None => None,
        }
    }
}

/// Adapt StreamingIterator to only return values every 'step' number of times.
///
/// This is a StreamingIterator version of Iterator::step_by
///(https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.step_by)
///
/// The iterator adaptor step_by(it, step) wraps a StreamingIterator. A
/// 'step' is specified and only the items located every 'step' are returned.
///
///Iterator indices begin at 0, thus step_by() converts step -> step - 1
struct StepBy<I> {
    it: I,
    step: usize,
    first_take: bool,
}

fn step_by<I, T>(it: I, step: usize) -> StepBy<I>
where
    I: Sized + StreamingIterator<Item = T>,
{
    assert!(step != 0);
    StepBy {
        it,
        step: step - 1,
        first_take: true,
    }
}

impl<I> StreamingIterator for StepBy<I>
where
    I: StreamingIterator,
{
    type Item = I::Item;

    #[inline]
    fn advance(&mut self) {
        if self.first_take {
            self.first_take = false;
            self.it.advance();
        } else {
            self.it.nth(self.step);
        }
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        self.it.get()
    }
}

/// Call the different demos.
fn main() {
    println!("\n fib_demo:\n");
    fib_demo();
    println!("\n cg_demo: \n");
    cg_demo();
}

/// Unit Tests Module
#[cfg(test)]
mod tests {
    use quickcheck::{quickcheck, TestResult};
    quickcheck! {
        fn prop(xs: Vec<f64>) -> TestResult {
            // TODO(daniel): replace by test of squareness
            if xs.len() != 9 {
                return TestResult::discard();
            }
            let v = rcarr1(&xs).reshape((3, 3));
            println!("v.sum(): {}", v.sum());
            let b = rcarr1(&[1.,2.,3.]);
            let p = make_3x3_psd_system(v, b);
            if !p.a.scalar_sum().abs().is_finite() {
                // If a is too weird, we really want to discard, but
                // for the current check seems to discard everything;
                // for exploration making that fail prints the matrix.
                //return TestResult::discard()
                return TestResult::from_bool(false)
            }
            let x = solve_approximately(p.clone());
            let res = p.a.dot(&x) - &p.b;
            let res_norm = res.dot(&res);
            println!("a: {}", p.a);
            println!("b: {}", p.b);
            println!("x: {}", x);

            //TestResult::from_bool(res_norm < 1e-3)
            TestResult::from_bool(false)
        }
    }

    use super::*;

    #[test]
    fn test_last() {
        let v = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut iter = convert(v.clone());
        assert!(last(iter) == 9);
    }

    #[test]
    #[should_panic(expected = "StreamingIterator last expects at least one non-None element.")]
    fn test_last_fail() {
        let v : Vec<u32> = vec![];
        let mut iter = convert(v.clone());
        last(iter);
    }

    #[test]
    fn cg_simple_test() {
        let p = make_3x3_psd_system_1();
        println!("Problem is: {:?}", p);
        show_progress(p.clone());
        let x = solve_approximately(p.clone());
        let r = p.a.dot(&x) - p.b;
        println!("Residual is: {}", r);
        let res_norm = r.dot(&r);
        println!("Residual norm is: {}", res_norm);
        assert!(res_norm < 1e-10);
    }

    #[test]
    fn cg_simple_passed() {
        let p = LinearSystem {
            a: rcarr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]),
            b: rcarr1(&[0.0, 1., 0.]),
            x0: None,
        };

        println!("Problem is: {:?}", p);
        show_progress(p.clone());
        let x = solve_approximately(p.clone());
        let r = p.a.dot(&x) - p.b;
        println!("Residual is: {}", r);
        let res_norm = r.dot(&r);
        println!("Residual norm is: {}", res_norm);
        assert!(res_norm < 1e-10);
    }

    #[test]
    fn step_by_test() {
        let v = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let iter = convert(v);
        let mut iter = step_by(iter, 3);
        let mut _index = 0i64;
        while let Some(element) = iter.next() {
            assert_eq!(*element, _index * 3);
            _index = _index + 1;
        }
    }
}

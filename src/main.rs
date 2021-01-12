//! # iterative-methods
//! A demonstration of the use of StreamingIterators and their adapters to implement iterative algorithms.

use ndarray::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::cmp::PartialEq;
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
            x: x,
            r: r,
            rs: rs,
            rsprev: 0.,
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
        self.p = (&self.r + &((&self.r / self.rsprev) * &self.p)).into_shared();
        self.ap = Some(ap);
    }
    fn get(&self) -> Option<&Self::Item> {
        Some(self)
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

/// Demonstrate usage and convergence of conjugate gradient as a streaming-iterator.
fn cg_demo() {
    let a = rcarr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]]);
    let b = rcarr1(&[0., 1., 0.]);
    let p = LinearSystem {
        a: a,
        b: b,
        x0: None,
    };
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
    let step_by_cg_iter = step_by(cg_iter, 4);
    let timed_cg_iter = time(step_by_cg_iter);
    let mut cg_print_iter = tee(timed_cg_iter, |TimedResult { result, duration }| {
        println!(
            "||Ax - b ||_2 = {:.5}, for x = {:.4}, and Ax - b = {:.5}; iteration duration {}μs",
            result.rsprev.sqrt(),
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

/// Weighted Sampling
/// The WeightedDatum struct wraps the values of a data set to include
/// a weight for each datum. Currently, the main motivation for this
/// is to use it for Weighted Reservoir Sampling.

// Switch type of weight to Option<f64>
#[derive(Debug, Clone, PartialEq)]
struct WeightedDatum<U> {
    value: U,
    weight: f64,
}

fn new_datum<U>(value: U, weight: f64) -> WeightedDatum<U>
where
    U: Clone,
{
    WeightedDatum {
        value: value,
        weight: weight,
    }
}

/// Weighted Reservoir Sampling
/// The weighted reservoir sampling algorithm of M. T. Chao is implemented.
/// See https://en.wikipedia.org/wiki/Reservoir_sampling#Weighted_random_sampling
/// or for the original paper, https://doi.org/10.1093/biomet/69.3.653.

/// Future work might include implementing paralellized batch processing:
/// https://dl.acm.org/doi/10.1145/3350755.3400287
#[derive(Debug, Clone)]
struct ReservoirSampleIterator<I, T> {
    it: I,
    reservoir: Vec<T>,
    capacity: usize,
    // total weight -- change this to generic type
    w_sum: f64,
    /// The oracle is currently Pcg64, which allows seeded rng. This should be
    /// extended to generic type bound by traits for implementing seeding.
    oracle: Pcg64,
}

// Create a ReservoirSampleIterator
fn reservoir_sample<I, T>(
    it: I,
    capacity: usize,
    custom_oracle: Option<Pcg64>,
) -> ReservoirSampleIterator<I, T>
where
    I: Sized + StreamingIterator<Item = T>,
    T: Clone,
{
    let oracle = match custom_oracle {
        Some(oracle) => oracle,
        None => Pcg64::seed_from_u64(1),
    };
    let res: Vec<T> = Vec::new();
    ReservoirSampleIterator {
        it,
        reservoir: res,
        capacity: capacity,
        w_sum: 0.0,
        oracle: oracle,
    }
}

impl<I, U> StreamingIterator for ReservoirSampleIterator<I, WeightedDatum<U>>
where
    U: Clone,
    I: StreamingIterator<Item = WeightedDatum<U>>,
{
    // type Item = I::Item;
    type Item = WeightedDatum<U>;

    #[inline]
    fn advance(&mut self) {
        if self.reservoir.len() < self.capacity {
            self.it.advance();
            if let Some(datum) = self.it.get() {
                let cloned_datum = datum.clone();
                self.reservoir.push(cloned_datum);
                self.w_sum += datum.weight;
            }
        } else {
            // will this skip a datum?
            if let Some(datum) = self.it.next() {
                self.w_sum += datum.weight;
                let p = &(datum.weight / self.w_sum);
                let j: f64 = self.oracle.gen();
                if j < *p {
                    let h = self.oracle.gen_range(0..self.capacity) as usize;
                    let datum_struct = datum.clone();
                    self.reservoir[h] = datum_struct;
                };
            }
        }
    }

    // let mut oracle1_iter = rs_iter.oracle1.iter();
    // let mut oracle2_iter = rs_iter.oracle2.iter();
    // rs_iter.w_sum += datum.weight;
    // let p = &(datum.weight / self.w_sum);
    // if let Some(j) = oracle1_iter.next() {
    // if j < p {
    //     if let Some(h) = oracle2_iter.next() {
    //         let datum_struct = *datum;
    //         self.reservoir.insert(*h, datum_struct);
    //     };
    // }
    // };

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

// Unit Tests Module
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // Test the step_by adaptor
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

    // Tests for the ReservoirSampleIterator adaptor
    #[test]
    fn test_datum_struct() {
        let samp = new_datum(String::from("hi"), 1.0);
        assert_eq!(samp.value, String::from("hi"));
        assert_eq!(samp.weight, 1.0);
    }

    #[test]
    fn fill_reservoir_test() {
        // think of v as the vec of weights, so samples are just weights
        let v: Vec<WeightedDatum<f64>> = vec![new_datum(0.5, 1.), new_datum(0.2, 2.)];
        let v_copy = v.clone();
        let iter = convert(v);
        let oracle1 = vec![0.1, 0.6, 0.4, 0.3, 0.5];
        let oracle2 = vec![0, 1, 2, 3, 4];
        let mut iter = reservoir_sample(iter, 2, None);
        for _element in v_copy {
            iter.advance();
        }
        assert_eq!(
            iter.reservoir[0],
            WeightedDatum {
                value: 0.5f64,
                weight: 1.0f64
            }
        );
        assert_eq!(
            iter.reservoir[1],
            WeightedDatum {
                value: 0.2f64,
                weight: 2.0f64
            }
        );
    }
}

//! # iterative-methods
//! A demonstration of the use of StreamingIterators and their adapters to implement iterative algorithms.

use ndarray::*;
use streaming_iterator::*;

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

/// Demostrate usage of fibonacci sequence as an Iterator.
fn fib_demo() {
    let fib = FibonnacciIterable::start(0.0, 1.0);

    for (i, n) in fib.enumerate().take(10) {
        println!("fib {} is {}", i, n)
    }
}

type S = f64;
type M = Array2<S>;
type V = Array1<S>;

/// A linear system to be solved iteratively, with an optional initial solution.
struct LinearSystem {
    a: M,
    b: V,
    x0: Option<V>,
}

/// The state of a conjugate gradient algorithm.
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
    fn advance(&mut self) {
        let ap = self.a.dot(&self.p);
        let alpha = self.rs / self.p.dot(&ap);
        self.x = &self.x + &(alpha * &self.p);
        self.r = &self.r - &(alpha * &ap);
        self.rsprev = self.rs;
        self.rs = self.r.dot(&self.r);
        self.p = &self.r + &((&self.r / self.rsprev) * &self.p);
        self.ap = Some(ap);
    }
    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }
}

/// Demonstrate usage and convergence of conjugate gradient as a streaming-iterator.
fn cg_demo() {
    let a = arr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]]);
    let b = arr1(&[0., 1., 0.]);
    let p = LinearSystem {
        a: a,
        b: b,
        x0: None,
    };
    let mut cg_iter = CGIterable::conjugate_gradient(p).take(10);
    while let Some(cgi) = cg_iter.next() {
        println!(
            "||Ax - b ||_2 = {:.5}, for x = {:.4}, and Ax - b = {:.5}",
            cgi.rsprev.sqrt(),
            cgi.x,
            cgi.a.dot(&cgi.x) - &cgi.b,
        );
    }
}

/// Call the different demos.
fn main() {
    fib_demo();
    cg_demo();
}

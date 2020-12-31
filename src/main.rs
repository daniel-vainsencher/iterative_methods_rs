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
        let out = Some(self.s0);
        let n = self.s0 + self.s1;
        self.s0 = self.s1;
        self.s1 = n;
        out
    }
}

fn fib_demo() {
    let fib = FibonnacciIterable::start(0.0, 1.0);

    for n in fib.take(5) {
        println!("fib: {}", n)
    }
}

type S = f64;
type M = Array2<S>;
type V = Array1<S>;

struct CGProblem {
    a: M,
    b: V,
    x0: Option<V>,
}

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
    fn conjugate_gradient(p: CGProblem) -> CGIterable {
        let x = match p.x0 {
            None => ArrayBase::zeros(p.a.shape()[1]),
            Some(init_x) => init_x,
        };
        let r = p.b.clone() - p.a.dot(&x).view();
        let rs = r.dot(&r);
        let cgi_p = r.clone();

        CGIterable {
            a: p.a,
            b: p.b,
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

fn cg_demo() {
    let a = arr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.5, 1.0]]);
    let b = arr1(&[0., 1., 0.]);
    let p = CGProblem {
        a: a,
        b: b,
        x0: None,
    };
    let mut cg_iter = CGIterable::conjugate_gradient(p).take(10);
    while let Some(cgi) = cg_iter.next() {
        println!(
            "x is {:.4}, residual is {:.5}",
            cgi.x,
            cgi.a.dot(&cgi.x) - &cgi.b
        );
    }
}

fn main() {
    fib_demo();
    cg_demo();
}

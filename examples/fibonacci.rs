use streaming_iterator::*;

/// State of Fibonacci iterator.
struct FibonacciIterable<T> {
    s0: T,
    s1: T,
}

impl FibonacciIterable<f64> {
    fn start(first: f64, second: f64) -> FibonacciIterable<f64> {
        FibonacciIterable::<f64> {
            s0: first,
            s1: second,
        }
    }
}

impl Iterator for FibonacciIterable<f64> {
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
    let fib = FibonacciIterable::start(0.0, 1.0);

    // enumerate is a simple iterator adaptor annotating the results
    // with their place in the sequence.
    for (i, n) in fib.enumerate().take(10) {
        println!("fib {} is {}", i, n)
    }
}

fn main() {
    fib_demo();
}

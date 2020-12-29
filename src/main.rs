struct FibonnacciIterable<T> {
    s0: T,
    s1: T,
}

impl FibonnacciIterable<f64> {
    fn start(first: f64, second: f64) -> FibonnacciIterable<f64> {
        FibonnacciIterable::<f64>{s0: first, s1: second}
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

fn main() {
    let mut fib = FibonnacciIterable::start(0.0, 1.0);

    for n in fib.take(5) {
        println!("fib: {}", n)
    }
}

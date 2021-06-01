//! Example from top level crate documentation
use iterative_methods::derivative_descent::*;
use iterative_methods::*;
use streaming_iterator::*;

fn main() {
    // Problem: minimize the convex parabola f(x) = x^2 + x
    let function = |x| x * x + x;

    // An iterative solution by gradient descent
    let derivative = |x| 2.0 * x + 1.0;
    let step_size = 0.2;
    let x_0 = 2.0;

    // Au naturale:
    let mut x = x_0;
    for i in 0..10 {
        x -= step_size * derivative(x);
        println!("x_{} = {:.2}; f(x_{}) = {:.4}", i, x, i, x * x + x);
    }

    // Using replaceable components:
    let dd = DerivativeDescent::new(function, derivative, step_size, x_0);
    let dd = enumerate(dd);
    let mut dd = dd.take(10);
    while let Some(&Numbered {
        item: Some(ref curr),
        count,
    }) = dd.next()
    {
        println!(
            "x_{} = {:.2}; f(x_{}) = {:.4}",
            count,
            curr.x,
            count,
            curr.value()
        );
    }
}

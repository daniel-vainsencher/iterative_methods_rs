use iterative_methods::*;
use streaming_iterator::*;

/// A simple Counter iterator to use in demos and tests.
#[derive(Clone, Debug)]
pub struct Counter {
    count: f64,
}

pub fn new_counter() -> Counter {
    Counter { count: 0. }
}

impl StreamingIterator for Counter {
    type Item = f64;

    fn advance(&mut self) {
        self.count += 1.;
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(&self.count)
    }
}

/// Expose the weight of the counter iterable. This is an arbitrary choice of weight.
/// ***WARNING: this function is duplicated in integration_tests.rs. Find a single location for it.
pub fn expose_w(count: &f64) -> f64 {
    count * count
}

fn wd_iterable_counter_demo() {
    println!("\n\n -----WDIterable Counter Demo----- \n\n");

    let counter_stream: Counter = new_counter();
    let mut counter_stream_copy = counter_stream.clone();
    let mut wd_iter = WDIterable {
        it: counter_stream,
        f: expose_w,
        wd: Some(new_datum(0., 0.)),
    };

    println!("A stream of values:\n");
    for _ in 0..6 {
        if let Some(val) = counter_stream_copy.next() {
            print!("{}\n", val);
        }
    }
    println!("\n\nThe stream with weights added that are the square of the value.\n\n");
    for _ in 0..6 {
        if let Some(wd) = wd_iter.next() {
            print!("{:#?}\n", wd);
        }
    }
}

// In Process: a demo using a stream of strings; weight is length
// fn wd_iterable_string_demo() {
//
// }

fn main() {
    wd_iterable_counter_demo();
}

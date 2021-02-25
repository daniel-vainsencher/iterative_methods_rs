use iterative_methods::*;
extern crate streaming_iterator;
use crate::streaming_iterator::StreamingIterator;
// mod algorithms;
use crate::algorithms::cg_method::*;
use ndarray::*;

#[test]
fn test_timed_iterable() {
    let p = make_3x3_psd_system_1();
    let cg_iter = CGIterable::conjugate_gradient(p).take(50);
    let cg_timed_iter = time(cg_iter);
    let mut start_times = Vec::new();
    let mut durations = Vec::new();

    let mut cg_print_iter = tee(
        cg_timed_iter,
        |TimedResult {
             result: _,
             start_time,
             duration,
         }| {
            start_times.push(start_time.as_nanos());
            durations.push(duration.as_nanos());
        },
    );
    while let Some(_x) = cg_print_iter.next() {}
    println!("Start times: {:?}", start_times);
    println!("Durations: {:?}", durations);
    let start_times = rcarr1(&start_times).map(|i| *i as f64);
    let st_diff = &start_times.slice(s![1..]) - &start_times.slice(s![..-1]);
    println!("start time diffs: {:?}", st_diff);
    // Ensure times are within factor 10 of typical value observed in dev
    assert!(durations.iter().all(|dur| 3000 < *dur && *dur < 300000));
    // Ensure that start times are strictly increasing.
    assert!(st_diff.iter().all(|diff| *diff >= 0.));
}

/// Test that WDIterable followed by ExtractValue is a roundtrip.
///
/// WDIterable wraps the items of a simple Counter iterable as WeightedDatum
/// with the square of the count as the weight. Then ExtractValue unwraps, leaving
/// items with only the original value. The items of a clone of the original iterator
/// and the wrapped/unwrapped iterator are checked to be equal.

/// ***WARNING: this function is duplicated in integration_tests.rs. Find a single location for it.
pub fn expose_w(count: &f64) -> f64 {
    count * count
}

#[test]
fn wd_iterable_extract_value_test() {
    let mut counter_stream: Counter = new_counter();
    let counter_stream_copy = counter_stream.clone();
    let wd_iter = WDIterable {
        it: counter_stream_copy,
        f: expose_w,
        wd: Some(new_datum(0., 0.)),
    };

    let mut extract_value_iter = extract_value(wd_iter);

    for _ in 0..6 {
        if let (Some(val1), Some(val2)) = (extract_value_iter.next(), counter_stream.next()) {
            assert!(val1 == val2);
        }
    }
}

use iterative_methods;

mod algorithms;

use algorithms::*;

// use eigenvalues::algorithms::lanczos::HermitianLanczos;
// use eigenvalues::SpectrumTarget;
// use na::{DMatrix, DVector, Dynamic};
// use ndarray::*;
// use quickcheck::{quickcheck, TestResult};

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

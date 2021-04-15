use crate::utils;
use iterative_methods::*;
extern crate streaming_iterator;
use crate::algorithms::cg_method::*;
use crate::streaming_iterator::*;
use ndarray::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;

#[ignore]
#[test]
fn test_timed_iterable() {
    let p = utils::make_3x3_psd_system_1();
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

#[test]
fn wd_iterable_extract_value_test() {
    let mut counter_stream: utils::Counter = utils::new_counter();
    let counter_stream_copy = counter_stream.clone();
    let wd_iter = WDIterable {
        it: counter_stream_copy,
        f: utils::expose_w,
        wd: Some(new_datum(0., 0.)),
    };

    let mut extract_value_iter = extract_value(wd_iter);

    for _ in 0..6 {
        if let (Some(val1), Some(val2)) = (extract_value_iter.next(), counter_stream.next()) {
            assert!(val1 == val2);
        }
    }
}

/// Test the integration of ReservoirIterable, Enumerate, and ToFileIterable.
///
/// A stream of 2 zeros and 8 ones subjected to reservoir sampling using a seeded rng.
/// The stream of reservoirs is adapted with enumerate() and then write_yaml_documents(). After
/// running the iteration the contents of the file are checked against a string.
#[test]
fn enumerate_reservoirs_to_yaml_test() {
    let test_file_path = "enumerate_reservoirs_to_yaml_test1.yaml";
    let stream_length = 10usize;
    let capacity = 2usize;
    let initial_value = 0i64;
    let final_value = 1i64;
    let stream = utils::generate_step_stream(stream_length, capacity, initial_value, final_value);
    let stream = reservoir_iterable(stream, capacity, Some(Pcg64::seed_from_u64(0)));
    let stream = enumerate(stream);
    let mut stream = write_yaml_documents(stream, String::from(test_file_path))
        .expect("Write scalar to Yaml: Create file and initialize Yaml iter failed.");
    while let Some(t) = stream.next() {
        println!("{:?}", t);
    }
    let contents = utils::read_yaml_to_string(test_file_path).expect("Could not read file.");
    let output =
        String::from("---\n- 0\n- - 0\n  - 0\n---\n- 1\n- - 0\n  - 1\n---\n- 2\n- - 1\n  - 1\n");
    assert_eq!(contents, output);
}

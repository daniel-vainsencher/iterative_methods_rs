use crate::utils::*;
use iterative_methods::*;
extern crate streaming_iterator;
use crate::streaming_iterator::*;
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

#[test]
fn enumerate_reservoirs_to_yaml_test() {
    let test_file_path = "enumerate_reservoirs_to_yaml_test.yaml";
    let stream_length = 4usize;
    let capacity = 2usize;
    let initial_value = 0i64;
    let final_value = 0i64;
    let stream = utils::generate_step_stream(stream_length, capacity, initial_value, final_value);
    let stream = convert(stream);
    let stream = reservoir_iterable(stream, capacity, None);
    let stream = enumerate(stream);
    let mut stream = item_to_file(stream, write_scalar_to_yaml, String::from(test_file_path))
        .expect("Write scalar to Yaml: Create file and initialize Yaml iter failed.");
    while let Some(t) = stream.next() {
        println!("{:?}", t);
    }
    let contents = utils::read_yaml_to_string(test_file_path).expect("Could not read file.");
    let output = String::from("---\n- 0\n- 0---\n- 1\n- 2---\n- 2\n- 4");
    assert_eq!(contents, output);
    // let mut read_file = File::open(test_file_path).expect("Could not open file with test data to asserteq.");
    // let mut contents = String::new();
    // read_file.read_to_string(&mut contents).expect("Could not read data from file.");
    // std::fs::remove_file(test_file_path).expect("Could not remove data file for test.")

    // let stream = stream_vec.iter();
    // let stream = convert(stream_vec);
    // let stream_c = stream.clone();
    // let mut stream_d = item_to_file(
    //     stream_c,
    //     write_vec_to_yaml_no_lifetime,
    //     String::from(population_file),
    // )
    // .expect("Create File and initialize yaml iter failed.");
    // while let Some(_) = stream_d.next() {}

    // Create the stream:
    // let stream = utils::generate_step_stream(stream_size, capacity, 0, 1);
    // let stream = convert(stream);
    // let res_iter = reservoir_iterable(stream, capacity, None);
    // let res_iter = step_by(res_iter, step);

    // let reservoir_samples_file = "./target/debug/examples/reservoirs.yaml";
    // // Write data to file for visualization.
    // let mut num_res: usize = 0;
    // let mut res_to_yaml = list_to_file(res_iter, write_vec_to_list, reservoir_samples_file.to_string())
    //     .expect("Create File and initialize yaml iter failed.");
    // while let Some(_item) = res_to_yaml.next() {
    //     num_res += 1;
    // }
    // parameters.insert("num_res", num_res);
    // let parameters_file_path = "./visualizations_python/parameters.yaml";
    // utils::write_parameters_to_yaml(parameters, parameters_file_path)?;
}

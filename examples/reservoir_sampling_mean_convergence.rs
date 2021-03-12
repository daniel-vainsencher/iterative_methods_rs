use crate::utils;
use iterative_methods::*;
use std::process::Command;
use streaming_iterator::*;

/// Create a stream of data, write the full stream and a
/// sequence of reservoir samples to yaml files,
/// return the stream_size, capacity, and step size used.
fn reservoir_sampling_mean_convergence_for_step() -> (usize, usize, usize, usize) {
    // Streamline up error handling
    let stream_size: usize = 10_i32.pow(4) as usize;
    let capacity: usize = 100;
    let step: usize = 1;
    println!(
        "The test uses a stream of size {:#?} and a reservoir capacity of {:#?}.",
        stream_size, capacity
    );
    let population_file = "./target/debug/examples/population.yaml";
    // Create the stream:
    let stream = utils::generate_step_stream(stream_size, capacity, 0, 1);
    let stream = convert(stream);
    // Create another copy of the stream to be turned into a vec:
    let mut stream_for_vec = utils::generate_step_stream(stream_size, capacity, 0, 1);
    let mut stream_vec: Vec<usize> = Vec::new();
    while let Some(value) = stream_for_vec.next() {
        stream_vec.push(value)
    }
    // Clear contents of the file.
    if let Err(error) = utils::clear_file(&population_file) {
        println!("{:#?}", error);
    };
    // Write to file for visualization
    if let Err(error) = utils::write_population_to_file(&stream_vec, &population_file) {
        println!("{:#?}", error);
    };

    let res_iter = reservoir_iterable(stream, capacity, None);
    let mut res_iter = step_by(res_iter, step);

    let reservoir_samples_file = "./target/debug/examples/reservoirs.yaml";
    // Clear conents of the file.
    if let Err(error) = utils::clear_file(&reservoir_samples_file) {
        println!("{:#?}", error);
    };
    // Write data to file for visualization.
    let mut num_res: usize = 0;
    while let Some(res) = res_iter.next() {
        num_res += 1;
        let mut res_values: Vec<usize> = Vec::new();
        for val in res {
            res_values.push(*val);
        }
        if let Err(error) = utils::write_res_to_file(&res_values, &reservoir_samples_file) {
            println!("{:#?}", error);
        }
    }

    (stream_size, capacity, step, num_res)
}

fn make_animations_in_python(
    stream_size: usize,
    capacity: usize,
    step: usize,
    num_res: usize,
    num_bins: usize,
) -> std::io::Result<()> {
    let output = Command::new("python3")
        .arg("./visualizations_python/reservoir_sampling_mean_convergence.py")
        .arg(stream_size.to_string())
        .arg(capacity.to_string())
        .arg(step.to_string())
        .arg(num_res.to_string())
        .arg(num_bins.to_string())
        .output()?;
    if !output.status.success() {
        println!(
            "Running reservoir_sampling_mean_convergence.py did not succeed. Error: \n\n{:#?}\n\n",
            output
        );
    } else {
        println!("{:#?}", output);
    };
    Ok(())
}

fn main() -> std::io::Result<()> {
    let num_bins = 50usize;
    let (stream_size, capacity, step, num_res) = reservoir_sampling_mean_convergence_for_step();
    make_animations_in_python(stream_size, capacity, step, num_res, num_bins)?;
    Ok(())
}

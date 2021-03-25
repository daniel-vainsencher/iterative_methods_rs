use crate::utils;
use iterative_methods::*;
use std::collections::HashMap;
use std::process::Command;
use streaming_iterator::*;

/// Create a stream of data, write the full stream and a
/// sequence of reservoir samples to yaml files. The stream
/// uses enumerated samples in order to track how much of the
/// stream has been used in each reservoir.
fn reservoir_sampling_mean_convergence_for_step() -> std::io::Result<()> {
    // Streamline up error handling
    let stream_size: usize = 10_i32.pow(3) as usize;
    let num_of_initial_values = stream_size / 2;
    let capacity: usize = 50;
    let mut parameters: HashMap<&str, usize> = HashMap::new();
    parameters.insert("stream_size", stream_size);
    parameters.insert("capacity", capacity);
    println!(
        "The test uses a stream of size {:#?} and a reservoir capacity of {:#?}.",
        stream_size, capacity
    );

    // Create a copy of the stream to be written to yaml:
    let population_file = "./target/debug/examples/population.yaml";
    let stream_to_yaml = utils::generate_step_stream(stream_size, num_of_initial_values, 0, 1);
    let stream_to_yaml = convert(stream_to_yaml);
    let stream_to_yaml = enumerate(stream_to_yaml);
    let mut stream_to_yaml = write_yaml_documents(stream_to_yaml, population_file.to_string())
        .expect("Create File and initialize yaml iter failed.");
    while let Some(_) = stream_to_yaml.next() {}

    // Create another copy of the stream to perform reservoir sampling and write to yaml:
    let stream = utils::generate_step_stream(stream_size, num_of_initial_values, 0, 1);
    let stream = convert(stream);
    let stream = enumerate(stream);
    let res_iter = reservoir_iterable(stream, capacity, None);
    let reservoir_samples_file = "./target/debug/examples/reservoirs.yaml";
    // Write data to file for visualization.
    let mut res_to_yaml = write_yaml_documents(res_iter, reservoir_samples_file.to_string())
        .expect("Create File and initialize yaml iter failed.");
    // num_res is used in the python script for visualizations to initialize the size of the array that will hold that data to visualize.
    let mut num_res = 0;
    while let Some(_item) = res_to_yaml.next() {
        num_res += 1
    }
    parameters.insert("num_res", num_res);
    let parameters_file_path = "./visualizations_python/parameters.yaml";
    utils::write_parameters_to_yaml(parameters, parameters_file_path)?;
    Ok(())
}

fn make_animations_in_python() -> std::io::Result<()> {
    let output = Command::new("python3")
        .arg("./visualizations_python/reservoir_sampling_mean_convergence.py")
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
    reservoir_sampling_mean_convergence_for_step()?;
    make_animations_in_python()?;
    Ok(())
}

use crate::utils;
use iterative_methods::*;
use std::collections::HashMap;
use std::process::Command;
use streaming_iterator::*;

/// Write the full stream and a sequence of reservoir samples
/// to yaml files. The stream
/// is enumerated in order to track how much of the
/// stream has been used in each reservoir.
fn reservoir_histogram_animation() -> std::io::Result<()> {
    // Streamline up error handling
    let stream_size: usize = 10_i32.pow(05) as usize;
    // let num_of_initial_values = stream_size / 2;
    let capacity: usize = 200;
    let mut parameters: HashMap<&str, String> = HashMap::new();
    parameters.insert("stream_size", stream_size.to_string());
    parameters.insert("capacity", capacity.to_string());
    println!(
        "The test uses a stream of size {:#?} and a reservoir capacity of {:#?}.",
        stream_size, capacity
    );
    let sigma = 0.25f64;
    parameters.insert("sigma", sigma.to_string());
    // Generate the data to use
    let mut stream_vec =
        utils::generate_stream_from_normal_distribution(stream_size, 0.25f64, sigma);
    let mut stream_vec_end =
        utils::generate_stream_from_normal_distribution(2 * stream_size, 0.75f64, sigma);
    stream_vec.append(&mut stream_vec_end);

    // Create a copy of the stream to be written to yaml:
    let population_file = "./target/debug/examples/population_for_histogram.yaml";
    parameters.insert("population_file", population_file.to_string());
    // let stream_to_yaml = utils::generate_step_stream(stream_size, num_of_initial_values, 0, 1);
    let population_stream = stream_vec.clone();
    let population_stream = population_stream.iter();
    let population_stream = convert(population_stream);
    let population_stream = enumerate(population_stream);
    let mut population_stream =
        write_yaml_documents(population_stream, population_file.to_string())
            .expect("Create File and initialize yaml iter failed.");
    while let Some(_) = population_stream.next() {}

    // Create another copy of the stream to perform reservoir sampling and write to yaml:
    // let stream = utils::generate_step_stream(stream_size, num_of_initial_values, 0, 1);
    let stream = stream_vec.iter();
    let stream = convert(stream);
    let stream = enumerate(stream);
    let res_iter = reservoir_iterable(stream, capacity, None);
    let res_iter = step_by(res_iter, 20);
    let reservoir_samples_file = "./target/debug/examples/reservoirs_for_histogram.yaml";
    parameters.insert("reservoir_samples_file", reservoir_samples_file.to_string());
    // Write data to file for visualization.
    let mut res_to_yaml = write_yaml_documents(res_iter, reservoir_samples_file.to_string())
        .expect("Create File and initialize yaml iter failed.");
    // num_res is used in the python script for visualizations to initialize the size of the array that will hold that data to visualize.
    let mut num_res = 0;
    while let Some(_item) = res_to_yaml.next() {
        num_res += 1
    }
    parameters.insert("num_res", num_res.to_string());
    let parameters_file_path = "./visualizations_python/parameters_for_histogram.yaml";
    parameters.insert("parameters_file_path", parameters_file_path.to_string());
    utils::write_parameters_to_yaml(parameters, parameters_file_path)?;
    Ok(())
}

fn make_animations_in_python() -> std::io::Result<()> {
    let output = Command::new("python3")
        .arg("./visualizations_python/reservoir_histogram_animation.py")
        .output()?;
    if !output.status.success() {
        println!(
            "Running reservoir_histogram_animation.py did not succeed. Error: {:#?}",
            output
        );
    } else {
        println!("{:#?}", output);
    };
    Ok(())
}

fn main() -> std::io::Result<()> {
    reservoir_histogram_animation()?;
    make_animations_in_python()?;
    Ok(())
}

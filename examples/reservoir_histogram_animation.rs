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
    let stream_size: usize = 10_i32.pow(04) as usize;
    let num_initial_values = stream_size / 4;
    let num_final_values = 3 * stream_size / 4;
    let capacity: usize = 10;
    let mut parameters: HashMap<&str, String> = HashMap::new();
    // Define file paths for yaml data
    let population_file = "./target/debug/examples/population_for_histogram.yaml";
    let reservoir_samples_file = "./target/debug/examples/reservoirs_for_histogram.yaml";
    let parameters_file_path = "./visualizations_python/parameters_for_histogram.yaml";

    parameters.insert("stream_size", stream_size.to_string());
    parameters.insert("num_initial_values", num_initial_values.to_string());
    parameters.insert("num_final_values", num_final_values.to_string());
    parameters.insert("capacity", capacity.to_string());
    parameters.insert("population_file", population_file.to_string());
    parameters.insert("reservoir_samples_file", reservoir_samples_file.to_string());
    parameters.insert("parameters_file_path", parameters_file_path.to_string());
    println!(
        "The test uses a stream of size {:#?} and a reservoir capacity of {:#?}.",
        stream_size, capacity
    );
    let sigma = 0.15f64;
    let mean_initial = 0.25f64;
    let mean_final = 0.75f64;
    parameters.insert("sigma", sigma.to_string());
    
    // Generate the data as a vec
    let mut stream_vec =
        utils::generate_stream_from_normal_distribution(num_initial_values, mean_initial, sigma);
    let mut stream_vec_end =
        utils::generate_stream_from_normal_distribution(num_final_values, mean_final, sigma);
    stream_vec.append(&mut stream_vec_end);

    // Convert the data to a StreamingIterator and adapt
    let stream = stream_vec.iter();
    let stream = convert(stream);
    let stream = enumerate(stream);
    let stream = write_yaml_documents(stream, population_file.to_string()).expect("Create File and initialize yaml iter failed.");
    let stream = reservoir_iterable(stream, capacity, None);
    let stream = step_by(stream, 20);
    let mut stream = write_yaml_documents(stream, reservoir_samples_file.to_string()).expect("Create File and initialize yaml iter failed.");
    // num_res is used in the python script for visualizations to initialize the size of the array that will hold that data to visualize.
    let mut num_res = 0;
    while let Some(item) = stream.next() {
        if num_res < 3 {
            let max_index = item.iter().map(|numbered| numbered.count).max();
            let mean: f64 = item.iter().map(|numbered| numbered.item.unwrap()).sum();
            let mean = mean / (capacity as f64);
            println!("Mean: {} Max Index: {:#?}", mean, max_index);
        }
        num_res += 1
    }
    parameters.insert("num_res", num_res.to_string());
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
        println!("Animation exported successfully.");
    };
    Ok(())
}

fn make_initial_final_histograms_in_python() -> std::io::Result<()> {
    let output = Command::new("python3")
        .arg("./visualizations_python/reservoir_histograms_initial_final.py")
        .output()?;
    if !output.status.success() {
        println!(
            "Running reservoir_histograms_initial_final.py did not succeed. Error: {:#?}",
            output
        );
    } else {
        println!("Still Image exported successfully.");
    };
    Ok(())
}

fn main() -> std::io::Result<()> {
    reservoir_histogram_animation()?;
    println!("Data is written to yaml files.");
    make_initial_final_histograms_in_python()?;
    make_animations_in_python()?;
    Ok(())
}

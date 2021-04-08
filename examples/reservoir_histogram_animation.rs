use crate::utils;
use iterative_methods::*;
use std::cmp;
use std::collections::HashMap;
use std::process::Command;
use streaming_iterator::*;

/// Write the full stream and a sequence of reservoir samples
/// to yaml files. The stream
/// is enumerated in order to track how much of the
/// stream has been used in each reservoir.
fn reservoir_histogram_animation() -> Result<Vec<String>, std::io::Error> {
    // Streamline up error handling
    let stream_size: usize = 10_i32.pow(04) as usize;
    let num_initial_values = stream_size / 4;
    let num_final_values = 3 * stream_size / 4;
    let capacity: usize = 500;
    let num_bins: usize = 50;
    let mut parameters: HashMap<&str, String> = HashMap::new();
    // Define file paths for yaml data
    let population_file = "./target/debug/examples/population_for_histogram.yaml";
    let reservoir_samples_file = "./target/debug/examples/reservoirs_for_histogram.yaml";
    let parameters_file_path = "./visualizations_python/parameters_for_histogram.yaml";
    let reservoir_means_file = "./target/debug/examples/reservoir_means.yaml";
    let file_list = vec![
        population_file.to_string(),
        reservoir_samples_file.to_string(),
        parameters_file_path.to_string(),
        reservoir_means_file.to_string(),
    ];

    parameters.insert("stream_size", stream_size.to_string());
    parameters.insert("num_initial_values", num_initial_values.to_string());
    parameters.insert("num_final_values", num_final_values.to_string());
    parameters.insert("capacity", capacity.to_string());
    parameters.insert("population_file", population_file.to_string());
    parameters.insert("reservoir_samples_file", reservoir_samples_file.to_string());
    parameters.insert("parameters_file_path", parameters_file_path.to_string());
    parameters.insert("reservoir_means_file", reservoir_means_file.to_string());
    parameters.insert("num_bins", num_bins.to_string());
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
    let stream = write_yaml_documents(stream, population_file.to_string())
        .expect("Create File and initialize yaml iter failed.");
    let stream = reservoir_iterable(stream, capacity, None);
    let stream = step_by(stream, 20);
    let stream = write_yaml_documents(stream, reservoir_samples_file.to_string())
        .expect("Create File and initialize yaml iter failed.");
    let reservoir_mean_and_max_index = |reservoir: &Vec<Numbered<&f64>>| -> Numbered<f64> {
        let mut max_index = 0i64;
        let mean: f64 = reservoir
            .iter()
            .map(|numbered| {
                max_index = cmp::max(max_index, numbered.count);
                numbered.item.unwrap()
            })
            .sum();
        let mean = mean / (capacity as f64);
        Numbered {
            count: max_index,
            item: Some(mean),
        }
    };
    let stream = stream.map(reservoir_mean_and_max_index);
    let mut stream = write_yaml_documents(stream, reservoir_means_file.to_string())
        .expect("Create File and initialize yaml iter failed.");
    // num_res is used in the python script for visualizations to initialize the size of the array that will hold that data to visualize.
    let mut num_res = 0i64;
    while let Some(_item) = stream.next() {
        num_res += 1;
    }
    parameters.insert("num_res", num_res.to_string());
    utils::write_parameters_to_yaml(parameters, parameters_file_path)?;
    Ok(file_list)
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
        println!("Initial and Final Histograms exported successfully.");
    };
    Ok(())
}

fn make_reservoir_means_plot_in_python() -> std::io::Result<()> {
    let output = Command::new("python3")
        .arg("./visualizations_python/reservoir_means.py")
        .output()?;
    if !output.status.success() {
        println!(
            "Running reservoir_means.py did not succeed. Error: {:#?}",
            output
        );
    } else {
        println!("Reservoir Means Plot exported successfully.");
    };
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

fn remove_yaml_files(file_list: Vec<String>) -> std::io::Result<()> {
    for file in file_list {
        let output = Command::new("rm").arg(file).output()?;
        if !output.status.success() {
            println!(
                "Running reservoir_histogram_animation.py did not succeed. Error: {:#?}",
                output
            );
        }
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let file_list = reservoir_histogram_animation()?;
    println!("Data is written to yaml files.");
    make_initial_final_histograms_in_python()?;
    make_reservoir_means_plot_in_python()?;
    make_animations_in_python()?;
    remove_yaml_files(file_list)?;
    Ok(())
}
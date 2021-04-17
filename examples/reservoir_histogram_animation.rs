use crate::utils;
use iterative_methods::*;
use std::cmp;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::process::Command;
use std::str::FromStr;
use streaming_iterator::*;

/// Write the full stream and a sequence of reservoir samples
/// to yaml files. The stream
/// is enumerated in order to track how much of the
/// stream has been used in each reservoir.
fn write_reservoir_visualizations_data_to_yaml(
    stream_size: usize,
    capacity: usize,
) -> std::io::Result<()> {
    // Streamline up error handling

    let num_initial_values = stream_size / 4;
    let num_final_values = 3 * stream_size / 4;
    let bin_size: f64 = 0.05;
    let mut parameters: HashMap<String, String> = HashMap::new();

    parameters.insert("stream_size".to_string(), stream_size.to_string());
    parameters.insert(
        "num_initial_values".to_string(),
        num_initial_values.to_string(),
    );
    parameters.insert("num_final_values".to_string(), num_final_values.to_string());
    parameters.insert("capacity".to_string(), capacity.to_string());
    parameters.insert("bin_size".to_string(), bin_size.to_string());

    parameters.insert(
        "stream_file".to_string(),
        "./target/debug/examples/stream_for_histogram.yaml".to_string(),
    );
    parameters.insert(
        "reservoir_samples_file".to_string(),
        "./target/debug/examples/reservoirs_for_histogram.yaml".to_string(),
    );
    parameters.insert(
        "parameters_file_path".to_string(),
        "./visualizations_python/parameters_for_histogram.yaml".to_string(),
    );
    parameters.insert(
        "reservoir_means_file".to_string(),
        "./target/debug/examples/reservoir_means.yaml".to_string(),
    );

    println!(
        "The test uses a stream of size {:#?} and a reservoir capacity of {:#?}.",
        stream_size, capacity
    );
    let sigma = 0.15f64;
    let mean_initial = 0.25f64;
    let mean_final = 0.75f64;
    parameters.insert("sigma".to_string(), sigma.to_string());

    // Generate the data as an Iterator
    let stream =
        utils::generate_stream_from_normal_distribution(num_initial_values, mean_initial, sigma);
    let mut stream_end =
        utils::generate_stream_from_normal_distribution(num_final_values, mean_final, sigma);
    let stream = stream.chain(&mut stream_end);

    // Convert the data to a StreamingIterator and adapt
    let stream = convert(stream);
    let stream = enumerate(stream);
    let stream = write_yaml_documents(stream, parameters["stream_file"].to_string())
        .expect("Create File and initialize yaml iter failed.");
    let stream = reservoir_iterable(stream, capacity, None);
    let stream = write_yaml_documents(stream, parameters["reservoir_samples_file"].to_string())
        .expect("Create File and initialize yaml iter failed.");

    // This closure converts items from reservoirs to Numbered{max_index, Some(reservoir_mean)}
    let reservoir_mean_and_max_index = |reservoir: &Vec<Numbered<f64>>| -> Numbered<f64> {
        let mean_and_max_index = reservoir.iter().scan(
            Numbered {
                count: 0,
                item: Some(0.),
            },
            |state, x| {
                state.count = cmp::max(state.count, x.count);
                if let Some(partial_sum) = (*state).item {
                    *state = Numbered {
                        count: state.count,
                        item: Some(partial_sum + x.item.unwrap()),
                    }
                }
                Some(state.clone())
            },
        );
        let result = &mean_and_max_index.last();
        let mean = result.as_ref().unwrap().item.unwrap() / (capacity as f64);
        let max_index = result.as_ref().unwrap().count;
        Numbered {
            count: max_index,
            item: Some(mean),
        }
    };

    let stream = stream.map(reservoir_mean_and_max_index);
    let mut stream = write_yaml_documents(stream, parameters["reservoir_means_file"].to_string())
        .expect("Create File and initialize yaml iter failed.");
    // num_res is used in the python script for visualizations to initialize the size of the array that will hold that data to visualize.
    let mut num_res = 0i64;
    while let Some(_item) = stream.next() {
        num_res += 1;
    }
    parameters.insert("num_res".to_string(), num_res.to_string());
    let param_file_path = parameters["parameters_file_path"].clone();
    utils::write_parameters_to_yaml(parameters, &param_file_path)?;
    Ok(())
}

/// Call a Python script to make a visualization.
fn make_visualization_in_python(
    path_to_script: &str,
    visualization_description: &str,
) -> std::io::Result<()> {
    let output = Command::new("python3").arg(path_to_script).output()?;
    if !output.status.success() {
        println!(
            "\n\n----------Running {:?} did not succeed.----------\n",
            path_to_script
        );
        std::io::stderr().write_all(&output.stdout).unwrap();
    } else {
        println!("{:?} exported successfully.", visualization_description);
    };
    Ok(())
}

/// A utility fn to remove the yaml files full of the data generated.
fn remove_yaml_files() -> Result<(), std::io::Error> {
    let file_list = vec![
        "./target/debug/examples/stream_for_histogram.yaml",
        "./target/debug/examples/reservoirs_for_histogram.yaml",
        "./visualizations_python/parameters_for_histogram.yaml",
        "./target/debug/examples/reservoir_means.yaml",
    ];
    for file in file_list.iter() {
        if let Ok(_) = fs::metadata(file) {
            Command::new("rm").arg(file).output()?;
        }
    }
    Ok(())
}

fn set_visualization_parameters() -> (bool, usize, usize) {
    let args: Vec<String> = env::args().collect();
    let mut visualize: bool = true;
    if args.len() > 1 {
        visualize = bool::from_str(&args[1])
            .expect("Invalid argument. Please use with 'true', 'false', or no argument.");
    }
    let mut stream_size: usize = 5 * 10_i32.pow(03) as usize;
    let mut capacity: usize = 100;
    if !visualize {
        stream_size = 12;
        capacity = 2;
    }
    (visualize, stream_size, capacity)
}

fn main() -> std::io::Result<()> {
    let (visualize, stream_size, capacity) = set_visualization_parameters();
    remove_yaml_files()?;
    write_reservoir_visualizations_data_to_yaml(stream_size, capacity)?;
    println!("Data is written to yaml files.");
    if visualize {
        make_visualization_in_python(
            "./visualizations_python/reservoir_histograms_initial_final.py",
            "Initial and Final Histograms",
        )?;
        make_visualization_in_python(
            "./visualizations_python/reservoir_means.py",
            "Reservoir and stream means",
        )?;
        make_visualization_in_python(
            "./visualizations_python/reservoir_histogram_animation.py",
            "Animation of reservoir and stream histograms",
        )?;
    } else {
        println!(
            "The following .yaml files have been created:\n
            ./target/debug/examples/population_for_histogram.yaml \n  
            ./target/debug/examples/reservoirs_for_histogram.yaml \n
            ./target/debug/examples/reservoir_means.yaml \n 
            ./target/debug/examples/stream_for_histogram.yaml \n
            "
        );
    }
    Ok(())
}

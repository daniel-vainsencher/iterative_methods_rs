use crate::utils;
use iterative_methods::*;
use std::cmp;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::process::Command;
use streaming_iterator::*;
use std::env;
use std::str::FromStr;  

/// Write the full stream and a sequence of reservoir samples
/// to yaml files. The stream
/// is enumerated in order to track how much of the
/// stream has been used in each reservoir.
fn reservoir_visualizations(file_list: Vec<String>, stream_size: usize, capacity: usize) -> std::io::Result<()> {
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

    for file in file_list {
        parameters.insert(file.clone(), file);
    }

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
    let stream = write_yaml_documents(stream, parameters["stream_file"].to_string())
        .expect("Create File and initialize yaml iter failed.");
    let stream = reservoir_iterable(stream, capacity, None);
    // let stream = step_by(stream, 20);
    let stream = write_yaml_documents(stream, parameters["reservoir_samples_file"].to_string())
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

/// Call a Python script to make histograms of the initial and final stream.
fn make_initial_final_histograms_in_python() -> std::io::Result<()> {
    let output = Command::new("python3")
        .arg("./visualizations_python/reservoir_histograms_initial_final.py")
        .output()?;
    if !output.status.success() {
        println!(
            "\n\n *****Running reservoir_histograms_initial_final.py did not succeed.*****\n\n"
        );
        std::io::stdout().write_all(&output.stdout).unwrap();
    } else {
        println!("Initial and Final Histograms exported successfully.");
    };
    Ok(())
}

/// Call a Python script to plot stream means vs reservoir means.
fn make_reservoir_means_plot_in_python() -> std::io::Result<()> {
    let output = Command::new("python3")
        .arg("./visualizations_python/reservoir_means.py")
        .output()?;
    if !output.status.success() {
        println!("\n\n *****Running reservoir_means.py did not succeed.*****\n\n");
        std::io::stdout().write_all(&output.stdout).unwrap();
    } else {
        println!("Reservoir Means Plot exported successfully.");
    };
    Ok(())
}

/// Call a Python script to create an animation showing the stream histogram and reservoir histogram
/// as the stream is processed.
fn make_animations_in_python() -> std::io::Result<()> {
    let output = Command::new("python3")
        .arg("./visualizations_python/reservoir_histogram_animation.py")
        .output()?;
    if !output.status.success() {
        println!("\n\n *****Running reservoir_histogram_animation.py did not succeed.*****\n\n");
        std::io::stdout().write_all(&output.stdout).unwrap();
    } else {
        println!("Animation exported successfully.");
    };
    Ok(())
}

/// A utility fn to remove the yaml files full of the data generated.
fn remove_yaml_files() -> Result<Vec<String>, std::io::Error> {
    // Define file paths for yaml data
    let stream_file = "./target/debug/examples/stream_for_histogram.yaml";
    let reservoir_samples_file = "./target/debug/examples/reservoirs_for_histogram.yaml";
    let parameters_file_path = "./visualizations_python/parameters_for_histogram.yaml";
    let reservoir_means_file = "./target/debug/examples/reservoir_means.yaml";

    let file_list = vec![
        stream_file.to_string(),
        reservoir_samples_file.to_string(),
        parameters_file_path.to_string(),
        reservoir_means_file.to_string(),
    ];
    for file in file_list.iter() {
        if let Ok(_) = fs::metadata(file) {
            Command::new("rm").arg(file).output()?;
        }
    }
    Ok(file_list)
}

fn make_visualization() -> (bool, usize, usize) {
    let args: Vec<String> = env::args().collect();
    let mut visualize: bool = true;
    if args.len() > 1 {
        visualize = bool::from_str(&args[1]).expect("Invalid argument. Please use with 'true', 'false', or no argument.");
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
    let (visualize, stream_size, capacity) = make_visualization();
    let file_list = remove_yaml_files()?;
    reservoir_visualizations(file_list, stream_size, capacity)?;
    println!("Data is written to yaml files.");
    if visualize {
        make_initial_final_histograms_in_python()?;
        make_reservoir_means_plot_in_python()?;
        make_animations_in_python()?;
    } else {
        println!("The following .yaml files have been created:\n
            ./target/debug/examples/population_for_histogram.yaml \n  
            ./target/debug/examples/reservoirs_for_histogram.yaml \n
            ./target/debug/examples/reservoir_means.yaml \n 
            ./target/debug/examples/stream_for_histogram.yaml \n
            ");
    }
    Ok(())
    
}

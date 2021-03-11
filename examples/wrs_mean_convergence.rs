use crate::utils;
use iterative_methods::*;
use rand::distributions::Uniform;
use rand::Rng;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::process::Command;
use streaming_iterator::*;

/// Generates a random sample from the uniform distribution on (0,1) returned as a Vec.
/// Utility function to test the moments of a WRS.
fn uniform_stream_as_vec(num: usize) -> Vec<f64> {
    let range = Uniform::from(0.0..1.0);
    let stream_vec: Vec<f64> = rand::thread_rng().sample_iter(&range).take(num).collect();
    stream_vec
}

fn write_population_to_file<T>(a_stream_vec: &Vec<T>, file_name: &str) -> std::io::Result<()>
where
    T: std::fmt::Display,
{
    let mut file = File::create(file_name)?;
    for val in a_stream_vec {
        let mut val = val.to_string();
        val = ["-", &val, "\n"].join(" ");
        file.write_all(val.as_bytes())?;
    }
    file.flush()?;
    Ok(())
}

fn clear_file(file_name: &str) -> std::io::Result<()> {
    let file = File::create(file_name)?;
    file.set_len(0)?;
    Ok(())
}

fn write_res_to_file<T>(res_vec_values: &Vec<T>, file_name: &str) -> std::io::Result<()>
where
    T: std::fmt::Display,
{
    let mut file = OpenOptions::new().append(true).create(true).open(file_name);
    if let Ok(ref mut f) = file {
        for val in res_vec_values {
            let mut val = val.to_string();
            val = ["-", &val, "\n"].join(" ");
            f.write_all(val.as_bytes())?;
        }
        f.write_all(b"--- # new reservoir \n")?;
        f.flush()?;
    };
    Ok(())
}

/// Create a stream of data, write the full stream and a
/// sequence of reservoir samples to yaml files,
/// return the stream_size, capacity, and step size used.
fn wrs_mean_convergence_for_step() -> (usize, usize, usize) {
    // Streamline up error handling
    let stream_size: usize = 10_i32.pow(5) as usize;
    let capacity: usize = 100;
    let step: usize = 1;
    println!(
        "The test uses a stream of size {:#?} and a reservoir capacity of {:#?}.",
        stream_size, capacity
    );
    let population_file = "./target/debug/examples/population.yaml";
    // Create the stream:
    let stream = utils::generate_step_stream(stream_size, capacity, 1.0, 0, 1);
    let stream = convert(stream);
    // Create another copy of the stream to be turned into a vec:
    let mut stream_for_vec = utils::generate_step_stream(stream_size, capacity, 1.0, 0, 1);
    let mut stream_vec: Vec<usize> = Vec::new();
    while let Some(wd) = stream_for_vec.next() {
        stream_vec.push(wd.value)
    }
    // Clear contents of the file.
    if let Err(error) = clear_file(&population_file) {
        println!("{:#?}", error);
    };
    // Write to file for visualization
    if let Err(error) = write_population_to_file(&stream_vec, &population_file) {
        println!("{:#?}", error);
    };

    // Produce reservoir samples and write to file
    // let stream = convert(stream_vec);
    // let wd_stream = wd_iterable(stream, |_x| 1.);
    let wrs_iter = reservoir_iterable(stream, capacity, None);
    let mut wrs_iter = step_by(wrs_iter, step);

    let reservoir_samples_file = "./target/debug/examples/reservoirs.yaml";
    // Clear conents of the file.
    if let Err(error) = clear_file(&reservoir_samples_file) {
        println!("{:#?}", error);
    };
    // Write data to file for visualization.
    while let Some(res) = wrs_iter.next() {
        let mut res_values: Vec<usize> = Vec::new();
        for wd in res {
            res_values.push(wd.value);
        }
        if let Err(error) = write_res_to_file(&res_values, &reservoir_samples_file) {
            println!("{:#?}", error);
        }
    }

    (stream_size, capacity, step)
}

fn make_animations_in_python(
    stream_size: usize,
    capacity: usize,
    step: usize,
    num_bins: usize,
) -> std::io::Result<()> {
    let output = Command::new("python3")
        .arg("./visualizations_python/wrs_mean_convergence.py")
        .arg(stream_size.to_string())
        .arg(capacity.to_string())
        .arg(step.to_string())
        .arg(num_bins.to_string())
        .output()?;
    if !output.status.success() {
        println!(
            "Running wrs_mean_convergence.py did not succeed. Error: \n\n{:#?}\n\n",
            output
        );
    } else {
        println!("{:#?}", output);
    };
    Ok(())
}

fn main() -> std::io::Result<()> {
    let num_bins = 50usize;
    let (stream_size, capacity, step) = wrs_mean_convergence_for_step();
    make_animations_in_python(stream_size, capacity, step, num_bins)?;
    Ok(())
}

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

fn write_population_to_file(a_stream_vec: &Vec<f64>, file_name: &str) -> std::io::Result<()> {
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

fn write_res_to_file(res_vec_values: &Vec<f64>, file_name: &str) -> std::io::Result<()> {
    // let mut file = File::create(file_name)?;
    let mut file = OpenOptions::new().append(true).create(true).open(file_name);
    for val in res_vec_values {
        let mut val = val.to_string();
        val = ["-", &val, "\n"].join(" ");
        if let Ok(ref mut f) = file {
            f.write_all(val.as_bytes())?;
        };
    }
    if let Ok(mut f) = file {
        f.write_all(b"--- # new reservoir \n")?;
        f.flush()?;
    }

    Ok(())
}

fn wrs_animation() -> (usize, usize, usize) {
    // Streamline up error handling
    let stream_size: usize = 10_i32.pow(3) as usize;
    let capacity: usize = 100;
    let step: usize = 10;
    println!(
        "The test uses a stream of size {} and a reservoir capacity of {}.",
        stream_size, capacity
    );
    let population_file = "./target/debug/examples/population.yaml";
    // Create the data as a vec
    let stream_vec = uniform_stream_as_vec(stream_size);
    // Clear contents of the file.
    if let Err(error) = clear_file(&population_file) {
        println!("{:?}", error);
    };
    // Write to file for visualization
    if let Err(error) = write_population_to_file(&stream_vec, &population_file) {
        println!("{:?}", error);
    };

    // Produce reservoir samples and write to file
    let stream = convert(stream_vec);
    let wd_stream = wd_iterable(stream, |_x| 1.);
    let wrs_iter = reservoir_iterable(wd_stream, capacity, None);
    let mut wrs_iter = step_by(wrs_iter, step);

    let reservoir_samples_file = "./target/debug/examples/reservoirs.yaml";
    // Clear conents of the file.
    if let Err(error) = clear_file(&reservoir_samples_file) {
        println!("{:?}", error);
    };
    // Write data to file for visualization.
    while let Some(res) = wrs_iter.next() {
        let mut res_values: Vec<f64> = Vec::new();
        for wd in res {
            res_values.push(wd.value);
        }
        if let Err(error) = write_res_to_file(&res_values, &reservoir_samples_file) {
            println!("{:?}", error);
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
        .arg("./visualizations_python/wrs_animation.py")
        .arg(stream_size.to_string())
        .arg(capacity.to_string())
        .arg(step.to_string())
        .arg(num_bins.to_string())
        .output()?;
    if !output.status.success() {
        println!(
            "Running wrs_animation.py did not succeed. Error: \n\n{:#?}\n\n",
            output
        );
    } else {
        println!("{:#?}", output);
    };
    Ok(())
}

fn main() -> std::io::Result<()> {
    let num_bins = 5usize;
    let (stream_size, capacity, step) = wrs_animation();
    make_animations_in_python(stream_size, capacity, step, num_bins)?;
    Ok(())
}

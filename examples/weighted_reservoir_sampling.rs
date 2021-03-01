use iterative_methods::*;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use streaming_iterator::*;

/// Utility function to generate a sequence of (float, int as float)
/// values wrapped in a WeightedDatum struct that will be used in tests
/// of ReservoirIterable.
fn generate_seeded_values(num_values: usize, int_range_bound: usize) -> Vec<WeightedDatum<f64>> {
    let mut prng = Pcg64::seed_from_u64(1);
    let mut seeded_values: Vec<WeightedDatum<f64>> = Vec::new();
    for _i in 0..num_values {
        let afloat = prng.gen();
        let anint = prng.gen_range(0..int_range_bound) as f64;
        let wd: WeightedDatum<f64> = new_datum(afloat, anint);
        seeded_values.push(wd);
    }
    seeded_values
}


fn wrs_demo() {
    let mut seeded_values = generate_seeded_values(6, 2);
    let mut stream: Vec<WeightedDatum<f64>> = Vec::new();
    for _i in 0..4 {
        if let Some(wd) = seeded_values.pop() {
            stream.push(wd);
        };
    }
    let probability_and_index = seeded_values;
    println!("Stream: \n {:#?} \n", stream);
    println!("Random Numbers for Alg: \n (The values are used as the probabilities and the weights as indices.) \n {:#?} \n ", probability_and_index);

    let stream = convert(stream);
    let mut stream = reservoir_iterable(stream, 2, Some(Pcg64::seed_from_u64(1)));
    println!("Reservoir - initially empty: \n {:#?} \n", stream.reservoir);
    let mut _index = 0usize;
    while let Some(reservoir) = stream.next() {
        if _index == 0 {
            println!(
                "Reservoir filled with the first items from the stream: {:#?} \n",
                reservoir
            );
        } else {
            println!("Reservoir: {:#?} \n", reservoir);
        }
        _index = _index + 1;
    }
}

fn main() {
    wrs_demo();
}

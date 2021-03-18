//! # iterative-methods
//! A demonstration of the use of StreamingIterators and their adapters to implement iterative algorithms.
#[cfg(test)]
extern crate quickcheck;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::cmp::PartialEq;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::string::String;
use std::time::{Duration, Instant};
use streaming_iterator::*;

pub mod algorithms;
pub mod utils;

/// Store a generic annotation next to the state.
#[derive(Clone)]
pub struct AnnotatedResult<T, A> {
    pub result: T,
    pub annotation: A,
}

/// An adaptor that annotates every underlying item `x` with `f(x)`.
pub struct AnnotatedIterable<I, T, F, A>
where
    I: Sized + StreamingIterator<Item = T>,
    T: Clone,
    F: FnMut(&T) -> A,
{
    pub it: I,
    pub f: F,
    pub current: Option<AnnotatedResult<T, A>>,
}

impl<I, T, F, A> AnnotatedIterable<I, T, F, A>
where
    I: StreamingIterator<Item = T>,
    T: Sized + Clone,
    F: FnMut(&T) -> A,
{
    /// Annotate every underlying item with the result of applying `f` to it.
    fn new(it: I, f: F) -> AnnotatedIterable<I, T, F, A> {
        AnnotatedIterable {
            it,
            f: f,
            current: None,
        }
    }
}

impl<I, T, F, A> StreamingIterator for AnnotatedIterable<I, T, F, A>
where
    I: StreamingIterator<Item = T>,
    T: Sized + Clone,
    F: FnMut(&T) -> A,
{
    type Item = AnnotatedResult<T, A>;

    fn advance(&mut self) {
        self.it.advance();
        self.current = match self.it.get() {
            Some(n) => {
                let annotation = (self.f)(n);
                Some(AnnotatedResult {
                    annotation,
                    result: n.clone(),
                })
            }
            None => None,
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        match &self.current {
            Some(tr) => Some(&tr),
            None => None,
        }
    }
}

/// Annotate every underlying item with its score, as defined by `f`.
pub fn assess<I, T, F>(it: I, f: F) -> AnnotatedIterable<I, T, F, f64>
where
    T: Clone,
    F: FnMut(&T) -> f64,
    I: StreamingIterator<Item = T>,
{
    AnnotatedIterable::new(it, f)
}

/// Apply `f` to every underlying item.
pub fn tee<I, F, T>(it: I, f: F) -> AnnotatedIterable<I, T, F, ()>
where
    I: Sized + StreamingIterator<Item = T>,
    F: FnMut(&T),
    T: Clone,
{
    AnnotatedIterable::new(it, f)
}

/// Get the item before the first None, assuming any exist.
pub fn last<I, T>(it: I) -> Option<T>
where
    I: StreamingIterator<Item = T>,
    T: Sized + Clone,
{
    it.fold(None, |_acc, i| Some((*i).clone()))
}

/// Times every call to `advance` on the underlying
/// StreamingIterator. Stores both the time at which it starts, and
/// the duration it took to run.
pub struct TimedIterable<I, T>
where
    I: StreamingIterator<Item = T>,
    T: Clone,
{
    it: I,
    current: Option<TimedResult<T>>,
    timer: Instant,
}

/// TimedResult decorates with two duration fields: start_time is
/// relative to the creation of the process generating results, and
/// duration is relative to the start of the creation of the current
/// result.
#[derive(Clone)]
pub struct TimedResult<T> {
    pub result: T,
    pub start_time: Duration,
    pub duration: Duration,
}

/// Wrap each value of a streaming iterator with the durations:
/// - between the call to this function and start of the value's computation
/// - it took to calculate that value
pub fn time<I, T>(it: I) -> TimedIterable<I, T>
where
    I: Sized + StreamingIterator<Item = T>,
    T: Sized + Clone,
{
    TimedIterable {
        it: it,
        current: None,
        timer: Instant::now(),
    }
}

impl<I, T> StreamingIterator for TimedIterable<I, T>
where
    I: Sized + StreamingIterator<Item = T>,
    T: Sized + Clone,
{
    type Item = TimedResult<T>;

    fn advance(&mut self) {
        let start_time = self.timer.elapsed();
        let before = Instant::now();
        self.it.advance();
        self.current = match self.it.get() {
            Some(n) => Some(TimedResult {
                start_time,
                duration: before.elapsed(),
                result: n.clone(),
            }),
            None => None,
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        match &self.current {
            Some(tr) => Some(&tr),
            None => None,
        }
    }
}

/// Adapt StreamingIterator to only return values every 'step' number of times.
///
/// This is a StreamingIterator version of Iterator::step_by
///(https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.step_by)
///
/// The iterator adaptor step_by(it, step) wraps a StreamingIterator. A
/// 'step' is specified and only the items located every 'step' are returned.
///
///Iterator indices begin at 0, thus step_by() converts step -> step - 1
pub struct StepBy<I> {
    it: I,
    step: usize,
    first_take: bool,
}

pub fn step_by<I, T>(it: I, step: usize) -> StepBy<I>
where
    I: Sized + StreamingIterator<Item = T>,
{
    assert!(step != 0);
    StepBy {
        it,
        step: step - 1,
        first_take: true,
    }
}

impl<I> StreamingIterator for StepBy<I>
where
    I: StreamingIterator,
{
    type Item = I::Item;

    #[inline]
    fn advance(&mut self) {
        if self.first_take {
            self.first_take = false;
            self.it.advance();
        } else {
            self.it.nth(self.step);
        }
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        self.it.get()
    }
}

/// Write scalar items of StreamingIterator to a list in a file.
///
/// Add error handling!
pub struct ToFileIterable<I, F> {
    it: I,
    write_function: F,
    file_writer: File,
}

pub fn list_to_file<I, T, F>(
    it: I,
    write_function: F,
    file_path: String,
) -> Result<ToFileIterable<I, F>, std::io::Error>
where
    I: Sized + StreamingIterator<Item = T>,
    T: std::fmt::Debug,
    F: FnMut(&T, &mut std::fs::File) -> std::io::Result<()>,
{
    let result = match std::fs::metadata(&file_path) {
        Ok(_) => {
            panic!("File to which you want to write already exists or permission does not exist.")
        }
        Err(_) => {
            let file_writer = OpenOptions::new()
                .append(true)
                .create(true)
                .open(file_path)?;
            Ok(ToFileIterable {
                it,
                write_function,
                file_writer,
            })
        }
    };
    result
}

/// Function used by ToFileIterable to specify how to write each item: scalar to file.
///
pub fn write_scalar_to_list<T>(item: &T, file_writer: &mut std::fs::File) -> std::io::Result<()>
where
    T: std::string::ToString + std::fmt::Debug,
{
    let item = item.to_string();
    let val = ["-", &item, "\n"].join(" ");
    file_writer
        .write_all(val.as_bytes())
        .expect("Writing value to file failed.");
    Ok(())
}

/// Function used by ToFileIterable to specify how to write each item: Vec to file.
///
/// Each item: Vec is separated into its own document so that opening the file as a yaml file
/// produces an iterator in which each item is a vec.
pub fn write_vec_to_list<T>(item: &Vec<T>, file_writer: &mut std::fs::File) -> std::io::Result<()>
where
    T: std::string::ToString + std::fmt::Debug,
{
    for val in item.iter() {
        let mut val = val.to_string();
        val = ["-", &val, "\n"].join(" ");
        file_writer
            .write_all(val.as_bytes())
            .expect("Writing value to file failed.");
    }
    file_writer
        .write_all(b"--- # new reservoir \n")
        .expect("Writing New Document line failed.");
    Ok(())
}

impl<I, T, F> StreamingIterator for ToFileIterable<I, F>
where
    I: Sized + StreamingIterator<Item = T>,
    T: std::fmt::Debug,
    F: FnMut(&T, &mut std::fs::File) -> std::io::Result<()>,
{
    type Item = I::Item;

    #[inline]
    fn advance(&mut self) {
        if let Some(item) = self.it.next() {
            (self.write_function)(&item, &mut self.file_writer)
                .expect("Write item to file in ToFileIterable advance failed.");
        } else {
            self.file_writer.flush().expect("Flush of file failed.");
        }
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        self.it.get()
    }
}

/// An optimal reservoir sampling algorithm is implemented.
/// `ReservoirIterable` wraps a `StreamingIterator`, `I` and
/// produces a `StreamingIterator` whose items are samples of size `capacity`
/// from the stream of `I`. (This is not the capacity of the `Vec` which holds the `reservoir`;
/// Rather, the length of the `reservoir` is normally referred to as its `capacity`.)
/// To produce a `reservoir` of length `capacity` on the first call, the
/// first call of the `advance` method automatically advances the input
/// iterator `capacity` steps. Subsequent calls of `advance` on `ReservoirIterator`
/// advance `I` by `skip + 1` steps and will at most replace a single element of the `reservoir`.

/// The random rng is of type `Pcg64` by default, which allows seeded rng. This should be
/// extended to generic type bound by traits for implementing seeding.

/// See Algorithm L in https://en.wikipedia.org/wiki/Reservoir_sampling#An_optimal_algorithm and
/// https://dl.acm.org/doi/abs/10.1145/198429.198435

#[derive(Debug, Clone)]
pub struct ReservoirIterable<I, T> {
    it: I,
    pub reservoir: Vec<T>,
    capacity: usize,
    w: f64,
    skip: usize,
    rng: Pcg64,
}

// Create a ReservoirIterable
pub fn reservoir_iterable<I, T>(
    it: I,
    capacity: usize,
    custom_rng: Option<Pcg64>,
) -> ReservoirIterable<I, T>
where
    I: Sized + StreamingIterator<Item = T>,
    T: Clone,
{
    let mut rng = match custom_rng {
        Some(rng) => rng,
        None => Pcg64::from_entropy(),
    };
    let res: Vec<T> = Vec::new();
    let w_initial = (rng.gen::<f64>().ln() / (capacity as f64)).exp();
    ReservoirIterable {
        it,
        reservoir: res,
        capacity: capacity,
        w: w_initial,
        skip: ((rng.gen::<f64>() as f64).ln() / (1. - w_initial).ln()).floor() as usize,
        rng: rng,
    }
}

impl<I, T> StreamingIterator for ReservoirIterable<I, T>
where
    T: Clone + std::fmt::Debug,
    I: StreamingIterator<Item = T>,
{
    type Item = Vec<T>;

    #[inline]
    fn advance(&mut self) {
        if self.reservoir.len() < self.capacity {
            while self.reservoir.len() < self.capacity {
                if let Some(datum) = self.it.next() {
                    let cloned_datum = datum.clone();
                    self.reservoir.push(cloned_datum);
                } else {
                    break;
                }
            }
        } else {
            if let Some(datum) = self.it.nth(self.skip) {
                let h = self.rng.gen_range(0..self.capacity) as usize;
                let datum_struct = datum.clone();
                self.reservoir[h] = datum_struct;
                self.w = self.w * (self.rng.gen::<f64>().ln() / (self.capacity as f64)).exp();
                self.skip =
                    ((self.rng.gen::<f64>() as f64).ln() / (1. - self.w).ln()).floor() as usize;
            }
        }
    }

    #[inline]
    fn get(&self) -> Option<&Self::Item> {
        if let Some(_wd) = &self.it.get() {
            Some(&self.reservoir)
        } else {
            None
        }
    }
}

/// Weighted Sampling
/// The WeightedDatum struct wraps the values of a data set to include
/// a weight for each datum. Currently, the main motivation for this
/// is to use it for Weighted Reservoir Sampling.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedDatum<U> {
    value: U,
    weight: f64,
}

pub fn new_datum<U>(value: U, weight: f64) -> WeightedDatum<U>
where
    U: Clone,
{
    if !weight.is_finite() {
        panic!("The weight is not finite and therefore cannot be used to compute the probability of inclusion in the reservoir.");
    }
    WeightedDatum {
        value: value,
        weight: weight,
    }
}

/// WDIterable provides an easy conversion of any iterable to one whose items are WeightedDatum.
/// WDIterable holds an iterator and a function. The function is defined by the user to extract
/// weights from the iterable and package the old items and extracted weights into items as
/// WeightedDatum
#[derive(Debug, Clone)]
pub struct WDIterable<I, T, F>
where
    I: StreamingIterator<Item = T>,
{
    pub it: I,
    pub wd: Option<WeightedDatum<T>>,
    pub f: F,
}

pub fn wd_iterable<I, T, F>(it: I, f: F) -> WDIterable<I, T, F>
where
    I: StreamingIterator<Item = T>,
    F: FnMut(&T) -> f64,
{
    WDIterable {
        it: it,
        wd: None,
        f: f,
    }
}

impl<I, T, F> StreamingIterator for WDIterable<I, T, F>
where
    I: StreamingIterator<Item = T>,
    F: FnMut(&T) -> f64,
    T: Sized + Clone,
{
    type Item = WeightedDatum<T>;

    fn advance(&mut self) {
        self.it.advance();
        self.wd = match self.it.get() {
            Some(item) => {
                let new_weight = (self.f)(item);
                let new_item = item.clone();
                Some(new_datum(new_item, new_weight))
            }
            None => None,
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        match &self.wd {
            Some(wdatum) => Some(&wdatum),
            None => None,
        }
    }
}

/// ExtractValue converts items from WeightedDatum<T> to T.
pub struct ExtractValue<I, T>
where
    I: StreamingIterator<Item = WeightedDatum<T>>,
{
    it: I,
}

pub fn extract_value<I, T>(it: I) -> ExtractValue<I, T>
where
    I: StreamingIterator<Item = WeightedDatum<T>>,
{
    ExtractValue { it }
}

impl<I, T> StreamingIterator for ExtractValue<I, T>
where
    I: StreamingIterator<Item = WeightedDatum<T>>,
{
    type Item = T;
    fn advance(&mut self) {
        self.it.advance();
    }

    fn get(&self) -> Option<&Self::Item> {
        match &self.it.get() {
            Some(item) => Some(&item.value),
            None => None,
        }
    }
}

/// The weighted reservoir sampling algorithm of M. T. Chao is implemented.
/// `WeightedReservoirIterable` wraps a `StreamingIterator`, `I`, whose items must be of type `WeightedDatum` and
/// produces a `StreamingIterator` whose items are samples of size `capacity`
/// from the stream of `I`. (This is not the capacity of the `Vec` which holds the `reservoir`;
/// Rather, the length of the `reservoir` is normally referred to as its `capacity`.)
/// To produce a `reservoir` of length `capacity` on the first call, the
/// first call of the `advance` method automatically advances the input
/// iterator `capacity` steps. Subsequent calls of `advance` on `ReservoirIterator`
/// advance `I` one step and will at most replace a single element of the `reservoir`.

/// The random rng is of type `Pcg64` by default, which allows seeded rng. This should be
/// extended to generic type bound by traits for implementing seeding.

/// See https://en.wikipedia.org/wiki/Reservoir_sampling#Weighted_random_sampling,
/// https://arxiv.org/abs/1910.11069, or for the original paper,
/// https://doi.org/10.1093/biomet/69.3.653.

/// Future work might include implementing parallellized batch processing:
/// https://dl.acm.org/doi/10.1145/3350755.3400287
#[derive(Debug, Clone)]
pub struct WeightedReservoirIterable<I, T> {
    it: I,
    pub reservoir: Vec<WeightedDatum<T>>,
    capacity: usize,
    weight_sum: f64,
    rng: Pcg64,
}

/// Create a random sample of the underlying weighted stream.
pub fn weighted_reservoir_iterable<I, T>(
    it: I,
    capacity: usize,
    custom_rng: Option<Pcg64>,
) -> WeightedReservoirIterable<I, T>
where
    I: Sized + StreamingIterator<Item = WeightedDatum<T>>,
    T: Clone,
{
    let rng = match custom_rng {
        Some(rng) => rng,
        None => Pcg64::from_entropy(),
    };
    let res: Vec<WeightedDatum<T>> = Vec::new();
    WeightedReservoirIterable {
        it,
        reservoir: res,
        capacity: capacity,
        weight_sum: 0.0,
        rng: rng,
    }
}

impl<I, T> StreamingIterator for WeightedReservoirIterable<I, T>
where
    T: Clone + std::fmt::Debug,
    I: StreamingIterator<Item = WeightedDatum<T>>,
{
    type Item = Vec<WeightedDatum<T>>;

    #[inline]
    fn advance(&mut self) {
        if self.reservoir.len() >= self.capacity {
            if let Some(datum) = self.it.next() {
                self.weight_sum += datum.weight;
                let p = &(datum.weight / self.weight_sum);
                let j: f64 = self.rng.gen();
                if j < *p {
                    let h = self.rng.gen_range(0..self.capacity) as usize;
                    let datum_struct = datum.clone();
                    self.reservoir[h] = datum_struct;
                };
            }
        } else {
            while self.reservoir.len() < self.capacity {
                if let Some(datum) = self.it.next() {
                    let cloned_datum = datum.clone();
                    self.reservoir.push(cloned_datum);
                    self.weight_sum += datum.weight;
                } else {
                    break;
                }
            }
        }
    }

    #[inline]
    fn get(&self) -> Option<&Self::Item> {
        if let Some(_wd) = &self.it.get() {
            Some(&self.reservoir)
        } else {
            None
        }
    }
}

/// A simple Counter iterator to use in demos and tests.
#[derive(Clone, Debug)]
pub struct Counter {
    count: f64,
}

pub fn new_counter() -> Counter {
    Counter { count: 0. }
}

impl StreamingIterator for Counter {
    type Item = f64;

    fn advance(&mut self) {
        self.count += 1.;
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(&self.count)
    }
}

/// Unit Tests Module
#[cfg(test)]
mod tests {

    use super::*;
    use crate::utils::generate_stream_with_constant_probability;
    use std::io::Read;
    use std::iter;

    #[test]
    fn test_last() {
        let v = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let iter = convert(v.clone());
        assert!(last(iter) == Some(9));
    }

    #[test]
    fn test_last_none() {
        let v: Vec<u32> = vec![];
        assert!(last(convert(v.clone())) == None);
    }

    #[test]
    fn step_by_test() {
        let v = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let iter = convert(v);
        let mut iter = step_by(iter, 3);
        let mut _index = 0i64;
        while let Some(element) = iter.next() {
            assert_eq!(*element, _index * 3);
            _index = _index + 1;
        }
    }

    #[test]
    fn annotate_test() {
        let v = vec![0., 1., 2.];
        let iter = convert(v);
        fn f(num: &f64) -> f64 {
            num * 2.
        }
        let target_annotations = vec![0., 2., 4.];
        let mut annotations: Vec<f64> = Vec::with_capacity(3);
        let mut ann_iter = AnnotatedIterable::new(iter, f);
        while let Some(n) = ann_iter.next() {
            annotations.push(n.annotation);
        }
        assert_eq!(annotations, target_annotations);
    }

    /// ToFileIterable Test: Write list to yaml
    ///
    /// This writes a vec to a yaml file using ToFileIterable iterable.
    /// It should be improved by asserting that the contents of the file are correct.
    #[test]
    fn list_to_file_test() {
        let test_file_path = "./list_to_file_test.yaml";
        let v = vec![0, 1, 2, 3];
        // println!("The vec to be iterated: {:#?}", v);
        let v_iter = convert(v);
        let mut file_list: Vec<usize> = Vec::with_capacity(4);
        let mut yaml_iter =
            list_to_file(v_iter, write_scalar_to_list, String::from(test_file_path))
                .expect("Create File and initialize yaml_iter failed.");
        while let Some(_) = yaml_iter.next() {}
        let mut read_file =
            File::open("./list_to_file_test.yaml").expect("Could not open file with test data.");
        let mut contents = String::new();
        read_file
            .read_to_string(&mut contents)
            .expect("Could not read contents of file.");
        // Improve this to use a yaml reader to obtain a vec or list etc.
        assert_eq!("- 0 \n- 1 \n- 2 \n- 3 \n", &contents);
        std::fs::remove_file(test_file_path).expect("Could not remove data file for test.");
    }

    /// ToFileIterable Test: Write reservoirs (Vecs) to yaml
    ///
    /// This writes a sequence of reservoirs (Vecs) to a yaml file using ToFileIterable iterable.
    /// It should be improved by asserting that the contents of the file are correct.
    #[test]
    fn reservoirs_to_yaml_test() {
        let test_file_path = "./reservoir_to_yaml_test.yaml";
        let capacity = 2;
        let stream = utils::generate_step_stream(100, capacity, 0, 1);
        let stream = convert(stream);
        let res_iter = reservoir_iterable(stream, capacity, None);
        let mut yaml_iter = list_to_file(res_iter, write_vec_to_list, String::from(test_file_path))
            .expect("Create File and initialize yaml_iter failed.");
        while let Some(_item) = yaml_iter.next() {}
        std::fs::remove_file(test_file_path).expect("Could not remove data file for test.");
    }

    /// Tests for the ReservoirIterable adaptor
    ///
    /// This test asserts that the reservoir is filled with the correct items.
    #[test]
    fn fill_reservoir_test() {
        // v is the data stream.
        let v: Vec<f64> = vec![0.5, 0.2];
        let iter = convert(v);
        let mut iter = reservoir_iterable(iter, 2, None);
        if let Some(reservoir) = iter.next() {
            assert_eq!(reservoir[0], 0.5);
            assert_eq!(reservoir[1], 0.2);
        }
    }

    #[test]
    /// Test that the initial reservoir is eventually completely replaced.
    /// This needs to be improved so that the parameters values are chosen such that
    /// the test passes at a specified success rate.
    fn reservoir_replacement_test() {
        let stream_length = 500usize;
        // reservoir capacity:
        let capacity = 5usize;
        // Generate a stream that with items initially 0 and then 1:
        let initial_stream = iter::repeat(0).take(capacity);
        let final_stream = iter::repeat(1).take(stream_length - capacity);
        let stream = initial_stream.chain(final_stream);
        let stream = convert(stream);
        let mut res_iter = reservoir_iterable(stream, capacity, None);
        if let Some(reservoir) = res_iter.next() {
            println!("Initial reservoir: \n {:#?} \n", reservoir);
            assert!(reservoir.into_iter().all(|x| *x == 0));
        } else {
            panic!("The initial reservoir was None.");
        };

        let mut final_reservoir: Vec<usize> = vec![0, 0, 0, 0, 0];
        let mut count: usize = 0;
        while let Some(reservoir) = res_iter.next() {
            count += 1;
            final_reservoir = reservoir.to_vec();
        }
        println!(
            "Final reservoir after {:?} iterations: \n {:#?} \n ",
            count, final_reservoir
        );
        assert!(final_reservoir.into_iter().all(|x| x == 1));
    }

    /// Tests for the WeightedReservoirIterable adaptor
    #[test]
    fn test_datum_struct() {
        let samp = new_datum(String::from("hi"), 1.0);
        assert_eq!(samp.value, String::from("hi"));
        assert_eq!(samp.weight, 1.0);
    }

    #[test]
    #[should_panic(
        expected = "The weight is not finite and therefore cannot be used to compute the probability of inclusion in the reservoir."
    )]
    fn test_new_datum_infinite() {
        let _wd: WeightedDatum<String> = new_datum(String::from("some value"), f64::INFINITY);
    }

    /// This test asserts that the weighted reservoir is filled with the correct items.
    #[test]
    fn fill_weighted_reservoir_test() {
        // v is the data stream.
        let v: Vec<WeightedDatum<f64>> = vec![new_datum(0.5, 1.), new_datum(0.2, 2.)];
        let iter = convert(v);
        let mut iter = weighted_reservoir_iterable(iter, 2, None);
        if let Some(reservoir) = iter.next() {
            assert_eq!(
                reservoir[0],
                WeightedDatum {
                    value: 0.5f64,
                    weight: 1.0f64
                }
            );
            assert_eq!(
                reservoir[1],
                WeightedDatum {
                    value: 0.2f64,
                    weight: 2.0f64
                }
            );
        }
    }

    #[test]
    fn stream_smaller_than_weighted_reservoir_test() {
        let stream_vec = vec![new_datum(1, 1.0), new_datum(2, 1.0)];
        let stream = convert(stream_vec);
        let mut stream = weighted_reservoir_iterable(stream, 3, None);
        while let Some(_reservoir) = stream.next() {
            println!("{:#?}", _reservoir);
        }
    }

    #[test]
    fn test_constant_probability() {
        let stream_length = 10usize;
        // reservoir capacity:
        let capacity = 3usize;
        let probability = 0.01;
        let initial_weight = 1.0;
        // We create a stream with constant probability for all elements:
        let mut stream = generate_stream_with_constant_probability(
            stream_length,
            capacity,
            probability,
            initial_weight,
            0,
            1,
        );
        let mut weight_sum = initial_weight;
        // Cue the stream to the first "final value" element:
        stream.nth(capacity - 1);
        // Check that the probabilities are approximately correct.
        while let Some(item) = stream.next() {
            weight_sum += item.weight;
            let p = item.weight / weight_sum;
            assert!((p - probability).abs() < 0.01 * probability);
        }
    }

    #[test]
    #[should_panic(
        expected = "The weight is not finite and therefore cannot be used to compute the probability of inclusion in the reservoir."
    )]
    fn test_constant_probability_fail_from_inf_weight() {
        let stream_length = 100usize;
        // reservoir capacity:
        let capacity = 3usize;
        let probability = 0.9999;
        let initial_weight = 1.0;
        // We create a stream with constant probability for all elements:
        let mut stream = generate_stream_with_constant_probability(
            stream_length,
            capacity,
            probability,
            initial_weight,
            0,
            1,
        );
        while let Some(_item) = stream.next() {
            ()
        }
    }

    #[test]
    fn test_stream_vec_generator() {
        let stream_length = 50usize;
        // reservoir capacity:
        let capacity = 10usize;
        let probability = 0.01;
        let initial_weight = 1.0;
        // We create a stream with constant probability for all elements:
        let stream = generate_stream_with_constant_probability(
            stream_length,
            capacity,
            probability,
            initial_weight,
            0,
            1,
        );
        let mut stream = convert(stream);
        let mut _index: usize = 0;
        while let Some(item) = stream.next() {
            match _index {
                x if x < capacity => assert_eq!(
                    item.value, 0,
                    "Error: item value was {} for index={}",
                    item.value, x
                ),
                _ => assert_eq!(
                    item.value, 1,
                    "Error: item value was {} for index={}",
                    item.value, _index
                ),
            }
            _index = _index + 1;
        }
    }

    #[test]
    fn wrs_no_replacement_test() {
        let stream_length = 20usize;
        // reservoir capacity:
        let capacity = 10usize;
        let probability = 0.001;
        let initial_weight = 1.0;
        // We create a stream with constant probability for all elements:
        let stream = generate_stream_with_constant_probability(
            stream_length,
            capacity,
            probability,
            initial_weight,
            0,
            1,
        );
        let stream = convert(stream);
        let mut wrs_iter = weighted_reservoir_iterable(stream, capacity, None);
        if let Some(reservoir) = wrs_iter.next() {
            assert!(reservoir.into_iter().all(|wd| wd.value == 0));
        };

        if let Some(reservoir) = wrs_iter.nth(stream_length - capacity - 1) {
            assert!(reservoir.into_iter().all(|wd| wd.value == 0));
        } else {
            panic!("The final reservoir was None.");
        };
    }

    // Add link to derivation of bounds.
    /// This _probabilistic_ test asserts that all items of the initial
    /// reservoir will be replaced by the end of the streaming processes.
    /// It uses a stream in which the probability of each item being
    /// added to the reseroivr is close to 1. By using a large enough
    /// stream, we can ensure that the test fails very infrequently.
    /// The probability of the test failing is less than .001 if three
    /// conditions are met: 1) the
    /// reservoir capacity = 20, 2) the probability of each item being
    /// added is >=0.9, and 3) the length of the stream is >=460. A
    /// derivation of bounds that ensure a given level of success for
    /// the test can be found in the docs [LINK].
    // Consider wrapping the test in a for loop that runs the test 10^6 times
    // and counts the number of failures.
    #[test]
    fn wrs_complete_replacement_test() {
        let stream_length = 333usize;
        // reservoir capacity:
        let capacity = 15usize;
        let probability = 0.9;
        let initial_weight = 1.0e-20;
        // We create a stream whose probabilities are all 0.9:
        let stream = generate_stream_with_constant_probability(
            stream_length,
            capacity,
            probability,
            initial_weight,
            0,
            1,
        );
        let stream = convert(stream);
        let mut wrs_iter = weighted_reservoir_iterable(stream, capacity, None);
        if let Some(reservoir) = wrs_iter.next() {
            assert!(reservoir.into_iter().all(|wd| wd.value == 0));
        };

        if let Some(reservoir) = wrs_iter.nth(stream_length - capacity - 1) {
            assert!(reservoir.into_iter().all(|wd| wd.value == 1));
        } else {
            panic!("The final reservoir was None.");
        };
    }
}

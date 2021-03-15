//! # iterative-methods
//! A demonstration of the use of StreamingIterators and their adapters to implement iterative algorithms.
#[cfg(test)]
extern crate quickcheck;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::cmp::PartialEq;
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

pub struct AnnotatedIterable<I, T, F, A>
where
    I: Sized + StreamingIterator<Item = T>,
    T: Clone,
    F: FnMut(&T) -> A,
{
    pub it: I,
    pub f: F,
    pub last: Option<AnnotatedResult<T, A>>,
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
        self.last = match self.it.get() {
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
        match &self.last {
            Some(tr) => Some(&tr),
            None => None,
        }
    }
}

/// Apply a score function to every Item in the underlying iterable.
pub fn assess<I, T, F>(it: I, f: F) -> AnnotatedIterable<I, T, F, f64>
where
    T: Clone,
    F: FnMut(&T) -> f64,
    I: StreamingIterator<Item = T>,
{
    AnnotatedIterable {
        it,
        f: f,
        last: None,
    }
}

pub fn tee<I, F, T>(it: I, f: F) -> AnnotatedIterable<I, T, F, ()>
where
    I: Sized + StreamingIterator<Item = T>,
    F: FnMut(&T),
    T: Clone,
{
    AnnotatedIterable {
        it,
        f: f,
        last: None,
    }
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

pub fn last<I, T>(it: I) -> T
where
    I: StreamingIterator<Item = T>,
    T: Sized + Clone,
{
    let last_some = it.fold(None, |_acc, i| Some((*i).clone()));
    let last_item = last_some
        .expect("StreamingIterator last expects at least one non-None element.")
        .clone();
    last_item
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
        timer: Instant::now(),
        current: None,
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
/// `ReservoirIterable` wraps a `StreamingIterator`, `I`, whose items must be of type `WeightedDatum` and
/// produces a `StreamingIterator` whose items are samples of size `capacity`
/// from the stream of `I`. (This is not the capacity of the `Vec` which holds the `reservoir`;
/// Rather, the length of the `reservoir` is normally referred to as its `capacity`.)
/// To produce a `reservoir` of length `capacity` on the first call, the
/// first call of the `advance` method automatically advances the input
/// iterator `capacity` steps. Subsequent calls of `advance` on `ReservoirIterator`
/// advance `I` one step and will at most replace a single element of the `reservoir`.

/// The random oracle is of type `Pcg64` by default, which allows seeded rng. This should be
/// extended to generic type bound by traits for implementing seeding.

/// See https://en.wikipedia.org/wiki/Reservoir_sampling#Weighted_random_sampling,
/// https://arxiv.org/abs/1910.11069, or for the original paper,
/// https://doi.org/10.1093/biomet/69.3.653.

/// Future work might include implementing parallellized batch processing:
/// https://dl.acm.org/doi/10.1145/3350755.3400287
#[derive(Debug, Clone)]
pub struct ReservoirIterable<I, T> {
    it: I,
    pub reservoir: Vec<WeightedDatum<T>>,
    capacity: usize,
    weight_sum: f64,
    oracle: Pcg64,
}

// Create a ReservoirIterable
pub fn reservoir_iterable<I, T>(
    it: I,
    capacity: usize,
    custom_oracle: Option<Pcg64>,
) -> ReservoirIterable<I, T>
where
    I: Sized + StreamingIterator<Item = WeightedDatum<T>>,
    T: Clone,
{
    let oracle = match custom_oracle {
        Some(oracle) => oracle,
        None => Pcg64::from_entropy(),
    };
    let res: Vec<WeightedDatum<T>> = Vec::new();
    ReservoirIterable {
        it,
        reservoir: res,
        capacity: capacity,
        weight_sum: 0.0,
        oracle: oracle,
    }
}

impl<I, T> StreamingIterator for ReservoirIterable<I, T>
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
                let j: f64 = self.oracle.gen();
                if j < *p {
                    let h = self.oracle.gen_range(0..self.capacity) as usize;
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
    use std::iter;

    #[test]
    fn test_last() {
        let v = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let iter = convert(v.clone());
        assert!(last(iter) == 9);
    }

    #[test]
    #[should_panic(expected = "StreamingIterator last expects at least one non-None element.")]
    fn test_last_fail() {
        let v: Vec<u32> = vec![];
        last(convert(v.clone()));
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

    /// Tests for the ReservoirIterable adaptor
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

    /// This test asserts that the reservoir is filled with the correct items.
    #[test]
    fn fill_reservoir_test() {
        // v is the data stream.
        let v: Vec<WeightedDatum<f64>> = vec![new_datum(0.5, 1.), new_datum(0.2, 2.)];
        let iter = convert(v);
        let mut iter = reservoir_iterable(iter, 2, None);
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
    fn stream_smaller_than_reservoir_test() {
        let stream_vec = vec![new_datum(1, 1.0), new_datum(2, 1.0)];
        let stream = convert(stream_vec);
        let mut stream = reservoir_iterable(stream, 3, None);
        while let Some(_reservoir) = stream.next() {
            println!("{:#?}", _reservoir);
        }
    }

    /// utility function for testing ReservoirIterable
    fn generate_stream_with_constant_probability(
        stream_length: usize,
        capacity: usize,
        probability: f64,
        initial_weight: f64,
    ) -> impl Iterator<Item = WeightedDatum<&'static str>> {
        // Create capacity of items with initial weight.
        let initial_iter = iter::repeat(new_datum("initial value", initial_weight)).take(capacity);
        if capacity > stream_length {
            panic!("Capacity must be less than or equal to stream length.");
        }
        let final_iter =
            iter::repeat(new_datum("final value", initial_weight)).take(stream_length - capacity);
        let mut power = 0i32;
        let mapped = final_iter.map(move |wd| {
            power += 1;
            new_datum(
                wd.value,
                initial_weight * probability / (1.0 - probability).powi(power),
            )
        });
        let stream = initial_iter.chain(mapped);
        stream
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
        );
        let mut stream = convert(stream);
        let mut _index: usize = 0;
        while let Some(item) = stream.next() {
            match _index {
                x if x < capacity => assert_eq!(
                    item.value, "initial value",
                    "Error: item value was {} for index={}",
                    item.value, x
                ),
                _ => assert_eq!(
                    item.value, "final value",
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
        );
        let stream = convert(stream);
        let mut wrs_iter = reservoir_iterable(stream, capacity, None);
        if let Some(reservoir) = wrs_iter.next() {
            assert!(reservoir
                .into_iter()
                .all(|wd| wd.value == String::from("initial value")));
        };

        if let Some(reservoir) = wrs_iter.nth(stream_length - capacity - 1) {
            assert!(reservoir
                .into_iter()
                .all(|wd| wd.value == String::from("initial value")));
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
        );
        let stream = convert(stream);
        let mut wrs_iter = reservoir_iterable(stream, capacity, None);
        if let Some(reservoir) = wrs_iter.next() {
            assert!(reservoir
                .into_iter()
                .all(|wd| wd.value == String::from("initial value")));
        };

        if let Some(reservoir) = wrs_iter.nth(stream_length - capacity - 1) {
            assert!(reservoir
                .into_iter()
                .all(|wd| wd.value == String::from("final value")));
        } else {
            panic!("The final reservoir was None.");
        };
    }
}

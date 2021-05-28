//! # Iterative methods
//!
//! Implements [iterative
//! methods](https://en.wikipedia.org/wiki/Iterative_method) and
//! utilities for using and developing them as
//! [StreamingIterators](https://crates.io/crates/streaming-iterator). A
//! series of [blog
//! posts](https://daniel-vainsencher.github.io/book/iterative_methods_part_1.html)
//! provide a gentle introduction.
//!
//!
//! ... but ok fine, here is a really quick example:
//!```
//!// Problem: minimize the convex parabola f(x) = x^2 + x
//!let function = |x| x * x + x;
//!
//!// An iterative solution by gradient descent
//!let derivative = |x| 2.0 * x + 1.0;
//!let step_size = 0.2;
//!let x_0 = 2.0;
//!
//!// Au naturale:
//!let mut x = x_0;
//!for i in 0..10 {
//!    x -= step_size * derivative(x);
//!    println!("x_{} = {:.2}; f(x_{}) = {:.4}", i, x, i, x * x + x);
//!}
//!
//!// Using replaceable components:
//!let dd = DerivativeDescent::new(function, derivative, step_size, x_0);
//!let dd = enumerate(dd);
//!let mut dd = dd.take(10);
//!while let Some(&Numbered{item: Some(ref curr), count}) = dd.next() {
//!    println!("x_{} = {:.2}; f(x_{}) = {:.4}", count, curr.x, count, curr.value());
//!}
//!```
//! 
//! Both produce the exact same output (below), and the first common
//! approach is much easier to look at, the descent step is right
//! there. The second separates the algorithm and every other concern
//! into an easily reusable and composable components. If that sounds
//! useful, have fun exploring.
//!
//!```
//! x_0 = 1.00; f(x_0) = 2.0000
//! x_1 = 0.40; f(x_1) = 0.5600
//! x_2 = 0.04; f(x_2) = 0.0416
//! x_3 = -0.18; f(x_3) = -0.1450
//! x_4 = -0.31; f(x_4) = -0.2122
//! x_5 = -0.38; f(x_5) = -0.2364
//! x_6 = -0.43; f(x_6) = -0.2451
//! x_7 = -0.46; f(x_7) = -0.2482
//! x_8 = -0.47; f(x_8) = -0.2494
//! x_9 = -0.48; f(x_9) = -0.2498
//!```

#[cfg(test)]
extern crate quickcheck;
extern crate yaml_rust;

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use std::cmp::PartialEq;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{Duration, Instant};
use streaming_iterator::*;
use yaml_rust::{Yaml, YamlEmitter};

pub mod algorithms;
pub mod conjugate_gradient;
pub mod utils;

/// Creates an iterator which returns initial elements until and
/// including the first satisfying a predicate.
#[inline]
pub fn take_until<I, F>(it: I, f: F) -> TakeUntil<I, F>
where
    I: StreamingIterator,
    F: FnMut(&I::Item) -> bool,
{
    TakeUntil {
        it,
        f,
        state: UntilState::Unfulfilled,
    }
}

/// An adaptor that returns initial elements until and including the
/// first satisfying a predicate.
#[derive(Clone)]
pub struct TakeUntil<I, F>
where
    I: StreamingIterator,
    F: FnMut(&I::Item) -> bool,
{
    pub it: I,
    pub f: F,
    pub state: UntilState,
}

#[derive(Clone, PartialEq)]
pub enum UntilState {
    Unfulfilled,
    Fulfilled,
    Done,
}

impl<I, F> StreamingIterator for TakeUntil<I, F>
where
    I: StreamingIterator,
    F: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;
    fn advance(&mut self) {
        match self.state {
            UntilState::Unfulfilled => {
                self.it.advance();
                if let Some(v) = self.it.get() {
                    if (self.f)(v) {
                        self.state = UntilState::Fulfilled
                    }
                }
            }
            UntilState::Fulfilled => self.state = UntilState::Done,
            UntilState::Done => {}
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        if UntilState::Done == self.state {
            None
        } else {
            self.it.get()
        }
    }
}

/// Store a generic annotation next to the state.
#[derive(Clone, Debug)]
pub struct AnnotatedResult<T, A> {
    pub result: T,
    pub annotation: A,
}

/// An adaptor that annotates every underlying item `x` with `f(x)`.
#[derive(Clone, Debug)]
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
    pub fn new(it: I, f: F) -> AnnotatedIterable<I, T, F, A> {
        AnnotatedIterable {
            it,
            f,
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
pub fn assess<I, T, F, A>(it: I, f: F) -> AnnotatedIterable<I, T, F, A>
where
    T: Clone,
    F: FnMut(&T) -> A,
    I: StreamingIterator<Item = T>,
{
    AnnotatedIterable::new(it, f)
}

/// Apply `f(_)->()` to every underlying item (for side-effects).
pub fn inspect<I, F, T>(it: I, f: F) -> AnnotatedIterable<I, T, F, ()>
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
        it,
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

/// An iterator for stepping iterators by a custom amount.
///
/// This is a StreamingIterator version of [std::iter::StepBy.](https://doc.rust-lang.org/std/iter/struct.StepBy.html)
///
/// The iterator adaptor step_by(it, step) wraps a StreamingIterator. A
/// `step` is specified and only the items located every `step` are returned.
///
/// Iterator indices begin at 0, thus step_by() converts step -> step - 1
#[derive(Clone, Debug)]
pub struct StepBy<I> {
    it: I,
    step: usize,
    first_take: bool,
}

/// Creates an iterator starting at the same point, but stepping by the given amount at each iteration.
///
/// This is a `StreamingIterator` version of [step_by](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.step_by)
/// in [`std::iter::Iterator`](https://doc.rust-lang.org/std/iter/trait.Iterator.html)

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

/// Write items of StreamingIterator to a file.
#[derive(Debug)]
pub struct ToFileIterable<I, F> {
    pub it: I,
    pub write_function: F,
    pub file_writer: File,
}

/// An adaptor that writes each item to a new line of a file.
pub fn item_to_file<I, T, F>(
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
            panic!("File to which you want to write already exists or permission does not exist. Please rename or remove the file or gain permission.")
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

/// Define a trait object for converting to YAML objects.
pub trait YamlDataType {
    fn create_yaml_object(&self) -> Yaml;
}

/// Allow for references.
impl<T> YamlDataType for &T
where
    T: YamlDataType,
{
    fn create_yaml_object(&self) -> Yaml {
        (*self).create_yaml_object()
    }
}

/// Implement for basic scalar types.
impl YamlDataType for i64 {
    fn create_yaml_object(&self) -> Yaml {
        Yaml::Integer(*self)
    }
}

impl YamlDataType for f64 {
    fn create_yaml_object(&self) -> Yaml {
        Yaml::Real((*self).to_string())
    }
}

impl YamlDataType for String {
    fn create_yaml_object(&self) -> Yaml {
        Yaml::String((*self).to_string())
    }
}

// Does this clone cause memory or speed issues?
// This circular impl was necessary to allow impl YamlDataType for Vec<T> where T impl YamlDataType.
impl YamlDataType for Yaml {
    fn create_yaml_object(&self) -> Yaml {
        self.clone()
    }
}

/// This allows recursive wrapping of YamlDataType in Vec, e.g. Vec<Vec<Vec<T>>>.
impl<T> YamlDataType for Vec<T>
where
    T: YamlDataType,
{
    fn create_yaml_object(&self) -> Yaml {
        let v: Vec<Yaml> = self.iter().map(|x| x.create_yaml_object()).collect();
        Yaml::Array(v)
    }
}

impl<T, A> YamlDataType for AnnotatedResult<T, A>
where
    T: YamlDataType,
    A: YamlDataType,
{
    fn create_yaml_object(&self) -> Yaml {
        let t = &self.result;
        let a = &self.annotation;
        Yaml::Array(vec![t.create_yaml_object(), a.create_yaml_object()])
    }
}

impl<T> YamlDataType for WeightedDatum<T>
where
    T: YamlDataType,
{
    fn create_yaml_object(&self) -> Yaml {
        let value = &self.value;
        let weight = &self.weight;
        Yaml::Array(vec![
            value.create_yaml_object(),
            weight.create_yaml_object(),
        ])
    }
}

/// Write items of StreamingIterator to a Yaml file.
#[derive(Debug)]
pub struct ToYamlIterable<I> {
    pub it: I,
    pub file_writer: File,
}

/// Adaptor that writes each item to a YAML document.
pub fn write_yaml_documents<I, T>(
    it: I,
    file_path: String,
) -> Result<ToYamlIterable<I>, std::io::Error>
where
    I: Sized + StreamingIterator<Item = T>,
    T: std::fmt::Debug,
{
    let result = match std::fs::metadata(&file_path) {
        Ok(_) => {
            panic!("Failed to create or gain permission of {}, please delete it or gain permission before running this demo. If the demo runs completely, it will delete the file upon completion.", file_path)
        }
        Err(_) => {
            let file_writer = OpenOptions::new()
                .append(true)
                .create(true)
                .open(file_path)?;
            Ok(ToYamlIterable { it, file_writer })
        }
    };
    result
}

/// Function used by ToYamlIterable to specify how to write each item to file.
///
pub fn write_yaml_object<T>(item: &T, file_writer: &mut std::fs::File) -> std::io::Result<()>
where
    T: YamlDataType,
{
    let yaml_item = item.create_yaml_object();
    let mut out_str = String::new();
    let mut emitter = YamlEmitter::new(&mut out_str);
    emitter
        .dump(&yaml_item)
        .expect("Could not convert item to yaml object.");
    out_str.push('\n');
    file_writer
        .write_all(out_str.as_bytes())
        .expect("Writing value to file failed.");
    Ok(())
}

impl<I, T> StreamingIterator for ToYamlIterable<I>
where
    I: Sized + StreamingIterator<Item = T>,
    T: std::fmt::Debug + YamlDataType,
{
    type Item = I::Item;

    #[inline]
    fn advance(&mut self) {
        if let Some(item) = self.it.next() {
            (write_yaml_object)(&item, &mut self.file_writer)
                .expect("Write item to file in ToYamlIterable advance failed.");
        } else {
            self.file_writer.flush().expect("Flush of file failed.");
        }
    }

    #[inline]
    fn get(&self) -> Option<&I::Item> {
        self.it.get()
    }
}

/// A struct that wraps an `Item` as `Option<Item>` and annotates it with an `i64`. Used by `Enumerate`.
#[derive(Clone, Debug, std::cmp::PartialEq)]
pub struct Numbered<T> {
    pub count: i64,
    pub item: Option<T>,
}

impl<T> YamlDataType for Numbered<T>
where
    T: YamlDataType,
{
    fn create_yaml_object(&self) -> Yaml {
        let t = (self.item).as_ref().unwrap();
        Yaml::Array(vec![Yaml::Integer(self.count), t.create_yaml_object()])
    }
}

/// An adaptor that enumerates items.
#[derive(Clone, Debug)]
pub struct Enumerate<I, T> {
    pub current: Option<Numbered<T>>,
    pub it: I,
}

/// Define a constructor in the Enumerate context.
impl<I, T> Enumerate<I, T>
where
    I: StreamingIterator<Item = T>,
{
    pub fn new(it: I) -> Enumerate<I, T> {
        Enumerate {
            current: Some(Numbered {
                count: -1,
                item: None,
            }),
            it,
        }
    }
}

/// A constructor for Enumerate.
pub fn enumerate<I, T>(it: I) -> Enumerate<I, T>
where
    I: StreamingIterator<Item = T>,
{
    Enumerate {
        current: Some(Numbered {
            count: -1,
            item: None,
        }),
        it,
    }
}

impl<I, T> StreamingIterator for Enumerate<I, T>
where
    I: StreamingIterator<Item = T>,
    T: Clone,
{
    type Item = Numbered<T>;

    fn advance(&mut self) {
        self.it.advance();
        self.current = match self.it.get() {
            Some(t) => {
                if let Some(n) = &self.current {
                    let c = n.count + 1;
                    Some(Numbered {
                        count: c,
                        item: Some(t.clone()),
                    })
                } else {
                    None
                }
            }
            None => None,
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        match &self.current {
            Some(t) => Some(&t),
            None => None,
        }
    }
}

/// Adaptor to reservoir sample.
///
/// `ReservoirSample` wraps a `StreamingIterator`, `I` and
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
pub struct ReservoirSample<I, T> {
    it: I,
    pub reservoir: Vec<T>,
    capacity: usize,
    w: f64,
    skip: usize,
    rng: Pcg64,
}

/// An adaptor for which the items are random samples of the underlying iterator up to the item processed.
/// The constructor for ReservoirSample.
pub fn reservoir_sample<I, T>(
    it: I,
    capacity: usize,
    custom_rng: Option<Pcg64>,
) -> ReservoirSample<I, T>
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
    ReservoirSample {
        it,
        reservoir: res,
        capacity,
        w: w_initial,
        skip: ((rng.gen::<f64>() as f64).ln() / (1. - w_initial).ln()).floor() as usize,
        rng,
    }
}

impl<I, T> StreamingIterator for ReservoirSample<I, T>
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
        } else if let Some(datum) = self.it.nth(self.skip) {
            let h = self.rng.gen_range(0..self.capacity) as usize;
            let datum_struct = datum.clone();
            self.reservoir[h] = datum_struct;
            self.w *= (self.rng.gen::<f64>().ln() / (self.capacity as f64)).exp();
            self.skip = ((self.rng.gen::<f64>() as f64).ln() / (1. - self.w).ln()).floor() as usize;
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
/// is to use it for Weighted Reservoir Sampling (WRS).
///
/// WRS is currently deprecated, but WeightedDatum and WdIterable are not.
///
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedDatum<U> {
    pub value: U,
    pub weight: f64,
}

/// Constructor for WeightedDatum.
pub fn new_datum<U>(value: U, weight: f64) -> WeightedDatum<U>
where
    U: Clone,
{
    if !weight.is_finite() {
        panic!("The weight is not finite and therefore cannot be used to compute the probability of inclusion in the reservoir.");
    }
    WeightedDatum { value, weight }
}

/// Adaptor wrapping items with a computed weight.
///
/// WdIterable provides an easy conversion of any iterable to one whose items are WeightedDatum.
/// WdIterable holds an iterator and a function. The function is defined by the user to extract
/// weights from the iterable and package the old items and extracted weights into items as
/// WeightedDatum

#[derive(Debug, Clone)]
pub struct WdIterable<I, T, F>
where
    I: StreamingIterator<Item = T>,
{
    pub it: I,
    pub wd: Option<WeightedDatum<T>>,
    pub f: F,
}

/// Annotates items of an iterable with a weight using a function `f`.
pub fn wd_iterable<I, T, F>(it: I, f: F) -> WdIterable<I, T, F>
where
    I: StreamingIterator<Item = T>,
    F: FnMut(&T) -> f64,
{
    WdIterable { it, wd: None, f }
}

impl<I, T, F> StreamingIterator for WdIterable<I, T, F>
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

/// An adaptor that converts items from `WeightedDatum<T>` to `T`.
///
#[derive(Clone, Debug)]
pub struct ExtractValue<I, T>
where
    I: StreamingIterator<Item = WeightedDatum<T>>,
{
    it: I,
}

/// The constructor for ExtractValue. Apply it to a StreamingIterator with
/// `Item = WeightedDatum<T>` and it returns a StreamingIterator with `Item = T`.
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

/// Adaptor that reservoir samples with weights
///
/// Uses the algorithm of M. T. Chao.
/// `WeightedReservoirSample` wraps a `StreamingIterator`, `I`, whose items must be of type `WeightedDatum` and
/// produces a `StreamingIterator` whose items are samples of size `capacity`
/// from the stream of `I`. (This is not the capacity of the `Vec` which holds the `reservoir`;
/// Rather, the length of the `reservoir` is normally referred to as its `capacity`.)
/// To produce a `reservoir` of length `capacity` on the first call, the
/// first call of the `advance` method automatically advances the input
/// iterator `capacity` steps. Subsequent calls of `advance` on `ReservoirIterator`
/// advance `I` one step and will at most replace a single element of the `reservoir`.

/// The random rng is of type `Pcg64` by default, which allows seeded rng.

/// See https://en.wikipedia.org/wiki/Reservoir_sampling#Weighted_random_sampling,
/// https://arxiv.org/abs/1910.11069, or for the original paper,
/// https://doi.org/10.1093/biomet/69.3.653.

/// Future work might include implementing parallellized batch processing:
/// https://dl.acm.org/doi/10.1145/3350755.3400287

#[derive(Debug, Clone)]
pub struct WeightedReservoirSample<I, T> {
    it: I,
    pub reservoir: Vec<WeightedDatum<T>>,
    capacity: usize,
    weight_sum: f64,
    rng: Pcg64,
}

/// Create a random sample of the underlying weighted stream.
pub fn weighted_reservoir_sample<I, T>(
    it: I,
    capacity: usize,
    custom_rng: Option<Pcg64>,
) -> WeightedReservoirSample<I, T>
where
    I: Sized + StreamingIterator<Item = WeightedDatum<T>>,
    T: Clone,
{
    let rng = match custom_rng {
        Some(rng) => rng,
        None => Pcg64::from_entropy(),
    };
    let reservoir: Vec<WeightedDatum<T>> = Vec::new();
    WeightedReservoirSample {
        it,
        reservoir,
        capacity,
        weight_sum: 0.0,
        rng,
    }
}

impl<I, T> StreamingIterator for WeightedReservoirSample<I, T>
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
                let p = &(self.capacity as f64 * datum.weight / self.weight_sum);
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

/// Unit Tests Module
#[cfg(test)]
mod tests {

    use super::*;
    use crate::utils::generate_stream_with_constant_probability;
    use crate::utils::mean_of_means_of_step_stream;
    use std::convert::TryInto;
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

    /// ToYamlIterable Test: Write stream of scalars to yaml
    ///
    /// This writes a stream of scalars to a yaml file using ToYamlIterable iterable.
    /// It would fail if the file path used to write the data already existed
    /// due to the functionality of write_yaml_documents().
    #[test]
    fn write_yaml_documents_test() {
        let test_file_path = "./write_yaml_documents_test.yaml";
        let v: Vec<i64> = vec![0, 1, 2, 3];
        let v_iter = convert(v.clone());
        let mut yaml_iter = write_yaml_documents(v_iter, String::from(test_file_path))
            .expect("Create File and initialize yaml_iter failed.");
        while let Some(_) = yaml_iter.next() {}
        let mut read_file =
            File::open(test_file_path).expect("Could not open file with test data to asserteq.");
        let mut contents = String::new();
        read_file
            .read_to_string(&mut contents)
            .expect("Could not read data from file.");
        // The following line is to be used when the test is revised to read the contents of the file.
        // let docs = Yaml::from_str(&contents);
        // This could be used instead of Yaml::from_str; not sure of tradeoffs.
        // let docs = YamlLoader::load_from_str(&contents).expect("Could not load contents of file to yaml object.");
        // Remove the file for the next run of the test.
        std::fs::remove_file(test_file_path).expect("Could not remove data file for test.");
        assert_eq!("---\n0\n---\n1\n---\n2\n---\n3\n", &contents);
    }

    /// ToYamlIterable Test: Write stream of vecs to yaml
    ///
    /// This writes a stream of vecs to a yaml file using ToYamlIterable iterable.
    /// It would fail if the file path used to write the data already existed
    /// due to the functionality of item_to_file().
    #[test]
    fn write_vec_to_yaml_test() {
        let test_file_path = "./vec_to_file_test.yaml";
        let v: Vec<Vec<i64>> = vec![vec![0, 1], vec![2, 3]];
        // println!("{:#?}", v);
        let vc = v.clone();
        let vc = vc.iter();
        let vc = convert(vc);
        let mut vc = write_yaml_documents(vc, String::from(test_file_path))
            .expect("Vec to Yaml: Create File and initialize yaml_iter failed.");
        while let Some(_) = vc.next() {}
        let mut read_file =
            File::open(test_file_path).expect("Could not open file with test data to asserteq.");
        let mut contents = String::new();
        read_file
            .read_to_string(&mut contents)
            .expect("Could not read data from file.");
        std::fs::remove_file(test_file_path).expect("Could not remove data file for test.");
        assert_eq!("---\n- 0\n- 1\n---\n- 2\n- 3\n", &contents);
    }

    /// Test write_yaml_object works on AnnotatedResult
    /// This shows that that write_yaml_object works on a custom struct.
    #[test]
    fn annotated_result_to_yaml_test() {
        let ann = AnnotatedResult {
            result: 0,
            annotation: "zero".to_string(),
        };
        let test_file_path = "./annotated_result_test.yaml";
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(test_file_path)
            .expect("Could not open test file.");
        write_yaml_object(&ann, &mut file)
            .expect(&format!("write_yaml_object Failed for {}", test_file_path));
        let contents = utils::read_yaml_to_string(test_file_path)
            .expect(&format!("Could not read {}", test_file_path));
        assert_eq!("---\n- 0\n- zero\n", &contents);
    }

    /// Test that write_yaml_object works on Numbered.
    /// This shows that that write_yaml_object works on a custom struct.
    #[test]
    fn numbered_to_yaml_test() {
        let num = Numbered {
            count: 0,
            item: Some(0.1),
        };
        let test_file_path = "./numbered_test.yaml";
        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open(test_file_path)
            .expect("Could not open test file.");
        write_yaml_object(&num, &mut file).expect("write_yaml_object Failed.");
        let contents = utils::read_yaml_to_string(test_file_path).expect("Could not read file.");
        assert_eq!("---\n- 0\n- 0.1\n", &contents);
    }

    // Test that enumerate() adaptor produces items wrapped in a Numbered struct with the enumeration count.
    #[test]
    fn enumerate_test() {
        let v = vec![0, 1, 2];
        let stream = v.iter();
        let stream = convert(stream);
        let mut stream = enumerate(stream);
        let mut count = 0;
        while let Some(item) = stream.next() {
            println!("item: {:#?} \n count: {}\n\n", item, count);
            assert_eq!(
                *item,
                Numbered {
                    count: count,
                    item: Some(&count)
                }
            );
            count += 1;
        }
    }
    /// A stream of 2 items, each of type Vec<Vec<i64>>, is written to .yaml. The stream used is:
    /// ---
    /// - - 0
    ///   - 3
    /// - - 1
    ///   - 6
    /// - - 2
    ///   - 9
    /// ---
    /// - - 0
    ///   - 5
    /// - - 1
    ///   - 10
    /// - - 2
    ///   - 15
    #[test]
    fn write_vec_vec_to_yaml_test() {
        let test_file_path = "./vec_vec_to_file_test.yaml";
        let data_1: Vec<i64> = vec![3, 6, 9];
        let data_2: Vec<i64> = vec![5, 10, 15];
        let data_1 = data_1.iter().enumerate();
        let data_2 = data_2.iter().enumerate();
        let mut data_1_vec: Vec<Vec<i64>> = Vec::new();
        let mut data_2_vec: Vec<Vec<i64>> = Vec::new();
        for (a, b) in data_1 {
            data_1_vec.push(vec![a.try_into().unwrap(), *b])
        }
        for (a, b) in data_2 {
            data_2_vec.push(vec![a.try_into().unwrap(), *b])
        }
        let v: Vec<Vec<Vec<i64>>> = vec![data_1_vec, data_2_vec];
        let v = v.iter();
        let v = convert(v);
        let mut v = write_yaml_documents(v, String::from(test_file_path))
            .expect("Vec to Yaml: Create File and initialize yaml_iter failed.");
        while let Some(item) = v.next() {
            println!("{:#?}", item);
        }
        let mut read_file =
            File::open(test_file_path).expect("Could not open file with test data to asserteq.");
        let mut contents = String::new();
        read_file
            .read_to_string(&mut contents)
            .expect("Could not read data from file.");
        std::fs::remove_file(test_file_path).expect("Could not remove data file for test.");
        assert_eq!("---\n- - 0\n  - 3\n- - 1\n  - 6\n- - 2\n  - 9\n---\n- - 0\n  - 5\n- - 1\n  - 10\n- - 2\n  - 15\n", &contents);
    }

    /// Tests for the ReservoirSample adaptor
    ///
    /// This test asserts that the reservoir is filled with the correct items.
    #[test]
    fn fill_reservoir_test() {
        // v is the data stream.
        let v: Vec<f64> = vec![0.5, 0.2];
        let iter = convert(v);
        let mut iter = reservoir_sample(iter, 2, None);
        if let Some(reservoir) = iter.next() {
            assert_eq!(reservoir[0], 0.5);
            assert_eq!(reservoir[1], 0.2);
        }
    }

    #[test]
    /// Test that the initial reservoir of zeros is eventually filled with at least 4 ones.
    /// Running the test 10000 times resulted in 9997 tests passed. Thus the fail rate is significantly less
    /// than 1 in 1000. If the test fails more than 1 in 1000 times, then a change in the library has introduced a bug.
    fn reservoir_replacement_test() {
        let stream_length = 1000usize;
        // reservoir capacity:
        let capacity = 5usize;
        // Generate a stream with items initially 0 and then 1:
        let initial_stream = iter::repeat(0).take(capacity);
        let final_stream = iter::repeat(1).take(stream_length - capacity);
        let stream = initial_stream.chain(final_stream);
        let stream = convert(stream);
        let mut res_iter = reservoir_sample(stream, capacity, None);
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
        assert!(final_reservoir.into_iter().sum::<usize>() >= 4);
    }

    /// Tests for the WeightedReservoirSample adaptor
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
        let mut iter = weighted_reservoir_sample(iter, 2, None);
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
        let mut stream = weighted_reservoir_sample(stream, 3, None);
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
            let p = capacity as f64 * item.weight / weight_sum;
            assert!((p - probability).abs() < 0.01 * probability);
        }
    }

    #[test]
    #[should_panic(
        expected = "The weight is not finite and therefore cannot be used to compute the probability of inclusion in the reservoir."
    )]
    fn test_constant_probability_fail_from_inf_weight() {
        let stream_length: usize = 10_usize.pow(4);
        // reservoir capacity:
        let capacity = 3usize;
        let probability = 0.999999999;
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
        let mut wrs_iter = weighted_reservoir_sample(stream, capacity, None);
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
        let stream_length = 200usize;
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
        let mut wrs_iter = weighted_reservoir_sample(stream, capacity, None);
        if let Some(reservoir) = wrs_iter.next() {
            assert!(reservoir.into_iter().all(|wd| wd.value == 0));
        };

        if let Some(reservoir) = wrs_iter.nth(stream_length - capacity - 1) {
            assert!(reservoir.into_iter().all(|wd| wd.value == 1));
        } else {
            panic!("The final reservoir was None.");
        };
    }

    // For a stream of the form [(0,1),..,(0,1),(1,1),..,(1,1)] with equal numbers
    // of zero and one values and all weights equal to 1, we expect weighted reservoir
    // sampling to reduce to reservoir sampling (no weights) and thus to produce
    // a reservoir whose mean estimates the mean of the entire stream, which in this case
    // is 0.5. A reservoir sample is generated 50 times. Each time the mean is calculated
    // and the mean of these means is taken. It is asserted that this mean of means is
    // within 5% of the true mean, 0.5.
    //
    // In wrs_mean_test_looped(), the current test (wrs_mean_test()) was run 3000 times
    // with one failure resulting. Thus we estimate the failure rate to be approximately
    // 1 in 3000. If this test fails more than once for you, there is likely a problem.
    #[test]
    fn wrs_mean_test() {
        let mean_means = mean_of_means_of_step_stream();
        assert!((mean_means - 0.5).abs() < 0.05 * 0.5);
    }

    // This test is used to estimate the failure rate of wrs_mean_test (see above).
    // wrs_mean_test() is run 3000 times. In our experience this has led to one
    // failure. Thus we estimate the failure rate of wrs_mean_test() to be approximately
    // 1 in 3000.
    #[test]
    #[ignore]
    fn wrs_mean_test_looped() {
        let mut failures = 0usize;
        let number_of_runs = 3_000usize;
        for _j in 0..number_of_runs {
            let mean_means = mean_of_means_of_step_stream();
            if (mean_means - 0.5).abs() > 0.05 * 0.5 {
                failures += 1;
            };
        }
        println!(
            "failures: {:?}, number of runs: {}",
            failures, number_of_runs
        );
    }
}

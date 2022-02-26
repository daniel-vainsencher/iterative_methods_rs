//! Library code for example from crate top-level documentation

use super::*;

#[derive(Debug, Clone)]
pub struct DerivativeDescent<V, D>
where
    V: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    pub value: V,
    pub derivative: D,
    pub step_size: f64,
    pub x: f64,
}

impl<V, D> DerivativeDescent<V, D>
where
    V: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    pub fn new(value: V, derivative: D, step_size: f64, x_0: f64) -> DerivativeDescent<V, D> {
        DerivativeDescent {
            value,
            derivative,
            step_size,
            x: x_0,
        }
    }

    pub fn value(&self) -> f64 {
        (&self.value)(self.x)
    }
}

impl<V, D> StreamingIterator for DerivativeDescent<V, D>
where
    V: Fn(f64) -> f64,
    D: Fn(f64) -> f64,
{
    type Item = DerivativeDescent<V, D>;
    fn advance(&mut self) {
        self.x -= self.step_size * (self.derivative)(self.x);
    }

    fn get(&self) -> Option<&Self::Item> {
        Some(self)
    }
}

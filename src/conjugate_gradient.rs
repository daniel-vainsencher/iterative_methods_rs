//! Implementation of conjugate gradient

//! following [lecture notes](http://www.math.psu.edu/shen_w/524/CG_lecture.pdf)
//! by Shen. Thanks Shen!
//!
//! Pseudo code:
//! Set r_0 = A*x_0 - b and p_0 =-r_0, k=0
//! while r_k != 0:
//!   alpha_k = ||r_k||^2 / ||p_k||^2_A
//!   x_{k+1} = x_k + alpha_k*p_k
//!   r_{k+1} = r_K + alpha_k*A*p_k
//!   beta_k = ||r_{k+1}||^2 / ||r_k||^2
//!   p_{k+1} = -r_k + beta_k * p_k
//!   k += 1

use crate::utils::{LinearSystem, M, S, V};
use crate::IterativeMethod; 
use ndarray::ArrayBase;
use std::f64::{MIN_POSITIVE, NAN};
use streaming_iterator::*;

// A few notes:
//
// - Why do we compute `r_{k+1}` using `r_k`? doesn't this accumulate
// numerical errors? we could instead use `A*x_k - b`, right? yes, but
// we do it because it almost halves the cost of each iteration.
//
// - What is the computationally expensive part of this algorithm?
// multiplying vectors by A is (by far) the most expensive operation
// in the loop. However, note that A*p_k is already being computed for
// ||p_k||^2_A = p_k.T * A * p_k, so we reuse it for the residual (see
// previous note), instead of computing `A*x_k`. This is a
// speed/stability tradeoff.
//
// - The above algorithm potentially skips the loop entirely whenever
// `r_0 == 0`, and implicitly returns `x_k` when the loop ends.

/// Store the state of a conjugate gradient computation.
///
/// This implementation deviates from the pseudocode given in this
/// module slightly: the prelude to the loop is run in initialization,
/// and advance implements the loop. Since advance is always called
/// before get, the currently calculated quantities (suffixed by `_k`)
/// are one step ahead of what we should return. Thus we store and
/// return the previous calculated quantities (suffixed `_km` for "k
/// minus one"). Also, once denominators are very close to zero we
/// stop to avoid numerical instabilities.
#[derive(Clone, Debug)]
pub struct ConjugateGradient {
    /// Problem data
    pub a: M,
    pub b: V,
    /// Latest iterate (might numerically deteriorate)
    pub x_k: V,
    /// x_{k-1}: this is the solution we guard from numerical issues
    /// and should be reported.
    pub solution: V,
    /// Residual
    pub r_k: V,
    pub p_k: V,
    pub alpha_k: S,
    pub beta_k: S,
    /// ||r_k||^2
    pub r_k2: S,
    /// ||r_{k-1}||^2 : one measure of residual of the reported
    /// solution.
    pub r_km2: S,
    /// Ap_k
    pub ap_k: V,
    /// ||p_k||_A^2 = p_k.T*A*p_k
    pub pap_k: S,
    /// ||p_{k-1}||_A^2: another measure of error of the reported
    /// solution.
    pub pap_km: S,
}

impl ConjugateGradient {
    /// Initialize a conjugate gradient iterative solver to solve linear system `p`.
    pub fn for_problem(p: &LinearSystem) -> ConjugateGradient {
        let x_0 = match &p.x0 {
            Some(x) => x.clone(),
            None => ArrayBase::zeros(p.a.shape()[0]),
        };

        // Set r_0 = A*x_0 - b and p_0 =-r_0, k=0
        let r_k = (&p.a.dot(&x_0) - &p.b).to_shared();
        let r_k2 = r_k.dot(&r_k);
        let r_km2 = NAN;
        let p_k = -r_k.clone();
        let ap_k = p.a.dot(&p_k).to_shared();
        let pap_k = p_k.dot(&ap_k);
        let pap_km = NAN;
        ConjugateGradient {
            x_k: x_0.clone(),
            solution: x_0,
            a: p.a.clone(),
            b: p.b.clone(),
            r_k,
            r_k2,
            r_km2,
            p_k,
            ap_k,
            pap_k,
            pap_km,
            alpha_k: NAN,
            beta_k: NAN,
        }
    }
}

/// A threshold below which we do not reduce denominators further to
/// avoid solution instability.
fn too_small(v: S) -> bool {
    v < 10. * MIN_POSITIVE
}

impl StreamingIterator for ConjugateGradient {
    type Item = Self;
    fn advance(&mut self) {
        // while r_k != 0:
        //   alpha_k = ||r_k||^2 / ||p_k||^2_A

        self.alpha_k = self.r_k2 / self.pap_k;
        if (!too_small(self.r_k2)) && (!too_small(self.pap_k)) {
            //   x_{k+1} = x_k + alpha_k*p_k
            self.solution = self.x_k.clone();
            self.x_k = (self.x_k.clone() + &self.p_k * self.alpha_k).to_shared();

            //   r_{k+1} = r_K + alpha_k*A*p_k
            self.r_k = (self.r_k.clone() + &self.ap_k * self.alpha_k).to_shared();
            self.r_km2 = self.r_k2;
            self.r_k2 = self.r_k.dot(&self.r_k);

            //   beta_k = ||r_{k+1}||^2 / ||r_k||^2
            self.beta_k = self.r_k2 / self.r_km2;

            //   p_{k+1} = -r_k + beta_k * p_k
            self.p_k = (-&self.r_k + (self.beta_k * &self.p_k)).to_shared();
            self.ap_k = (self.a.dot(&self.p_k)).to_shared();
            self.pap_km = self.pap_k;
            self.pap_k = self.p_k.dot(&self.ap_k);
            //   k += 1 is implicit in this implementation since the
            //   counter is not maintained.
        } else {
            self.r_km2 = self.r_k2;
            self.pap_km = self.pap_k;
        }
    }

    fn get(&self) -> Option<&Self::Item> {
        if !too_small(self.r_km2) && (!too_small(self.pap_km)) {
            Some(self)
        } else {
            None
        }
    }
}

impl IterativeMethod<ConjugateGradient, V> for ConjugateGradient {
    fn solution(&self) -> &V {
        &self.solution
    }
    fn cost(&self) -> f64 {
        self.r_km2
    }
}

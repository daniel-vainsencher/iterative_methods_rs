use crate::utils::LinearSystem;
use ndarray::ArcArray1;
use ndarray::ArcArray2;
use ndarray::ArrayBase;
use streaming_iterator::*;

// Scalar, Vector and Matrix types:
pub type S = f64;
pub type V = ArcArray1<S>;
pub type M = ArcArray2<S>;

/// Implementation of conjugate gradient

/// following
/// http://www.math.psu.edu/shen_w/524/CG_lecture.pdf.
/// Thanks Shen!

// Pseudo code:
// Set r_0 = A*x_0 - b and p_0 =-r_0, k=0
// while r_k != 0:
//   alpha_k = ||r_k||^2 / ||p_k||^2_A 
//   x_{k+1} = x_k + alpha_k*p_k
//   r_{k+1} = r_K + alpha_k*A*p_k
//   beta_k = ||r_{k+1}||^2 / ||r_k||^2
//   p_{k+1} = -r_k + beta_k * p_k
//   k += 1

// A few notes:
//
// - Why do we compute `r_{k+1}` using `r_k`? doesn't this accumulate
// numerical errors? we could instead use `A*x_k - b`, right? yes, but
// we do it because it almost halves the cost of each iteration.
//
// - What is the computationally expensive part of this algorithm?
// multiplying vectors by A is (by far) the most expensive operation
// in the loop. However, note that A*p_k is already being computed for
// ||p_k||^2_A = p_k.T * A * p_k, so we can use a cached version for
// the residual, instead of computing `A*x_k`.
//
// - The above algorithm potentially skips the loop entirely whenever
// `r_0 == 0`, and implicitly returns `x_k` when the loop ends. For a
// streaming implementation we will modify it mildly:
//
// 1. We return each x_k as it becomes available.
//
// 2. In particular, x_0 will be returned unconditionally and before
// computing the first matrix vector product.
// 
// 3. To initialize state only as necessary, we split it up.
//
// 4. Conjugate gradient divides by ||r_k||^2, which is eventually
// close enough to 0 to cause numerical issues. After returning the
// last good iterate, we mark the end of the stream by returning None.

struct SimpleConjugateGradient {
    V x_k;
    V solution;
    V b;
    M a;
    i64 k;
    V r_k;
    V p_k;
    S alpha_k;
    S beta_k;
    // Cached quantities
    S r_k2;
    S pr_k2;
    V ap_k;
}

/// This implementation deviates from the algorithm slightly: the
/// prelude to the loop is run in initialization, and advance
/// implements the loop. Since advance is always called before get,
/// this means the very first element in the sequence, x_0, is not
/// returned.
fn simple_conjugate_gradient(&LinearSystem p) -> SimpleConjugateGradient {
    let x_0 = match p.x_0 {
        Some(x) => x,
        None() => ArrayBase::zeros(problem.b.shape()),
    }
    
    let r_k = &p.a.dot(&x_0) - &p.b;
    let r_k2 = r_k.dot(r_k);
    let pr_k2 = NAN;
    let p_k = -r_k;
    let ap_k = p.a.dot(p_k);
    SimpleConjugateGradient {
        x_k = x_0,
        solution = x_0;
        a = p.a,
        b = p.b,
        k=-1,
        r_k,
        r_k2,
        pr_k2,
        p_k,
        ap_k,
        alpha_k=f64::NAN,
        beta_k=f64::NAN,
    }
}
// Set r_0 = A*x_0 - b and p_0 =-r_0, k=0
// while r_k != 0:
//   alpha_k = ||r_k||^2 / ||p_k||^2_A 
//   x_{k+1} = x_k + alpha_k*p_k
//   r_{k+1} = r_K + alpha_k*A*p_k
//   beta_k = ||r_{k+1}||^2 / ||r_k||^2
//   p_{k+1} = -r_k + beta_k * p_k
//   k += 1

impl StreamingIterator for SimpleConjugateGradient {
    type Item = Self;
    fn advance(&mut self) {
        if !too_small(self.r_k2) {
            //   alpha_k = ||r_k||^2 / ||p_k||^2_A 
            self.alpha_k = self.r_k2 / self.p.dot(self.ap_k);
            
            //   x_{k+1} = x_k + alpha_k*p_k
            self.solution = self.x_k;
            self.x_k = self.x_k + self.alpha_k * self.p_k;
            
            //   beta_k = ||r_{k+1}||^2 / ||r_k||^2
            self.beta_k = self.r_k2 / self.pr_k2;
            
            //   p_{k+1} = -r_k + beta_k * p_k
            self.p_k = -self.r_k + self.beta_k * self.p_k;
            
            // precalculate for next condition check
            self.pr_k2 = self.r_k2;
            self.r_k2 = self.r_k.dot(self.r_k);
        }
    }
    fn get(&self) -> &self {
        self
    }
}

/// Iterative solver for a*x=b
/// Where a is p.s.d.
struct ConjugateGradient {
    V x_k;
    V b;
    M a;
    u64 k;
    Option<ConjugateGradientState> state;
}

/// When inside the loop, we have the following additional state
struct ConjugateGradientState {
    V r_k;
    V p_k;
    S alpha_k;
    S beta_k;
}

fn conjugate_gradient(&LinearSystem p) -> conjugate_gradient {
    // r_0 = A*x_0 - b and p_0 =-r_0, k=0
    let x_0 = match p.x_0 {
        Some(x) => x,
        None() => ArrayBase::zeros(problem.b.shape()),
    }
    
    ConjugateGradient {
        x_k = x_0,
        a = p.a,
        b = p.b,
        next_k=0,
        state=None,
    }
}

impl StreamingIterator for ConjugateGradient {
    type Item = Self;
    fn advance(&mut self) {
        (self.next_k, self.state) = match (self.next_k, self.state) {
            // to get x_0 we need only advance the next_k.
            (0, None) => (1, None),
            // we return x_1 only if r_0 was not sufficiently small
            (1, None) => {
                r_0 = self.a.dot(self.x_k) - self.b;
                p_0 = -r_k;
                (2, Some(ConjugateGradientState({
                    alpha_k = 0,
                    beta_k = 0,
                    r_k = r_0,
                    p_k = p_0,
                }))),
            },
            // the full loop body
            (k, Some(prev)) => {
                //   alpha_k = ||r_k||^2 / ||p_k||^2_A
                let alpha = r_k
                //   x_{k+1} = x_k + alpha_k*p_k
                //   r_{k+1} = r_K + alpha_k*A*p_k
                //   beta_k = ||r_{k+1}||^2 / ||r_k||^2
                //   p_{k+1} = -r_k + beta_k * p_k
            },
            (bad, Some(prev)) => (bad, Some(prev))
        }
        
        
    }

    fn get(&self) -> &Self::Item {
        &self.x_k
    }
}

pub use crate::conjugate_gradient::ConjugateGradient;
use ndarray::ArcArray1;
use ndarray::ArcArray2;
pub type S = f64;
pub type M = ArcArray2<S>;
pub type V = ArcArray1<S>;

/// Unit Tests Module
#[cfg(test)]
mod tests {

    use std::fmt::format;

    use crate::conjugate_gradient::ConjugateGradient;
    use crate::inspect;
    use crate::last;
    use crate::utils::make_3x3_pd_system_1;
    use crate::utils::make_3x3_psd_system;
    use crate::utils::{LinearSystem, M, V};
    extern crate nalgebra as na;
    use eigenvalues::algorithms::lanczos::HermitianLanczos;
    use eigenvalues::SpectrumTarget;
    use na::{DMatrix, DVector, Dynamic};
    use ndarray::rcarr1;
    use ndarray::rcarr2;
    use quickcheck::{quickcheck, TestResult};
    use streaming_iterator::StreamingIterator;

    pub fn solve_approximately(p: LinearSystem) -> V {
        let solution = ConjugateGradient::for_problem(&p).take(20);
        last(solution.map(|s| s.x_k.clone()))
            .expect("ConjugateGradient should always return a solution.")
    }

    pub fn show_progress(p: LinearSystem) {
        let cg_iter = ConjugateGradient::for_problem(&p).take(20);
        let mut cg_print_iter = inspect(cg_iter, |result| {
            eprintln!(
                "pap_k = {:.10}, ||Ax - b ||_2^2 = {:.5}, for x = {:.4}, and Ax - b = {:.5}",
                result.pap_k,
                result.r_k2,
                result.solution,
                result.a.dot(&result.solution) - &result.b,
            );
        });
        while let Some(_cgi) = cg_print_iter.next() {}
    }

    #[test]
    fn test_alt_eig() {
        let dm = DMatrix::from_row_slice(3, 3, &[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]);
        eprintln!("dm: {}", dm);

        let high = HermitianLanczos::new(dm, 3, SpectrumTarget::Highest)
            .unwrap()
            .eigenvalues[(0, 0)];
        eprintln!("high: {}", &high);
        assert!((high - 3.).abs() < 0.001);
    }

    fn eigvals(m: &M) -> Result<DVector<f64>, String> {
        let shape = m.shape();
        let h = shape[0];
        let w = shape[1];
        assert_eq!(h, w);
        let elems = m.reshape(h * w).to_vec();
        let dm = na::DMatrix::from_vec_generic(Dynamic::new(h), Dynamic::new(w), elems);
        Ok(HermitianLanczos::new(dm, 3, SpectrumTarget::Highest)?.eigenvalues)
    }
    fn check_psd_eigvals(p: &LinearSystem, lower: f64, upper: f64) -> Result<(), String> {
        let eigvals =
            eigvals(&p.a).unwrap_or_else(|_| panic!("Failed to compute eigenvalues for {}", &p.a));

        // Ensure A is positive definite with no extreme eigenvalues.
        if eigvals.iter().all(|ev| &lower < ev && ev < &upper) {
            Ok(())
        } else {
            Err(format!(
                "system: {:?}\n has out of bounds eigen values: {}",
                p, eigvals
            ))
        }
    }
    fn test_arbitrary_3x3_psd(p: LinearSystem) -> TestResult {
        // Decomposition should always succeed as p.a is p.s.d. by
        // construction; if not this is a bug in the test.

        show_progress(p.clone());
        let x = solve_approximately(p.clone());
        let res = p.a.dot(&x) - &p.b;
        let res_square_norm = res.dot(&res);
        eprintln!("x: {}", x);
        //
        if res_square_norm < 1e-10 {
            TestResult::passed()
        } else {
            TestResult::error(format!(
                "Residual Square Norm is too big: {}",
                res_square_norm
            ))
        }
    }

    fn maybe_instance(vs: Vec<u16>, b: Vec<u16>) -> Option<LinearSystem> {
        // Currently require dimension 3
        if b.len() < 3 || vs.len() < 9 {
            return None;
        };
        let vs = rcarr1(&vs[0..9])
            .reshape((3, 3))
            .map(|i| *i as f64)
            .into_shared();
        let b = rcarr1(&b[0..3]).map(|i| *i as f64).into_shared();
        Some(make_3x3_psd_system(vs, b))
    }

    #[test]
    fn test_arb_counter() {
        let tres = test_arbitrary_3x3_psd(
            maybe_instance(
                vec![0, 0, 0, 0, 0, 0, 0, 0, 8178],
                vec![0, 0, 22014, 7230, 22299],
            )
            .expect("Valid case"),
        );
        eprintln!("{:?}", tres);
    }
    #[test]
    fn test_arb_counter2() {
        let tres = test_arbitrary_3x3_psd(
            maybe_instance(
                vec![1, 0, 0, 1, 1, 0, 0, 1, 78],
                vec![0, 0, 22014, 7230, 22299],
            )
            .expect("Valid case"),
        );
        eprintln!("{:?}", tres);
    }
    #[test]
    fn test_arb_counter3() {
        let tres = test_arbitrary_3x3_psd(
            maybe_instance(
                vec![1, 0, 0, 0, 1, 0, 0, 1, 78],
                vec![0, 0, 22014, 7230, 22299],
            )
            .expect("Valid case"),
        );
        eprintln!("{:?}", tres);
    }
    quickcheck! {
        /// Test that we obtain a low precision solution for small p.s.d.
        /// matrices of not-too-large numbers.
        fn prop_small_numbers(vs: Vec<u16>, b: Vec<u16>) -> TestResult {
            match maybe_instance(vs, b) {
                Some(p) => test_arbitrary_3x3_psd(p),
                _ => TestResult::discard()
            }

        }
    }

    #[test]
    fn cg_simple_test() {
        let p = make_3x3_pd_system_1();
        eprintln!("Problem is: {:?}", p);
        show_progress(p.clone());
        let x = solve_approximately(p.clone());
        let r = p.a.dot(&x) - p.b;
        eprintln!("Residual is: {}", r);
        let res_square_norm = r.dot(&r);
        eprintln!("Residual squared norm is: {}", res_square_norm);
        assert!(res_square_norm < 1e-10);
    }

    #[test]
    fn cg_simple_passed() {
        let p = LinearSystem {
            a: rcarr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]),
            b: rcarr1(&[0.0, 1., 0.]),
            x0: None,
        };

        eprintln!("Problem is: {:?}", p);
        show_progress(p.clone());
        eprintln!("done showing");
        let x = solve_approximately(p.clone());
        let r = p.a.dot(&x) - p.b;
        eprintln!("Residual is: {}", r);
        let res_square_norm = r.dot(&r);
        eprintln!("Residual squared norm is: {}", res_square_norm);
        assert!(res_square_norm < 1e-10);
    }

    #[test]
    fn cg_zero_x() {
        let result = test_arbitrary_3x3_psd(
            maybe_instance(vec![0, 0, 1, 1, 0, 0, 0, 1, 0], vec![0, 0, 0]).expect("Valid case"),
        );
        assert!(!result.is_failure());
        assert!(!result.is_error());
    }

    #[ignore]
    #[test]
    fn cg_rank_one_v() {
        // This test is currently discarded by test_arbitrary_3x3_pd
        let result = test_arbitrary_3x3_psd(
            maybe_instance(vec![0, 0, 0, 0, 0, 0, 1, 43, 8124], vec![0, 0, 1]).expect("Valid case"),
        );
        assert!(!result.is_failure());
        assert!(!result.is_error());
    }

    #[test]
    fn cg_horribly_conditioned() {
        // This example is very highly ill-conditioned:
        // eigvals: [2904608166.992541+0i, 0.0000000010449559455574797+0i, 0.007460513747178893+0i]
        // therefore is currently discarded by the upper bound on eigenvalues.
        assert_eq!(
            maybe_instance(vec![0, 0, 0, 0, 0, 1, 101, 4654, 53693], vec![0, 0, 6]),
            None
        );
    }
}

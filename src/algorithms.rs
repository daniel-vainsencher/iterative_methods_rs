pub use crate::conjugate_gradient::ConjugateGradient;
use ndarray::ArcArray1;
use ndarray::ArcArray2;
pub type S = f64;
pub type M = ArcArray2<S>;
pub type V = ArcArray1<S>;

/// Unit Tests Module
#[cfg(test)]
mod tests {

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
        let cg_iter = ConjugateGradient::for_problem(&p);
        let mut cg_print_iter = inspect(cg_iter.take(20), |result| {
            eprintln!(
                "******\npap_k = {:.16}, ||Ax - b ||_2^2 = {:.16}, rk_m2 = {:.16}, for \nx = \n{:.36}, with \nA-norm residual of {:.16}, and \nAx - b = \n{:.5}, and \nap_k = \n{:.16}\npap_k = \n{:.16}\nand alpha = {:.5}, and beta = {:.5}",
                result.pap_k,
                result.r_k2,
                result.r_km2,
                result.x_k,
                result.r_k.dot(&result.a).dot(&result.r_k),
                result.a.dot(&result.x_k) - &result.b,
                result.ap_k,
                result.pap_k,
                result.alpha_k,
                result.beta_k
            );
            //eprintln!("result = {:?}", result);
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
    fn test_arbitrary_3x3_psd(p: LinearSystem) {
        // Decomposition should always succeed as p.a is p.s.d. by
        // construction; if not this is a bug in the test.
        eprintln!("p: {:?}", p);
        show_progress(p.clone());
        let result = last(ConjugateGradient::for_problem(&p).take(20))
            .expect("ConjugateGradient return None");
        //let x = solve_approximately(p.clone());

        eprintln!("x: {:.6}", &result.x_k);
        eprintln!("res: {:.6}", &result.r_k);
        eprintln!("res_k2: {:.6}", &result.r_k2);
        eprintln!("res_km2: {:.6}", &result.r_km2);
        eprintln!("pap_k2: {:.6}", &result.pap_k);
        let residual = result.a.dot(&result.x_k) - &result.b;
        eprintln!("residual: {:.6}", &residual);
        let a_res = &result.a.dot(&residual);
        eprintln!("a_residual: {:.6}", &a_res);
        let residual_a_norm = residual.dot(a_res);
        eprint!("residual A norm: {:.16}", residual_a_norm);
        assert!(
            residual_a_norm < 1e-10,
            "Norm of residual in A norm is too big: {:.16}",
            residual_a_norm
        );
        assert!(
            result.pap_k < 1e-10,
            "Norm of update direction in A norm is too big: {}",
            result.pap_k
        );
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
    #[test]
    fn test_arb_counter4() {
        test_arbitrary_3x3_psd(
            maybe_instance(vec![0, 0, 0, 0, 0, 0, 0, 0, 0], vec![0, 0, 1]).expect("Valid case"),
        );
    }
    quickcheck! {
        /// Test that we obtain a low precision solution for small p.s.d.
        /// matrices of not-too-large numbers.
        fn prop_small_numbers(vs: Vec<u16>, b: Vec<u16>) -> TestResult {
            match maybe_instance(vs.clone(), b.clone()) {
                Some(p) => {
                    eprintln!("vs: {:?}", vs);
                    eprintln!("b: {:?}", b);
                    eprintln!("p: {:?}", p);
                    test_arbitrary_3x3_psd(p);
                    TestResult::passed()
                }
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
        test_arbitrary_3x3_psd(
            maybe_instance(vec![0, 0, 1, 1, 0, 0, 0, 1, 0], vec![0, 0, 0]).expect("Valid case"),
        );
    }

    #[test]
    fn cg_rank_one_v() {
        // This test is currently discarded by test_arbitrary_3x3_pd
        test_arbitrary_3x3_psd(
            maybe_instance(vec![0, 0, 0, 0, 0, 0, 1, 43, 8124], vec![0, 0, 1]).expect("Valid case"),
        );
    }

    #[test]
    fn cg_horribly_conditioned() {
        // This example is very highly ill-conditioned:
        // eigvals: [2904608166.992541+0i, 0.0000000010449559455574797+0i, 0.007460513747178893+0i]
        // therefore is currently discarded by the upper bound on eigenvalues.
        /*assert_eq!(
            maybe_instance(vec![0, 0, 0, 0, 0, 1, 101, 4654, 53693], vec![0, 0, 6]),
            None
        );*/
    }
}

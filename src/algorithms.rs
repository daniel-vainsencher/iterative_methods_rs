pub mod cg_method {

    use crate::last;
    use crate::tee;
    use crate::utils::LinearSystem;
    use ndarray::ArcArray1;
    use ndarray::ArcArray2;
    use ndarray::ArrayBase;
    use streaming_iterator::*;
    pub type S = f64;
    pub type M = ArcArray2<S>;
    pub type V = ArcArray1<S>;

    /// The state of a conjugate gradient algorithm.
    #[derive(Clone)]
    pub struct CGIterable {
        pub a: M,
        pub b: V,
        pub x: V,
        pub alpha: S,
        pub r: V,
        pub rs: S,
        pub rsprev: S,
        pub p: V,
        pub ap: V,
    }

    impl CGIterable {
        /// Convert a LinearSystem problem into a StreamingIterator of conjugate gradient solutions.
        pub fn conjugate_gradient(problem: LinearSystem) -> CGIterable {
            let x = match problem.x0 {
                None => ArrayBase::zeros(problem.a.shape()[1]),
                Some(init_x) => init_x,
            };
            let r = problem.b.clone() - problem.a.dot(&x).view();
            let p = r.clone();
            let ap = problem.a.dot(&p).into_shared();
            CGIterable {
                a: problem.a,
                b: problem.b,
                x,
                alpha: 1.,
                r,
                rs: 1.,
                rsprev: 1.,
                p,
                ap,
            }
        }
    }

    impl StreamingIterator for CGIterable {
        type Item = CGIterable;
        /// Implementation of conjugate gradient iteration
        fn advance(&mut self) {
            self.rsprev = self.rs;
            self.rs = self.r.dot(&self.r);
            if self.rs.abs() <= std::f64::MIN_POSITIVE * 10. {
                return;
            }
            self.p = (&self.r + &(&self.rs / self.rsprev * &self.p)).into_shared();
            self.ap = self.a.dot(&self.p).into_shared();
            self.alpha = self.rs / self.p.dot(&self.ap);
            self.x += &(self.alpha * &self.p);
            self.r -= &(self.alpha * &self.ap);
        }
        fn get(&self) -> Option<&Self::Item> {
            if self.rsprev.abs() <= std::f64::MIN_POSITIVE * 10. {
                None
            } else {
                Some(self)
            }
        }
    }

    pub fn solve_approximately(p: LinearSystem) -> V {
        let solution = CGIterable::conjugate_gradient(p).take(200);
        last(solution.map(|s| s.x.clone()))
    }

    pub fn show_progress(p: LinearSystem) {
        let cg_iter = CGIterable::conjugate_gradient(p).take(50);
        //.take_while(|cgi| cgi.rsprev.sqrt() > 1e-6);
        let mut cg_print_iter = tee(cg_iter, |result| {
            let res = result.a.dot(&result.x) - &result.b;
            let res_norm = res.dot(&res);
            println!(
                "rs = {:.10}, ||Ax - b ||_2^2 = {:.5}, for x = {:.4}, and Ax - b = {:.5}",
                result.rs,
                res_norm,
                result.x,
                result.a.dot(&result.x) - &result.b,
            );
        });
        while let Some(_cgi) = cg_print_iter.next() {}
    }
}

/// Unit Tests Module
#[cfg(test)]
mod tests {

    use super::cg_method::*;
    use crate::last;
    use crate::utils::make_3x3_psd_system;
    use crate::utils::make_3x3_psd_system_1;
    use crate::utils::LinearSystem;

    use streaming_iterator::*;
    extern crate nalgebra as na;
    use eigenvalues::algorithms::lanczos::HermitianLanczos;
    use eigenvalues::SpectrumTarget;
    use na::{DMatrix, DVector, Dynamic};
    use ndarray::rcarr1;
    use ndarray::rcarr2;

    use quickcheck::{quickcheck, TestResult};

    #[test]
    fn test_alt_eig() {
        let dm = DMatrix::from_row_slice(3, 3, &[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0]);
        println!("dm: {}", dm);

        let high = HermitianLanczos::new(dm.clone(), 3, SpectrumTarget::Highest)
            .unwrap()
            .eigenvalues[(0, 0)];
        println!("high: {}", &high);
        assert!((high - 3.).abs() < 0.001);
    }

    fn eigvals(m: &M) -> Result<DVector<f64>, String> {
        let shape = m.shape();
        let h = shape[0];
        let w = shape[1];
        assert_eq!(h, w);
        let elems = m.reshape(h * w).to_vec();
        let dm = na::DMatrix::from_vec_generic(Dynamic::new(h), Dynamic::new(w), elems);
        Ok(
            HermitianLanczos::new(dm.clone(), 3, SpectrumTarget::Highest)?
                .eigenvalues
                .clone(),
        )
    }

    fn test_arbitrary_3x3_psd(vs: Vec<u16>, b: Vec<u16>) -> TestResult {
        // Currently require dimension 3
        if b.len().pow(2) != vs.len() || b.len() != 3 {
            return TestResult::discard();
        }
        let vs = rcarr1(&vs).reshape((3, 3)).map(|i| *i as f64).into_shared();
        let b = rcarr1(&b).map(|i| *i as f64).into_shared();
        let p = make_3x3_psd_system(vs, b);
        // Decomposition should always succeed as p.a is p.s.d. by
        // construction; if not this is a bug in the test.
        let eigvals = eigvals(&p.a).expect(&format!("Failed to compute eigenvalues for {}", &p.a));

        // Ensure A is positive definite with no extreme eigenvalues.
        if !eigvals.iter().all(|ev| &1e-8 < ev && ev < &1e9) {
            return TestResult::discard();
        }

        println!("eigvals of a: {}", eigvals);

        println!("a: {}", p.a);
        println!("b: {}", p.b);
        let x = solve_approximately(p.clone());
        let res = p.a.dot(&x) - &p.b;
        let res_square_norm = res.dot(&res);
        println!("x: {}", x);
        show_progress(p.clone());
        //
        TestResult::from_bool(res_square_norm < 1e-40)
    }

    quickcheck! {
        /// Test that we obtain a low precision solution for small p.s.d.
        /// matrices of not-too-large numbers.
        fn prop(vs: Vec<u16>, b: Vec<u16>) -> TestResult {
            test_arbitrary_3x3_psd(vs, b)
        }
    }

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
    fn cg_simple_test() {
        let p = make_3x3_psd_system_1();
        println!("Problem is: {:?}", p);
        show_progress(p.clone());
        let x = solve_approximately(p.clone());
        let r = p.a.dot(&x) - p.b;
        println!("Residual is: {}", r);
        let res_square_norm = r.dot(&r);
        println!("Residual squared norm is: {}", res_square_norm);
        assert!(res_square_norm < 1e-10);
    }

    #[test]
    fn cg_simple_passed() {
        let p = LinearSystem {
            a: rcarr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]),
            b: rcarr1(&[0.0, 1., 0.]),
            x0: None,
        };

        println!("Problem is: {:?}", p);
        show_progress(p.clone());
        let x = solve_approximately(p.clone());
        let r = p.a.dot(&x) - p.b;
        println!("Residual is: {}", r);
        let res_square_norm = r.dot(&r);
        println!("Residual squared norm is: {}", res_square_norm);
        assert!(res_square_norm < 1e-10);
    }

    #[test]
    fn cg_zero_x() {
        let result = test_arbitrary_3x3_psd(vec![0, 0, 1, 1, 0, 0, 0, 1, 0], vec![0, 0, 0]);
        assert!(!result.is_failure());
        assert!(!result.is_error());
    }

    #[test]
    fn cg_rank_one_v() {
        // This test is currently discarded by test_arbitrary_3x3_pd
        let result = test_arbitrary_3x3_psd(vec![0, 0, 0, 0, 0, 0, 1, 43, 8124], vec![0, 0, 1]);
        assert!(!result.is_failure());
        assert!(!result.is_error());
    }

    #[test]
    fn cg_horribly_conditioned() {
        // This example is very highly ill-conditioned:
        // eigvals: [2904608166.992541+0i, 0.0000000010449559455574797+0i, 0.007460513747178893+0i]
        // therefore is currently discarded by the upper bound on eigenvalues.
        let result =
            test_arbitrary_3x3_psd(vec![0, 0, 0, 0, 0, 1, 101, 4654, 53693], vec![0, 0, 6]);
        assert!(!result.is_failure());
        assert!(!result.is_error());
    }
}

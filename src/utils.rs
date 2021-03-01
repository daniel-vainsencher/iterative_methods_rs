use crate::algorithms::cg_method::*;
use ndarray::{rcarr1, rcarr2};

/// A linear system, ax-b=0, to be solved iteratively, with an optional initial solution.
#[derive(Clone, Debug)]
pub struct LinearSystem {
    pub a: M,
    pub b: V,
    pub x0: Option<V>,
}

pub fn make_3x3_psd_system_1() -> LinearSystem {
    make_3x3_psd_system(
        rcarr2(&[[1., 2., -1.], [0., 1., 0.], [0., 0., 1.]]),
        rcarr1(&[0., 1., 0.]),
    )
}

pub fn make_3x3_psd_system_2() -> LinearSystem {
    make_3x3_psd_system(
        rcarr2(&[[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]),
        rcarr1(&[0., 1., 0.]),
    )
}

pub fn make_3x3_psd_system(m: M, b: V) -> LinearSystem {
    let a = (m.t().dot(&m)).into_shared();
    LinearSystem {
        a: a,
        b: b,
        x0: None,
    }
}

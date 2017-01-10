extern crate rusty_machine;
extern crate rand;
extern crate rayon;

use self::rusty_machine::linalg::{Matrix, Vector};
use self::rusty_machine::stats::dist::Distribution;
use self::rusty_machine::stats::dist::gaussian::Gaussian;
use self::rand::distributions::{Normal, IndependentSample};
use self::rayon::prelude::*;


#[derive(Debug)]
pub struct BaseParticleFilter;

pub trait _BaseParticleFilter {
    fn generate_random_values_from_norm_dist(&self, n_particles: &usize) -> Vector<f64>;
    fn compute_normal_pdf(&self, upper: &f64, loc: f64, scale: f64) -> f64;
}

impl _BaseParticleFilter for BaseParticleFilter {
    fn generate_random_values_from_norm_dist(&self, n_particles: &usize) -> Vector<f64> {
        let norm = Normal::new(0f64, 1f64);
        fn normal_sampling(norm: &Normal) -> f64 {norm.ind_sample(&mut rand::thread_rng())}

        // Vector::from((0..*n_particles).map(|_| normal_sampling(&norm)).collect::<Vec<f64>>())
        let mut random_vals = vec![0f64; *n_particles];
        vec![0f64; *n_particles].par_iter()
                                .map(|_| normal_sampling(&norm))
                                .collect_into(&mut random_vals);
        Vector::from(random_vals)
    }

    fn compute_normal_pdf(&self, upper: &f64, loc: f64, scale: f64) -> f64 {
        let gauss = Gaussian::new(loc, scale);
        gauss.pdf(*upper)
    }
}

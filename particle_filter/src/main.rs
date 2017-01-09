extern crate rusty_machine;
extern crate rand;
extern crate rayon;

use rusty_machine::linalg::{Vector};
use rand::Rng;

mod base;
use base::{_BaseParticleFilter, BaseParticleFilter};


struct ParticleFilter {
    n_particles: usize,
    predicted_value: Vec<f64>,  // number of particles.
    filtered_value: Vec<f64>,
    system_noise_variance: f64,  // Variance of system noise.
    log_likelihood: f64,  // log likelihood.
    teeth_of_comb: Vector<f64>,  // Horizontal comb used for resampling(reuse).
    weights: Vector<f64>,  // Unit mass of particle (fitness to observation data)
    particles: Vector<f64>,
    predicted_particles: Vector<f64>,  // Predictive distribution.
    lsm: f64,  // Square error.
    base: BaseParticleFilter,
}

impl Default for ParticleFilter {
    fn default() -> ParticleFilter {
        ParticleFilter::new(1000, 2.0f64.powf(3f64).sqrt())
    }
}

impl ParticleFilter {
    fn new(n_particles: usize, system_noise_variance: f64) -> ParticleFilter {
        ParticleFilter {
            system_noise_variance: system_noise_variance,
            predicted_particles: Vector::new(vec![]),
            log_likelihood: 0f64,
            predicted_value: Vec::new(),
            filtered_value: Vec::new(),
            lsm: 0f64,

            n_particles: n_particles,
            weights: Vector::zeros(n_particles),
            particles: Vector::zeros(n_particles),
            teeth_of_comb: Vector::new(vec![]),
            base: BaseParticleFilter,
        }
    }

    fn init_praticles_distribution(&mut self) {
        // initialize particles
        // x_0|0
        self.particles = self.base.generate_random_values_from_norm_dist(&self.n_particles);
        self.teeth_of_comb = Vector::from((0..self.n_particles)
                                            .map(|x| {let a: f64 = x as f64;
                                                      let b: f64= self.n_particles.pow(2) as f64;
                                                      a/b})
                                            .collect::<Vec<f64>>());
    }

    fn generate_system_noise(&self) -> Vector<f64> {
        // v_t
        self.base.generate_random_values_from_norm_dist(&self.particles.size())
    }

    fn compute_pred_particles(&mut self) -> Vector<f64> {
        // only update
        // calculate system function
        // x_t|t-1
        println!("generate_system_noise: {}", &self.generate_system_noise().size());
        println!("particles: {}", &self.particles.size());
        &self.particles + &self.generate_system_noise()
    }

    fn calculate_particles_weight(&mut self, y: &f64) {
        // only update
        // calculate fitness probabilities between observation value and predicted value
        // w_t
        let locs = &self.compute_pred_particles();
        self.predicted_particles = locs.clone();
        self.weights = Vector::from((0..self.particles.size())
                        .map(|i| self.base.compute_normal_pdf(y, locs[i], self.system_noise_variance))
                        .collect::<Vec<f64>>());
    }

    fn calculate_likelihood(&mut self) {
        // alculate likelihood at that point
        // p(y_t|y_1:t-1)
        let res = self.weights.sum()/(self.n_particles as f64);
        self.log_likelihood += res.ln();
    }

    fn normalize_weights(&mut self) {
        // wtilda_t
        let sum_of_weights: f64 = self.weights.sum();
        self.weights = Vector::from(self.weights.iter()
                                                 .map(|x| x/sum_of_weights)
                                                 .collect::<Vec<f64>>());
    }

    fn memorize_predicted_value(&mut self) {
        let predicted_value = (&self.predicted_particles.elemul(&self.weights)).sum();
        self.predicted_value.push(predicted_value);
    }

    fn memorize_filtered_value(&mut self, selected_idx: &Vec<usize>, y: &f64) {
        let filtered_value =
            (&self.particles.elemul(&self.weights.select(selected_idx))).sum()
            /&self.weights.select(selected_idx).sum();
        self.filtered_value.push(filtered_value);
        self.calculate_lsm(y, &filtered_value);
    }

    fn resample(&mut self, y: &f64) {
        fn cumsum(valus: &Vector<f64>) -> Vector<f64> {
            let mut sum = 0f64;
            let mut res = Vec::new();
            for v in valus {
                sum += *v;
                res.push(sum);
            }
            Vector::from(res)
        }
        // x_t|t
        self.normalize_weights();
        self.memorize_predicted_value();

        // accumulate weight
        let cum = cumsum(&self.weights);

        // create roulette pointer
        let base = rand::thread_rng().next_f64()/(self.n_particles as f64);
        let pointers = &self.teeth_of_comb.data()
                                         .iter()
                                         .map(|x| x+&base)
                                         .collect::<Vec<f64>>();

        // select particles
        let selected_idx = &cum.iter()
                              .enumerate()
                              .filter(|&(idx, v)| v>=&pointers[idx])
                              .map(|(idx, _)| idx)
                              .collect::<Vec<usize>>();

        // println!("{:?}", &selected_idx);
        self.particles = self.predicted_particles.select(&selected_idx);
        self.memorize_filtered_value(&selected_idx, y);
    }

    fn calculate_lsm(&mut self, y: &f64, filterd_value: &f64) {
        self.lsm += (y-filterd_value).powf(2f64);
    }

    fn forward(&mut self, y: &f64) {
        // compute system model and observation model.
        self.calculate_particles_weight(y);
        self.calculate_likelihood();
        self.resample(y);
    }
}

fn main() {
    let mut pf = ParticleFilter::default();
    &pf.init_praticles_distribution();
    let mut data = Vec::new();
    data.extend_from_slice(&[0.5f64; 20]);
    data.extend_from_slice(&[1.5f64; 60]);
    data.extend_from_slice(&[-1f64; 20]);

    for (idx, d) in data.iter().enumerate() {
        println!("iter: {}", idx);
        &pf.forward(d);
    }
    println!("log likelihood: {}", &pf.log_likelihood);
    println!("lsm: {}", &pf.lsm);
    println!("===predicted_value================");
    println!("{:?}", &pf.predicted_value[0..20]);
    println!("{:?}", &pf.predicted_value[20..80]);
    println!("{:?}", &pf.predicted_value[80..100]);
}

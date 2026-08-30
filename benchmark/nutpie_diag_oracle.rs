//! Standalone physical oracle for the ReactiveKernels nutpie diagonal-adaptation corpus.
//!
//! This deliberately has no crate dependencies.  Its operations and ownership order are
//! transcribed from pymc-devs/nuts-rs revision
//! 97be9ab88cfaadfafd9e5f4409a3b1d5af62805a.  The generated TOML receipt records the
//! complete upstream file digests, every raw position/gradient input, and every state value
//! as exact IEEE-754 bits.  ReactiveKernels is neither imported nor consulted.

const N: usize = 4;
const LOWER: f64 = 1e-20;
const UPPER: f64 = 1e20;

#[derive(Clone)]
struct RunningVariance {
    mean: [f64; N],
    variance: [f64; N],
    count: u64,
}

impl RunningVariance {
    fn new() -> Self {
        Self { mean: [0.0; N], variance: [0.0; N], count: 0 }
    }

    // src/transform/adapt/diagonal.rs::RunningVariance::add_sample plus
    // src/math/cpu_math.rs::array_update_variance.
    fn add_sample(&mut self, value: &[f64; N]) {
        self.count += 1;
        if self.count == 1 {
            self.mean.copy_from_slice(value);
        } else {
            let diff_scale = (self.count as f64).recip();
            for i in 0..N {
                let diff = value[i] - self.mean[i];
                self.mean[i] += diff * diff_scale;
                // Source-faithful: nuts-rs uses old-mean diff squared, not Welford M2.
                self.variance[i] += diff * diff;
            }
        }
    }
}

#[derive(Clone)]
struct DiagMassMatrix {
    mean: [f64; N],
    inv_stds: [f64; N],
    stds: [f64; N],
    logdet: f64,
    id: i64,
}

impl DiagMassMatrix {
    fn new() -> Self {
        Self { mean: [0.0; N], inv_stds: [0.0; N], stds: [0.0; N], logdet: 0.0, id: -1 }
    }

    // src/transform/diagonal.rs::update_diag_grad and
    // src/math/cpu_math.rs::array_update_var_inv_std_grad.
    fn update_diag_grad(&mut self, position: &[f64; N], gradient: &[f64; N]) {
        for i in 0..N {
            let val = gradient[i].abs().clamp(LOWER, UPPER).recip();
            let val = if val.is_finite() { val } else { 1.0 };
            self.stds[i] = val.sqrt();
            self.inv_stds[i] = val.recip().sqrt();
        }
        for i in 0..N {
            let variance = self.stds[i] * self.stds[i];
            self.mean[i] = variance * gradient[i];
            self.mean[i] += position[i];
        }
        self.logdet = self.inv_stds.iter().map(|value| value.ln()).sum();
        self.id += 1;
    }

    // src/transform/diagonal.rs::update_diag_draw_grad and
    // src/math/cpu_math.rs::array_update_var_inv_std_draw_grad.
    fn update_diag_draw_grad(
        &mut self,
        draw_mean: &[f64; N],
        grad_mean: &[f64; N],
        draw_variance: &[f64; N],
        grad_variance: &[f64; N],
    ) {
        for i in 0..N {
            let val = (draw_variance[i] / grad_variance[i]).sqrt();
            // fill_invalid = None: invalid dimensions retain the previous scale.
            if val.is_finite() && val != 0.0 {
                let val = val.clamp(LOWER, UPPER);
                self.stds[i] = val.sqrt();
                self.inv_stds[i] = val.recip().sqrt();
            }
        }
        for i in 0..N {
            let variance = self.stds[i] * self.stds[i];
            self.mean[i] = variance * grad_mean[i];
            self.mean[i] += draw_mean[i];
        }
        self.logdet = self.inv_stds.iter().map(|value| value.ln()).sum();
        self.id += 1;
    }
}

struct Adaptation {
    draw: RunningVariance,
    grad: RunningVariance,
    background_draw: RunningVariance,
    background_grad: RunningVariance,
    transform: DiagMassMatrix,
}

impl Adaptation {
    fn new() -> Self {
        Self {
            draw: RunningVariance::new(),
            grad: RunningVariance::new(),
            background_draw: RunningVariance::new(),
            background_grad: RunningVariance::new(),
            transform: DiagMassMatrix::new(),
        }
    }

    // Exact Strategy::init ownership order: current draw, background draw,
    // current gradient, background gradient, then transform initialization.
    fn init(&mut self, position: &[f64; N], gradient: &[f64; N]) {
        self.draw.add_sample(position);
        self.background_draw.add_sample(position);
        self.grad.add_sample(gradient);
        self.background_grad.add_sample(gradient);
        self.transform.update_diag_grad(position, gradient);
    }

    // Exact Strategy::update_estimators ownership order.
    fn update_estimators(&mut self, position: &[f64; N], gradient: &[f64; N], is_good: bool) {
        if is_good {
            self.draw.add_sample(position);
            self.grad.add_sample(gradient);
            self.background_draw.add_sample(position);
            self.background_grad.add_sample(gradient);
        }
    }

    // Exact Strategy::switch move order: draw first, then gradient.
    fn switch(&mut self) {
        self.draw = std::mem::replace(&mut self.background_draw, RunningVariance::new());
        self.grad = std::mem::replace(&mut self.background_grad, RunningVariance::new());
    }

    fn adapt(&mut self) -> bool {
        assert_eq!(self.draw.count, self.grad.count);
        if self.draw.count < 3 {
            return false;
        }
        self.transform.update_diag_draw_grad(
            &self.draw.mean,
            &self.grad.mean,
            &self.draw.variance,
            &self.grad.variance,
        );
        true
    }
}

struct Step {
    position: [f64; N],
    gradient: [f64; N],
    is_good: bool,
    switch_now: bool,
    adapt_now: bool,
}

fn print_bits(name: &str, values: &[f64; N]) {
    print!("{name} = [");
    for (i, value) in values.iter().enumerate() {
        if i != 0 { print!(", "); }
        print!("\"{:016x}\"", value.to_bits());
    }
    println!("]");
}

fn print_input(name: &str, position: &[f64; N], gradient: &[f64; N],
               is_good: bool, switch_now: bool, adapt_now: bool) {
    println!("\n[[inputs]]");
    println!("name = \"{name}\"");
    print_bits("position", position);
    print_bits("gradient", gradient);
    println!("is_good = {is_good}");
    println!("switch_now = {switch_now}");
    println!("adapt_now = {adapt_now}");
}

fn print_stage(name: &str, state: &Adaptation, adapted: bool) {
    println!("\n[[stages]]");
    println!("name = \"{name}\"");
    println!("adapted = {adapted}");
    println!("count = {}", state.draw.count);
    println!("background_count = {}", state.background_draw.count);
    print_bits("draw_mean", &state.draw.mean);
    print_bits("draw_variance", &state.draw.variance);
    print_bits("grad_mean", &state.grad.mean);
    print_bits("grad_variance", &state.grad.variance);
    print_bits("background_draw_mean", &state.background_draw.mean);
    print_bits("background_draw_variance", &state.background_draw.variance);
    print_bits("background_grad_mean", &state.background_grad.mean);
    print_bits("background_grad_variance", &state.background_grad.variance);
    print_bits("stds", &state.transform.stds);
    print_bits("inv_stds", &state.transform.inv_stds);
    print_bits("transformation_mean", &state.transform.mean);
    println!("logdet = \"{:016x}\"", state.transform.logdet.to_bits());
    println!("transformation_id = {}", state.transform.id);
}

fn main() {
    println!("schema = 1");
    println!("upstream_repo = \"https://github.com/pymc-devs/nuts-rs\"");
    println!("upstream_revision = \"97be9ab88cfaadfafd9e5f4409a3b1d5af62805a\"");
    println!("rng_consumption = \"none\"");
    println!("float_encoding = \"lowercase IEEE-754 binary64 bits\"");
    println!("ownership_order = \"init: current draw, background draw, current gradient, background gradient, transform; step: current draw, current gradient, background draw, background gradient, switch draw, switch gradient, adapt\"");
    for (path, sha256) in [
        ("src/transform/adapt/diagonal.rs", "7b6c0ed9c914eb26e492c77e38c9536d73b211cf90ca05f315e487d081c72960"),
        ("src/transform/diagonal.rs", "3a8eb25fa91da6343a4fe36800b1b4e11abd8a80393dba19e6954104252946b7"),
        ("src/math/cpu_math.rs", "84e4ae66e629fcbbb9259da115ca7d3ec265dc16a50e9ba611942e4ce6aafeb7"),
        ("src/adapt_strategy.rs", "93c0ddd5ec7ddfa653e1e7382887944d728e12e861af85efd84b3cc4e6f0b837"),
    ] {
        println!("[[upstream_files]]");
        println!("path = \"{path}\"");
        println!("sha256 = \"{sha256}\"");
    }

    let init_position = [1.0, 10.0, -2.0, 3.0];
    let init_gradient = [4.0, 2.0, -0.5, 1.0];
    print_input("init", &init_position, &init_gradient, true, false, true);
    let mut state = Adaptation::new();
    state.init(&init_position, &init_gradient);
    print_stage("init", &state, true);

    let steps = [
        Step { position: [2.0, 10.0, -1.0, 3.0], gradient: [3.0, 3.0, -0.5, 1.0], is_good: true, switch_now: false, adapt_now: true },
        Step { position: [4.0, 10.0, 1.0, 3.0], gradient: [1.0, 5.0, -0.5, 1.0], is_good: true, switch_now: false, adapt_now: true },
        Step { position: [99.0, 99.0, 99.0, 99.0], gradient: [99.0, 99.0, 99.0, 99.0], is_good: false, switch_now: false, adapt_now: true },
        Step { position: [7.0, 10.0, 4.0, 3.0], gradient: [-2.0, 8.0, -0.5, 1.0], is_good: true, switch_now: true, adapt_now: true },
        Step { position: [11.0, 10.0, 8.0, 3.0], gradient: [-6.0, 13.0, -0.5, 1.0], is_good: true, switch_now: false, adapt_now: true },
    ];

    for (index, step) in steps.iter().enumerate() {
        let name = format!("step{}", index + 1);
        print_input(&name, &step.position, &step.gradient,
                    step.is_good, step.switch_now, step.adapt_now);
        // GlobalStrategy ordering: update, optional switch, then adapt.
        state.update_estimators(&step.position, &step.gradient, step.is_good);
        if step.switch_now { state.switch(); }
        let adapted = if step.adapt_now { state.adapt() } else { false };
        print_stage(&name, &state, adapted);
    }
}

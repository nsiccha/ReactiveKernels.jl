#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "walnutpie/walnuts.hpp"

namespace wd = walnutpie::detail;

struct GaussianLogpGrad {
  double stiffness;
  mutable std::size_t calls = 0;

  void operator()(const Eigen::VectorXd& theta, double& logp,
                  Eigen::VectorXd& grad) const {
    ++calls;
    logp = -0.5 * stiffness * theta.squaredNorm();
    grad = -stiffness * theta;
  }
};

struct AcceptanceRecorder {
  std::vector<double> values;
  void operator()(double value) { values.push_back(value); }
  double step_size() const { return 0.0; }
};

struct MacroResult {
  bool accepted;
  std::size_t logp_grad_calls;
  double theta;
  double rho;
  double joint;
  double base_accept;
};

MacroResult run_macro(double stiffness, double theta0, double rho0,
                      double macro_time, std::size_t max_step_halvings,
                      std::size_t min_micro_steps, double max_error) {
  GaussianLogpGrad logp_grad{stiffness};
  Eigen::VectorXd theta(1), rho(1), grad(1), inv_mass(1);
  theta << theta0;
  rho << rho0;
  inv_mass << 1.0;
  double logp_pos;
  logp_grad(theta, logp_pos, grad);
  const double joint = logp_pos + wd::logp_momentum(rho, inv_mass);
  auto span = wd::SpanW::from_initial_point(
      Eigen::VectorXd(theta), Eigen::VectorXd(rho), Eigen::VectorXd(grad),
      logp_pos, joint);

  Eigen::VectorXd theta_next, rho_next, grad_next;
  double logp_pos_next = 0.0;
  double logp_next = 0.0;
  AcceptanceRecorder recorder;
  logp_grad.calls = 0;
  const bool accepted = wd::macro_step<wd::Direction::Forward>(
      logp_grad, inv_mass, macro_time / min_micro_steps,
      max_step_halvings, min_micro_steps, max_error, span, theta_next,
      rho_next, grad_next, logp_pos_next, logp_next, recorder);
  return {accepted,
          logp_grad.calls,
          theta_next(0),
          rho_next(0),
          logp_next,
          recorder.values.empty() ? -1.0 : recorder.values.front()};
}

void print_result(const std::string& id, double stiffness, double theta0,
                  double rho0, double macro_time,
                  std::size_t max_step_halvings,
                  std::size_t min_micro_steps, double max_error) {
  const auto out = run_macro(stiffness, theta0, rho0, macro_time,
                             max_step_halvings, min_micro_steps, max_error);
  std::cout << id << '\t' << out.accepted << '\t' << out.logp_grad_calls
            << '\t' << out.theta << '\t' << out.rho << '\t' << out.joint
            << '\t' << out.base_accept << '\n';
}

int main() {
  std::cout << std::setprecision(17);
  std::cout << "case\taccepted\tlogp_grad_calls\ttheta\trho\tjoint\tbase_accept\n";
  // Base grid: one forward micro step and no reverse coarser-grid check.
  print_result("base_grid_accept", 1.0, 1.0, 0.3, 0.1, 4, 1, 1.0);
  // The 1- and 2-step forward grids fail; 4 passes.  Reverse checks run on
  // the 2- and 1-step coarser grids, so the exact callback count is 1+2+4+2+1.
  print_result("dyadic_reverse_accept", 10.0, 1.0, 0.3, 0.5, 4, 1,
               0.5);
  // The forward 1-step grid fails and the 2-step grid passes, but the reverse
  // 1-step grid also passes.  WALNUTS-D therefore rejects (1+2+1 calls).
  print_result("reverse_grid_reject", 0.1, -2.0, -3.0, 3.0, 4, 1, 1.0);
  // Every allowed forward grid fails: 1+2+4+8 calls, no reverse checks.
  print_result("all_grids_reject", 10.0, 1.0, 0.3, 0.5, 4, 1, 1e-6);
}

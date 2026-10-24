#include <RcppEigen.h>
#include <vector>
#include <cmath>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Rcpp::List heavy_ball_cpp(
    const Eigen::MappedSparseMatrix<double>& A,
    const Eigen::Map<Eigen::VectorXd>& b,
    double lambda_min,
    double lambda_max,
    double tol = 1e-6,
    int max_iter = 1000
) {

  int n = b.size();

  double denom = std::sqrt(lambda_max) + std::sqrt(lambda_min);
  const double H = std::pow(2.0 / denom, 2.0);
  const double ALPHA = std::pow((std::sqrt(lambda_max) - std::sqrt(lambda_min))
                                  / denom, 2.0);

  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd x_prev = x;
  Eigen::VectorXd r = b;

  std::vector<double> res_log;
  double r_norm = r.norm();
  res_log.push_back(r_norm);

  int k;

  for (k = 0; k < max_iter; k++) {

    if (r_norm <= tol) {
      break;
    }

    Eigen::VectorXd x_next = x + H * r + ALPHA * (x - x_prev);

    x_prev = x;
    x = x_next;

    r = b - A * x;

    r_norm = r.norm();
    res_log.push_back(r_norm);

  }

  return Rcpp::List::create(
    Rcpp::Named("x") = x,
    Rcpp::Named("iterations") = k,
    Rcpp::Named("residual_history") = res_log
  );

}

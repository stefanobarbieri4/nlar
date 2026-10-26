#include <RcppEigen.h>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Rcpp::List jacobi_cpp(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    double tol = 1e-6,
    int max_iter = 1000
) {
  int n = b.size();
  Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd r = b;

  std::vector<double> res_log;
  double r_norm = r.norm();
  res_log.push_back(r_norm);

  const Eigen::VectorXd diag = A.diagonal();

  int k;
  for (k = 0; k < max_iter; k++) {
    if (r_norm <= tol) {
      break;
    }

    x = x + (r.array() / diag.array()).matrix();

    r = b - A * x;
    r_norm = r.norm()
    res_log.push_back(r_norm);
  }

  return Rcpp::List::create(
    Rcpp::Named("x") = x,
    Rcpp::Named("iterations") = k,
    Rcpp::Named("residual_history") = res_log
  );
}

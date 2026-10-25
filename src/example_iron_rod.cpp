#include <RcppEigen.h>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Rcpp::List create_iron_rod_example(
  int n,
  double beta,
  Rcpp::Function f
) {

  double h = 1.0 / (n + 1);
  double a = -1.0 - (h * beta) / 2.0;
  double e = -1.0 + (h * beta) / 2.0;

  Eigen::SparseMatrix<double> A(n, n);
  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(3 * n);

  for (int i = 0; i < n; ++i) {

    triplets.push_back(Eigen::Triplet<double>(i, i, 2.0));

    if (i > 0) triplets.push_back(Eigen::Triplet<double>(i, i - 1, a));
    if (i < n - 1) triplets.push_back(Eigen::Triplet<double>(i, i + 1, e));
  }

  A.setFromTriplets(triplets.begin(), triplets.end());

  Eigen::VectorXd b(n);
  for (int i = 0; i < n; i++) {
    double y = (i + 1) * h;
    double f_val = Rcpp::as<double>(f(y));
    b(i) = h * h * f_val;
  }

  return Rcpp::List::create(
    Rcpp::Named("A") = A,
    Rcpp::Named("b") = b,
    Rcpp::Named("a") = Eigen::VectorXd::Constant(n - 1, a),
    Rcpp::Named("d") = Eigen::VectorXd::Constant(n, 2.0),
    Rcpp::Named("e") = Eigen::VectorXd::Constant(n - 1, e)
  );
}

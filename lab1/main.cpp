#include <iostream>
#include <cmath>
#include <omp.h>

#define EPS 1e-10

bool compare(const double &x, const double &y) {
  return fabs(x - y) < EPS;
}

double coeff(int dCount, double *x,
             const double *xleft, const double *xright) {
  double result = 0;
  for (int i = 0; i < dCount; i++) {
    if (compare(x[i], xleft[i]) || compare(x[i], xright[i])) result++;
  }

  return 1.0 / pow(2, result);
}

double multH(int dCount, const double *h) {
  double result = 1;
  for (int i = 0; i < dCount; i++) {
    result *= h[i];
  }
  return result;
}

int multN(int dCount, const int n) {
  return (int)pow(n, dCount);
}

double initH(int dCount, double *h, int n, double *xleft, double *xright) {
  for (int i = 0; i < dCount; i++) {
    h[i] = ((xright[i] - xleft[i]) / n);
  }
}

double func(double *x) {
  return exp(-x[0] - x[1] - x[2]);
//  return x[0] + x[1];
}

void getPointByStep(int dCount, int step, int n, double *h,
                    double *x, double *xleft){
  for (int i = 0; i<dCount; i++){
    x[i] = step % (n + 1);
    x[i] = xleft[i] + x[i]*h[i];
    step/= (n + 1);
  }
}

double integral(int dCount, int n, double *xleft, double *xright){
  double *h = new double[dCount];
  initH(dCount, h, n, xleft, xright);
  double H = multH(dCount, h);
  int N = multN(dCount, n);
  double sum = 0; double coef = 0;
  double *x = new double[dCount];
  for (int i = 0; i<N; i++){
    getPointByStep(dCount, i, n, h, x, xleft);
    coef = coeff(dCount, x, xleft, xright);
    sum+=coef*H*func(x);
  }
  return sum;
}

double pintegral(int dCount, int n, double *xleft, double *xright){
  double *h = new double[dCount];
  initH(dCount, h, n, xleft, xright);
  double H = multH(dCount, h);
  int N = multN(dCount, n);
  double sum = 0; double coef = 0;
  double localSum = 0; int i;
  #pragma omp parallel private(i, coef) firstprivate(localSum) \
      shared(H, N, dCount, xleft, xright) reduction(+:sum) num_threads(2)
  {
    double *x = new double[dCount];
    #pragma omp for
    for (i = 0; i < N; i++) {
      getPointByStep(dCount, i, n, h, x, xleft);
      coef = coeff(dCount, x, xleft, xright);
      localSum += coef * H * func(x);
    }
    sum+=localSum;
  }
  return sum;
}

int main() {
  const int dCount = 3;
  double *xleft = new double[dCount], *xright = new double[dCount];
  int n = 500;

  xleft[0] = 0; xright[0] = 1;
  xleft[1] = 0; xright[1] = 5;
  xleft[2] = 0; xright[2] = 1;


  double t0, t1, diffTime1, diffTime2;
  double result;

  t0 = omp_get_wtime();
  result = integral(dCount, n, xleft, xright);
  t1 = omp_get_wtime();
  diffTime1 = t1 - t0;
  std::cout << "Integral: " << result << std::endl;
  std::cout << "Time linear: " << diffTime1 << std::endl;

  std::cout << "========================" << std::endl;

  t0 = omp_get_wtime();
  result = pintegral(dCount, n, xleft, xright);
  t1 = omp_get_wtime();
  diffTime2 = t1-t0;
  std::cout<<"Integral: "<<result<<std::endl;
  std::cout<<"Time parallel with tasks: "<<diffTime2<<std::endl;

  std::cout<<"========================"<<std::endl;

  std::cout<<"Profit = "<<diffTime1/diffTime2<<std::endl;
  return 0;
}
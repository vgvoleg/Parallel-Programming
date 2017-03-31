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

double H_m = 0;

double multH(int dCount, const double *h) {
  double result = 1;
  for (int i = 0; i < dCount; i++) {
    result *= h[i];
  }
  return result;
}

double initH(int dCount, double *h, int *n, double *xleft, double *xright) {
  for (int i = 0; i < dCount; i++) {
    h[i] = ((xright[i] - xleft[i]) / n[i]);
  }
}

double func(double *x) {
  return exp(-x[0] - x[1] - x[2] - x[3]);
}

double linear_nIntegral(int dCount, int dCurr, double *h, int *n,
                        double *x, double *xleft, double *xright) {
  if (dCurr == dCount - 1) {
    double sum = 0, localSum = 0;
    double coef;
    int i;
    for (i = 0; i <= n[dCurr]; i++) {
      x[dCurr] = xleft[dCurr] + i * h[dCurr];
      coef = coeff(dCount, x, xleft, xright);
      localSum += coef * H_m * func(x);
    }
    sum += localSum;
    return sum;
  }

  if (dCount - dCurr == 2) {
    double sum = 0;
    for (int i = 0; i <= n[dCurr]; i++) {
      x[dCurr] = xleft[dCurr] + i * h[dCurr];
      sum += linear_nIntegral(dCount, dCurr + 1, h, n, x, xleft, xright);
    }
    return sum;
  }

  double sum1 = 0, sum2 = 0;
  for (double i = xleft[dCurr]; i < (xright[dCurr] - xleft[dCurr]) / 2; i += h[dCurr]) {
    x[dCurr] = i;
    sum1 += linear_nIntegral(dCount, dCurr + 1, h, n, x, xleft, xright);
  }
  for (double i = (xright[dCurr] - xleft[dCurr]) / 2; i <= xright[dCurr]; i += h[dCurr]) {
    x[dCurr] = i;
    sum2 += linear_nIntegral(dCount, dCurr + 1, h, n, x, xleft, xright);
  }
  return sum1 + sum2;

}

double parallel_nIntegral(int dCount, int dCurr, double *h, int *n,
                           double *x, double *xleft, double *xright) {
  if (dCount - dCurr == 1) {
    double sum = 0, localSum = 0;
    double coef;
    int i;
    for (i = 0; i <= n[dCurr]; i++) {
      x[dCurr] = xleft[dCurr] + i * h[dCurr];
      coef = coeff(dCount, x, xleft, xright);
      localSum += coef * H_m * func(x);
    }
    sum += localSum;
    return sum;
  }
  if (dCount - dCurr == 2) {
    double sum = 0;
    for (int i = 0; i <= n[dCurr]; i++) {
      x[dCurr] = xleft[dCurr] + i * h[dCurr];
      sum += linear_nIntegral(dCount, dCurr + 1, h, n, x, xleft, xright);
    }
    return sum;
  }

  double sum1 = 0, sum2 = 0;
  #pragma omp task firstprivate (x) shared (sum1, h, xleft, xright)
  {
    for (double i = xleft[dCurr]; i < (xright[dCurr] - xleft[dCurr]) / 2; i += h[dCurr]) {
      x[dCurr] = i;
      sum1 += parallel_nIntegral(dCount, dCurr + 1, h, n, x, xleft, xright);
    }
  }
  #pragma omp task firstprivate (x) shared (sum2, h, xleft, xright)
  {
    for (double i = (xright[dCurr] - xleft[dCurr]) / 2; i <= xright[dCurr]; i += h[dCurr]) {
      x[dCurr] = i;
      sum2 += parallel_nIntegral(dCount, dCurr + 1, h, n, x, xleft, xright);
    }
  }
  #pragma omp taskwait
  return sum1 + sum2;
}


//double parallel_nIntegralTBB(int dCount, int dCurr, double *h, int *n,
//                          double *x, double *xleft, double *xright) {
//  if (dCount - dCurr == 1) {
//    double sum = 0, localSum = 0;
//    double coef;
//    int i;
//    for (i = 0; i <= n[dCurr]; i++) {
//      x[dCurr] = xleft[dCurr] + i * h[dCurr];
//      coef = coeff(dCount, x, xleft, xright);
//      localSum += coef * H_m * func(x);
//    }
//    sum += localSum;
//    return sum;
//  }
//  double sum1 = 0, sum2 = 0;
//  tbb::task_group g;
//  g.run([&]{
//    for (double i = xleft[dCurr]; i < (xright[dCurr] - xleft[dCurr]) / 2; i += h[dCurr]) {
//      x[dCurr] = i;
//      sum1 += parallel_nIntegral(dCount, dCurr + 1, h, n, x, xleft, xright);
//    }
//  });
//
//  g.run([&]{
//    for (double i = (xright[dCurr] - xleft[dCurr]) / 2; i <= xright[dCurr]; i += h[dCurr]) {
//      x[dCurr] = i;
//      sum2 += parallel_nIntegral(dCount, dCurr + 1, h, n, x, xleft, xright);
//    }
//  });
//  g.wait();
//  return sum1 + sum2;
//}

int main() {
  const int dCount = 4;
  int *n = new int[dCount];
  double *h = new double[dCount];
  double *xleft = new double[dCount], *xright = new double[dCount];

  n[0] = 50;
  n[1] = 50;
  n[2] = 50;
  n[3] = 50;
//  n[3] = 350;
//  n[4] = 20;

  xleft[0] = -3; xright[0] = 1;
  xleft[1] = -2; xright[1] = 3;
  xleft[2] = 0; xright[2] = 1;
  xleft[3] = -3; xright[3] = 3;
//  xleft[4] = -3; xright[4] = 3;

  initH(dCount, h, n, xleft, xright);
  H_m = multH(dCount, h);

  double t0, t1, diffTime1, diffTime2;
  double result;

  double *x = new double[dCount];

  t0 = omp_get_wtime();
  result = linear_nIntegral(dCount, 0, h, n, x, xleft, xright);
  t1 = omp_get_wtime();
  diffTime1 = t1 - t0;
  std::cout << "Integral: " << result << std::endl;
  std::cout << "Time linear: " << diffTime1 << std::endl;

  std::cout << "========================" << std::endl;

  #pragma omp parallel num_threads(2)
  {
    #pragma omp single
    {
      t0 = omp_get_wtime();
      result = parallel_nIntegral(dCount, 0, h, n, x, xleft, xright);
      t1 = omp_get_wtime();
      diffTime2 = t1-t0;
    }
  }
  std::cout<<"Integral: "<<result<<std::endl;
  std::cout<<"Time parallel with tasks: "<<diffTime2<<std::endl;

  std::cout<<"========================"<<std::endl;

  std::cout<<"Profit = "<<diffTime1/diffTime2<<std::endl;


//  tbb::task_scheduler_init init(2);
//  t0 = omp_get_wtime();
//  result = parallel_nIntegralTBB(dCount, 0, h, n, x, xleft, xright);
//  t1 = omp_get_wtime();
//  diffTime2 = t1-t0;
//  std::cout<<"Integral: "<<result<<std::endl;
//  std::cout<<"Time parallel with tasks: "<<diffTime2<<std::endl;
//
//  std::cout<<"========================"<<std::endl;
//
//  std::cout<<"Profit = "<<diffTime1/diffTime2<<std::endl;
  return 0;
}
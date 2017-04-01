#include <iostream>
#include "tbb/tbb.h"
#include "tbb/parallel_reduce.h"

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
  return (int) pow(n + 1, dCount);
}

double initH(int dCount, double *h, int n, double *xleft, double *xright) {
  for (int i = 0; i < dCount; i++) {
    h[i] = ((xright[i] - xleft[i]) / n);
  }
}

void getPointByStep(int dCount, int step, int n, double *h,
                    double *x, double *xleft) {
  for (int i = 0; i < dCount; i++) {
    x[i] = step % (n + 1);
    x[i] = xleft[i] + x[i] * h[i];
    step /= (n + 1);
  }
}

double integral(int dCount, int n, double *xleft, double *xright, double (*function)(double *)) {
  double *h = new double[dCount];
  initH(dCount, h, n, xleft, xright);
  double H = multH(dCount, h);
  int N = multN(dCount, n);
  double sum = 0;
  double coef = 0;
  double *x = new double[dCount];
  for (int i = 0; i < N; i++) {
    getPointByStep(dCount, i, n, h, x, xleft);
    coef = coeff(dCount, x, xleft, xright);
    sum += coef * H * function(x);
  }
  return sum;
}

int main() {
  int dCount = 3;
  int n = 500;
  double *xleft = new double[dCount];
  double *xright = new double[dCount];
  xleft[0] = 0; xright[0] = 1;
  xleft[1] = 0; xright[1] = 5;
  xleft[2] = 0; xright[2] = 1;
  auto f = [](double *x) { return exp(-x[0]-x[1]-x[2]); };

  double *h = new double[dCount];
  initH(dCount, h, n, xleft, xright);
  double H = multH(dCount, h);

  double diffTime1 = 0;
  double diffTime2 = 0;

  tbb::tick_count start, finish;

  start = tbb::tick_count::now();
  double linear = integral(dCount, n, xleft, xright, f);
  finish = tbb::tick_count::now();
  diffTime1 = (finish - start).seconds();

  std::cout<< "Result: "<<linear<< std::endl;
  std::cout<< "Time linear: "<<diffTime1<< std::endl;
  std::cout<< "===================== "<<std::endl;


  tbb::task_scheduler_init init(2);
  start = tbb::tick_count::now();
  double parallel = tbb::parallel_reduce(tbb::blocked_range<int>(0, multN(dCount, n)), 0.0,
  [dCount, n, h, xleft, xright, H, f](tbb::blocked_range<int>&r, double sum) -> double {
    int begin = r.begin(), end = r.end();
    double *x = new double[dCount];
    double coef = 0;
    for (int i = begin; i<end; i++) {
      getPointByStep(dCount, i, n, h, x, xleft);
      coef = coeff(dCount, x, xleft, xright);
      sum += coef * H * f(x);
    }
    return sum;
  },
  [](double sumAll, double sum) -> double {
    sumAll += sum;
    return sumAll;
  });
  finish = tbb::tick_count::now();

  diffTime2 = (finish - start).seconds();

  std::cout<< "Result: "<<parallel<< std::endl;
  std::cout<< "Time parallel: "<<diffTime2<< std::endl;
  std::cout<< "===================== "<<std::endl;
  std::cout<< "Profit: "<<diffTime1/diffTime2<< std::endl;


  return 0;
}
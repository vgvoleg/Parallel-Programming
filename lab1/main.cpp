#include <iostream>
#include <cmath>
#include <omp.h>

#define EPS 1e-10

bool compare(const double &x, const double &y){
  return fabs(x-y) < EPS;
}

double coeff(int dCount, double *x,
             const double *xleft, const double *xright){
  double result = 0;
  for (int i = 0; i<dCount; i++){
    if (compare(x[i],xleft[i]) || compare(x[i],xright[i])) result++;
  }

  return 1.0/pow(2, result);
}

double multH(int dCount, const double *h){
  double result = 1;
  for (int i = 0; i<dCount; i++){
    result*=h[i];
  }
}

double initH(int dCount , double *h, int *n, double *xleft, double *xright){
  for (int i = 0; i<dCount; i++){
    h[i] = ((xright[i] - xleft[i])/n[i]);
  }
}


double func(double* x){
  return x[0]+2*x[1]-x[2]*x[2]*x[2];
}

double linear_nIntegral(int dCount, int dCurr, int *n, double *h,
                 double *x, double *xleft, double *xright){
  if (dCurr == dCount){
    double coef = coeff(dCount, x, xleft, xright);
    double H = multH(dCount, h);
    return coef*func(x)*H;
  } else {
    double summ = 0;
    for (double i = xleft[dCurr]; i<=xright[dCurr]; i+=h[dCurr]) {
      x[dCurr] = i;
      summ+=linear_nIntegral(dCount, dCurr+1, n, h, x, xleft, xright);
    }
    return summ;
  }
}

double parallel_nIntegral(int dCount, int *n, double *h, double *xleft, double *xright) {
  omp_set_num_threads(2);

  double *x = new double[dCount];
  double localSum = 0, totalSum = 0;

  int i;

  #pragma omp parallel shared(n, h, xleft, xright) private(i) firstprivate(x, localSum) reduction(+:totalSum)
  {
    #pragma omp for
    for (i = 0; i <= n[0]; i++) {
      x[0] = xleft[0] + i * h[0];
      localSum += linear_nIntegral(dCount, 1, n, h, x, xleft, xright);
    }
    totalSum+=localSum;
  }
  return totalSum;
}

int main() {
  const int dCount = 3;
  int *n = new int[dCount];
  double* h = new double[dCount], *x = new double[dCount];
  double* xleft = new double[dCount], *xright = new double[dCount];
  n[0] = 200;
  n[1] = 200;
  n[2] = 200;
  //n[3] = 100;
  xleft[0] = 0; xright[0] = 1;
  xleft[1] = 1; xright[1] = 3;
  xleft[2] = -3; xright[2] = 3;
  //xleft[3] = -1; xright[3] = 4;

  initH(dCount, h, n, xleft, xright);

  double t0, t1, diffTime1, diffTime2;
  double result;

  t0 = omp_get_wtime();
  result = linear_nIntegral(dCount, 0, n, h, x, xleft, xright);
  t1 = omp_get_wtime();
  diffTime1 = t1-t0;
  std::cout<<"Integral: "<<result<<std::endl;
  std::cout<<"Time linear: "<<diffTime1<<std::endl;

  std::cout<<"========================"<<std::endl;

  t0 = omp_get_wtime();
  result = parallel_nIntegral(dCount, n, h, xleft, xright);
  t1 = omp_get_wtime();
  diffTime2 = t1-t0;
  std::cout<<"Integral: "<<result<<std::endl;
  std::cout<<"Time parallel: "<<diffTime2<<std::endl;

  std::cout<<"========================"<<std::endl;


  std::cout<<"Profit = "<<diffTime1/diffTime2<<std::endl;

  return 0;
}
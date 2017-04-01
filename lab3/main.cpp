#include <iostream>
#include <cmath>
#include "mpi.h"
#include "omp.h"

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

int main(int argc, char ** argv) {

  // MPI init
  int size, rank;
  int ROOT = 0;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double startTime, stopTime;

  // Initial problem
  const int dCount = 3;
  int n = 1000;
  double *xleft = new double[dCount], *xright = new double[dCount];
  xleft[0] = 0; xright[0] = 1;
  xleft[1] = 0; xright[1] = 5;
  xleft[2] = 0; xright[2] = 1;
  auto f = [](double *x) { return exp(-x[0]-x[1]-x[2]); };
  double result = 0 ;

  // Main program

  double *h = new double[dCount];
  initH(dCount, h, n, xleft, xright);
  double H = multH(dCount, h);
  int N = multN(dCount, n);
  double sum = 0;
  double coef = 0;
  double localSum = 0;
  int i;

  startTime = MPI_Wtime();

  #pragma omp parallel private(i, coef) firstprivate(localSum) \
      shared(H, N, xleft, xright) reduction(+:sum)
  {
    double *x = new double[dCount];
    #pragma omp for
    for (i = rank; i < N; i+=size) {
      getPointByStep(dCount, i, n, h, x, xleft);
      coef = coeff(dCount, x, xleft, xright);
      localSum += coef * H * f(x);
    }
    sum += localSum;
  }

  MPI_Reduce(&sum, &result, 1, MPI_DOUBLE, MPI_SUM, ROOT, MPI_COMM_WORLD);

  if (rank == ROOT){
    stopTime = MPI_Wtime();

    printf("Time with %d process: %.15f\n", size, stopTime - startTime);
    printf("Integral value: %.15f\n", result);
  }

  MPI_Finalize();
  return 0;
}
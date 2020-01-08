#include <likwid.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv)
{

    /// Initialisation

    const double tol = atof(argv[1]);
    const int N = atoi(argv[2]);
    const int M = atoi(argv[3]);

    double *x = malloc((sizeof(double)) * (M * N));
    double *y = malloc((sizeof(double)) * (M * N));

    likwid_markerInit();
    likwid_markerRegisterRegion("jacobi");

    for(int j = 0; j < M; ++j) {
        for(int i = 0; i < N; ++i) {
            x[i + j * M] = 1.0 * sin(17.0*M_PI * i / N) * sin(17.0*M_PI * j / M);
        }
    }

    int iter = 0;

    /// Kernel

    while(x[N/2 + M*M/2] > tol) {

        likwid_markerStartRegion("jacobi");

        // sweep
        for (int j = 1; j < M - 1; ++j) {
            for (int i = 1; i < N -1; ++i) {
                y[i + j * M] = 0.25 * (x[(i-1) + j * M] + x[(i+1) + j * M] + x[i + (j-1) * M] + x[i + (j+1) * M]);
            }
        }

        likwid_markerStopRegion("jacobi");

        // swap
        double* tmp = x;
        x = y;
        y = tmp;

        ++iter;
    }

    // clean up

    likwid_markerClose();

    printf("Needed %d iterations to achieve an error of %.9f.\n", iter, x[N/2 + M*M/2]);
}

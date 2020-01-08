#include <likwid.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv)
{
    /// Initialisation

    likwid_markerInit();
    likwid_markerRegisterRegion("jacobi_2d");
    likwid_markerRegisterRegion("jacobi_3d");

    const int R = atoi(argv[1]);
    const int N = atoi(argv[2]);
    const int M = atoi(argv[3]);

    const int S = N;

    double *x = malloc((sizeof(double)) * (M * N));
    double *y = malloc((sizeof(double)) * (M * N));

    double *a = malloc((sizeof(double)) * (S * S * S));
    double *b = malloc((sizeof(double)) * (S * S * S));

    for(int j = 0; j < M; ++j) {
        for(int i = 0; i < N; ++i) {
            x[i + j * M] = 1.0 * sin(17.0*M_PI * i / N) * sin(17.0*M_PI * j / M);
        }
    }

    for(int k = 0; k < S; ++k) {
        for(int j = 0; j < S; ++j) {
            for(int i = 0; i < S; ++i) {
                a[i + j * S + k * S * S] = 1.0 * sin(17.0*M_PI * i / S) * sin(17.0*M_PI * j / S) * sin(17.0*M_PI * k / S);
            }
        }
    }

    /// Kernel

    likwid_markerStartRegion("jacobi_2d");

    for(int r = 0; r < R; ++r) {

        // sweep
        for (int j = 1; j < M - 1; ++j) {
            for (int i = 1; i < N -1; ++i) {
                y[i + j * M] = 0.25 * (x[(i-1) + j * M] + x[(i+1) + j * M] + x[i + (j-1) * M] + x[i + (j+1) * M]);
            }
        }

        // swap
        double* tmp = x;
        x = y;
        y = tmp;
    }

    likwid_markerStopRegion("jacobi_2d");

    // second kernel

    likwid_markerStartRegion("jacobi_3d");

    for(int r = 0; r < R; ++r) {

        // sweep
        for (int k = 1; k < S - 1; ++k) {
            for (int j = 1; j < S - 1; ++j) {
                for (int i = 1; i < S - 1; ++i) {
                    b[i + j * S + k * S * S] = (a[(i-1) + j * S + k * S * S] + a[(i+1) + j * S + k * S * S] +
                                                a[i + (j-1) * S+ k * S * S] + a[i + (j+1) * S+ k * S * S] +
                                                a[i + j * S + (k-1) * S * S] + a[i + j * S + (k+1) * S * S]) / 6.0;
                }
            }
        }

        // swap
        double* tmp = a;
        a = b;
        b = tmp;
    }

    likwid_markerStopRegion("jacobi_3d");

    /// clean up

    likwid_markerClose();

    printf("An error of %.9f was achieved after %d iterations.\n", x[N/2 + M*M/2], R);
}

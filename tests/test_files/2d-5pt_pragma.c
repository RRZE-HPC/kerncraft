double a[M][N];
double b[M][N];
double s;

// this pragma makes no sens, but we want to test if it pases parsing
#pragma unroll
for(int j=1; j<M-1; ++j) {
#pragma simd
    for(int i=1; i<N-1; ++i)
        b[j][i] = ( a[j][i-1] + a[j][i+1]
                  + a[j-1][i] + a[j+1][i]) * s;
}
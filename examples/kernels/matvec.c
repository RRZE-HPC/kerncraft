double a[N][M];
double b[N];
double c[N];

for(int j=0; j<M; ++j) {
    for(int i=0; i<N; ++i) {
        c[i] += a[i][j] * b[i];
    }
}


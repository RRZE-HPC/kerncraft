double a[M*N];
double b[M*N];
double s;

for(int j=1; j<M-1; ++j)
    for(int i=1; i<N-1; ++i)
        b[j*N+i] = ( a[j*N+i-1] + a[j*N+i+1]
                   + a[(j-1)*N+i] + a[(j+1)*N+i]) * s;

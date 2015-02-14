double a[N], b[N], c;

for(int i=1; i<N-1; ++i)
    b[i] = c * (a[i-1] - 2.0*a[i] + a[i+1]);

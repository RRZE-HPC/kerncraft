double a[M][N];
double b[M][N];
double s;

for(int j=1; j<M-1; ++j)
    for(int i=1; i<N-1; i+=4) {
        b[j][i] = ( a[j][i-1] + a[j][i+1]
                  + a[j-1][i] + a[j+1][i]) * s;
        b[j][i+1] = ( a[j][i] + a[j][i+2]
                    + a[j-1][i] + a[j+1][i]) * s;
        b[j][i+2] = ( a[j][i+1] + a[j][i+3]
                    + a[j-1][i] + a[j+1][i]) * s;
        b[j][i+3] = ( a[j][i+2] + a[j][i+4]
                    + a[j-1][i] + a[j+1][i]) * s;
    }

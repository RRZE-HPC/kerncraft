double a[M][N][N];
double b[M][N][N];
double coeffs[6][M][N][N], s;

for(int k=1; k<M-1; ++k)
    for(int j=1; j<N-1; ++j)
        for(int i=1; i<N-1; ++i)
            b[k][j][i] = ( coeffs[0][k][j][i]*a[k][j][i-1]
                         + coeffs[1][k][j][i]*a[k][j][i+1]
                         + coeffs[2][k][j][i]*a[k][j-1][i]
                         + coeffs[3][k][j][i]*a[k][j+1][i]
                         + coeffs[4][k][j][i]*a[k-1][j][i]
                         + coeffs[5][k][j][i]*a[k+1][j][i]) * s;

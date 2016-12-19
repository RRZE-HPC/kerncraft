double a[M][N][N];
double b[M][N][N];
double coeffs_N[M][N][N], coeffs_S[M][N][N], coeffs_W[M][N][N],
       coeffs_E[M][N][N], coeffs_F[M][N][N], coeffs_B[M][N][N],
       s;

for(int k=1; k<M-1; ++k)
    for(int j=1; j<N-1; ++j)
        for(int i=1; i<N-1; ++i)
            b[k][j][i] = ( coeffs_W[k][j][i]*a[k][j][i-1]
                         + coeffs_E[k][j][i]*a[k][j][i+1]
                         + coeffs_N[k][j][i]*a[k][j-1][i]
                         + coeffs_S[k][j][i]*a[k][j+1][i]
                         + coeffs_B[k][j][i]*a[k-1][j][i]
                         + coeffs_F[k][j][i]*a[k+1][j][i]) * s;

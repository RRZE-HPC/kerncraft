double a[M][N][N];
double b[M][N][N];
double coeffs_N, coeffs_S, coeffs_W, coeffs_E,
       coeffs_F, coeffs_B, s;

for(int k=1; k<M-1; ++k)
    for(int j=1; j<N-1; ++j)
        for(int i=1; i<N-1; ++i)
            b[k][j][i] = ( coeffs_W*a[k][j][i-1]
                         + coeffs_E*a[k][j][i+1]
                         + coeffs_N*a[k][j-1][i]
                         + coeffs_S*a[k][j+1][i]
                         + coeffs_B*a[k-1][j][i]
                         + coeffs_F*a[k+1][j][i]) * s;

/* Extracted from the Himeno Benchmark */

float gosa, ss, s0, omega;
float a[4][L][M][N], b[3][L][M][N], c[3][L][M][N];
float p[L][M][N], bnd[L][M][N], wrk1[L][M][N], wrk2[L][M][N];

for(int i=1 ; i<L-1; i++)
  for(int j=1 ; j<M-1; j++)
    for(int k=1 ; k<N-1; k++){
          s0 = a[0][i][j][k]*p[i+1][j][k]
             + a[1][i][j][k]*p[i][j+1][k]
             + a[2][i][j][k]*p[i][j][k+1]
             + b[0][i][j][k]
              *( p[i+1][j+1][k] - p[i+1][j-1][k]
               - p[i-1][j+1][k] + p[i-1][j-1][k] )
             + b[1][i][j][k]
              *( p[i][j+1][k+1] - p[i][j-1][k+1]
               - p[i][j+1][k-1] + p[i][j-1][k-1] )
             + b[2][i][j][k]
              *( p[i+1][j][k+1] - p[i-1][j][k+1]
               - p[i+1][j][k-1] + p[i-1][j][k-1] )
             + c[0][i][j][k] * p[i-1][j][k]
             + c[1][i][j][k] * p[i][j-1][k]
             + c[2][i][j][k] * p[i][j][k-1]
             + wrk1[i][j][k];

          ss = (s0*a[3][i][j][k] - p[i][j][k])*bnd[i][j][k];

          gosa += ss*ss;
          wrk2[i][j][k] = p[i][j][k] + omega*ss;
}

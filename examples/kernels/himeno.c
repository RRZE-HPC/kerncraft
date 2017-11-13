/* Extracted from the Himeno Benchmark */

float gosa, ss, s0;
float a[4][L][M][N], b[3][L][M][N], c[3][L][M][N];
float p[L][M][N], bnd[L][M][N], wrk1[L][M][N], wrk2[L][M][N];

for(i=1 ; i<L; i++)
  for(j=1 ; j<M ; j++)
    for(k=1 ; k<N ; k++){
          s0 = a[0][i][j][k]*p[0][i+1][j][k]
             + a[1][i][j][k]*p[0][i][j+1][k]
             + a[2][i][j][k]*p[0][i][j][k+1]
             + b[0][i][j][k]
              *( p[0][i+1][j+1][k] - p[0][i+1][j-1][k]
               - p[0][i-1][j+1][k] + p[0][i-1][j-1][k] )
             + b[1][i][j][k]
              *( p[0][i][j+1][k+1] - p[0][i][j-1][k+1]
               - p[0][i][j+1][k-1] + p[0][i][j-1][k-1] )
             + b[2][i][j][k]
              *( p[0][i+1][j][k+1] - p[0][i-1][j][k+1]
               - p[0][i+1][j][k-1] + p[0][i-1][j][k-1] )
             + c[0][i][j][k] * p[0][i-1][j][k]
             + c[1][i][j][k] * p[0][i][j-1][k]
             + c[2][i][j][k] * p[0][i][j][k-1]
             + wrk1[0][i][j][k];

          ss = (s0*a[3][i][j][k] - p[0][i][j][k])*bnd[0][i][j][k];

          gosa += ss*ss;
          wrk2[0][i][j][k] = p[0][i][j][k] + omega*ss;
}
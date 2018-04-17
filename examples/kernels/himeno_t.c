// Extracted from the Himeno Benchmark

// WARNING: kerncraft cannot do analytic LC analysis on this code
// Use --cache-predictor SIM instead
float gosa, ss, s0, omega;
float a[L][M][N][4], b[L][M][N][3], c[L][M][N][3];
float p[L][M][N], bnd[L][M][N], wrk1[L][M][N], wrk2[L][M][N];

#pragma omp parallel for private(s0,ss) reduction(+:gosa) schedule(static)
for(int i=1 ; i<L-1; i++) {
  for(int j=1 ; j<M-1; j++) {
    for(int k=1 ; k<N-1; k++){
          s0 = a[i][j][k][0]*p[i+1][j][k]
             + a[i][j][k][1]*p[i][j+1][k]
             + a[i][j][k][2]*p[i][j][k+1]
             + b[i][j][k][0]
              *( p[i+1][j+1][k] - p[i+1][j-1][k]
               - p[i-1][j+1][k] + p[i-1][j-1][k] )
             + b[i][j][k][1]
              *( p[i][j+1][k+1] - p[i][j-1][k+1]
               - p[i][j+1][k-1] + p[i][j-1][k-1] )
             + b[i][j][k][2]
              *( p[i+1][j][k+1] - p[i-1][j][k+1]
               - p[i+1][j][k-1] + p[i-1][j][k-1] )
             + c[i][j][k][0] * p[i-1][j][k]
             + c[i][j][k][1] * p[i][j-1][k]
             + c[i][j][k][2] * p[i][j][k-1]
             + wrk1[i][j][k];

          ss = (s0*a[i][j][k][3] - p[i][j][k])*bnd[i][j][k];

          gosa += ss*ss;
          wrk2[i][j][k] = p[i][j][k] + omega*ss;
}
}
}


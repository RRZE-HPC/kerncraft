double u1[M][N][N];
double d1[M][N][N];
double xx[M][N][N];
double xy[M][N][N];
double xz[M][N][N];
double c1, c2, d, dth;

for(int k=2; k<M-2; k++) {
    for(int j=2; j<N-2; j++) {
        for(int i=2; i<N-2; i++) {
            d = 0.25*(d1[ k ][j][i] + d1[ k ][j-1][i]
                    + d1[k-1][j][i] + d1[k-1][j-1][i]);
            u1[k][j][i] = u1[k][j][i] + (dth/d)
             * ( c1*(xx[ k ][ j ][ i ] - xx[ k ][ j ][i-1])
               + c2*(xx[ k ][ j ][i+1] - xx[ k ][ j ][i-2])
               + c1*(xy[ k ][j+1][ i ] - xy[ k ][j-1][ i ])
               + c2*(xy[ k ][j+1][ i ] - xy[ k ][j-2][ i ])
               + c1*(xz[ k ][ j ][ i ] - xz[k-1][ j ][ i ])
               + c2*(xz[k+1][ j ][ i ] - xz[k-2][ j ][ i ]));
}}}

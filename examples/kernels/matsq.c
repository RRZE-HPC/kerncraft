double S[N][N];
double D[N][N];

for(int i=0; i<N; i++) {
    for(int j=0; j<N; j++) {
        for(int k=0; k<N; k++) {
            D[i][j] = D[i][j] + S[i][k]*S[k][j];
        }
    }
}

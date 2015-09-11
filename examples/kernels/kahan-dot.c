double a[N];
double b[N];
double c;
double sum;
double prod;
double t;
double y;

for(int i=0; i<N; ++i) {
    prod = a[i]*b[i];
    y = prod-c;
    t = sum+y;
    c = (t-sum)-y;
    sum = t;
}

# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <float.h>
# include <time.h>
# include <omp.h>

#define a(i,j,k) a[(k)*mxy+(j)*mx+(i)]
#define anew(i,j,k) anew[(k)*mxy+(j)*mx+(i)]

int main ( int argc, char *argv[] ) {
	double error ;
	double epsilon ;
	int    i,j,k,istep;
	int    nx,ny,nz,nstep,nprint;
	int    mx,my,mz,mxy,mxyz ;
	int    i1,j1,k1,in,jn,kn ;
	int    i0,j0,k0,im,jm,km ;
	double asum,amax,amin ;
	double txm,txp,tym,typ,tzm,tzp ;
	double *a,*anew;
//	clock_t t0,t1,ts,te ;
	double t0,t1,ts,te ;
	FILE *INPUT ;

	istep =     0 ;
	error = 999.9 ;

	printf("%%heatx, SOP\n") ;
	INPUT = fopen("heatx.inp","r");
	fscanf(INPUT,"%d %d %d                ",&nx,&ny,&nz                  );
	fscanf(INPUT,"%d %d                   ",&nstep,&nprint               );
	fscanf(INPUT,"%lf %lf %lf %lf %lf %lf ",&txp,&txm,&typ,&tym,&tzp,&tzm);
	fscanf(INPUT,"%le                     ",&epsilon                     );
	fclose(INPUT) ;

#ifdef _OPENMP
	printf("%%heatx, Pure OpenMP Offloading\n") ;
#else
	printf("%%heatx, _NO_ OpenMP Offloading\n") ;
#endif
	printf("%%heatx, nx ny nz               : %d %d %d         \n",nx,ny,nz               ) ;
	printf("%%heatx, nstep nprint           : %d %d            \n",nstep,nprint           ) ;
	printf("%%heatx, txp,txm,typ,tym,tzp,tzm: %f %f %f %f %f %f\n",txp,txm,typ,tym,tzp,tzm) ;
	printf("%%heatx, epsilon                : %e               \n",epsilon                ) ;
	printf("%%heatx,                                           \n"                        ) ;

	i1 =  1 ; j1 =  1 ; k1 =  1 ;
	in = nx ; jn = ny ; kn = nz ;
	i0 = i1-1 ; j0 = j1-1 ; k0 = k1-1 ;
	im = in+1 ; jm = jn+1 ; km = kn+1 ;

	mx = im - i0 + 1 ;
	my = jm - j0 + 1 ;
	mz = km - k0 + 1 ;

	mxy  = mx*my     ;
	mxyz = mxy  *mz  ;

	a    = (double *) malloc(sizeof(double)*mxyz) ;
	anew = (double *) malloc(sizeof(double)*mxyz) ;

for ( k = k1 ; k <= kn ; k++) {
for ( j = j1 ; j <= jn ; j++) {
for ( i = i1 ; i <= in ; i++) {
			a(i  ,j  ,k  )   = 0.0 ;
	if (k ==  1)	a(i  ,j  ,k-1) = tzm ;
	if (k == nz)	a(i  ,j  ,k+1) = tzp ;
	if (j ==  1)	a(i  ,j-1,k  ) = tym ;
	if (j == ny)	a(i  ,j+1,k  ) = typ ;
	if (i ==  1)	a(i-1,j  ,k  ) = txm ;
	if (i == nx)	a(i+1,j  ,k  ) = txp ;
}}}

//t0 = clock();
//ts = clock();
t0 = omp_get_wtime();
ts = omp_get_wtime();

#pragma omp target data map(a[:mxyz],anew[:mxyz])
while ( error >= epsilon && istep <= nstep ) {

	istep = istep + 1 ;

	error = 0.0 ;
#pragma omp target teams distribute parallel for \
reduction(max:error) collapse(3) schedule(static,1)
for( k = k1; k <= kn; k++) {
for( j = j1; j <= jn; j++) {
for( i = i1; i <= in; i++) {
	anew(i,j,k) = ( a(i+1,j,k) + a(i,j+1,k) + a(i,j,k+1)
                      + a(i-1,j,k) + a(i,j-1,k) + a(i,j,k-1) ) / 6.0 ;
	error = fmax( error , fabs(anew(i,j,k)-a(i,j,k)) ) ;
}}}
#pragma omp target teams distribute parallel for \
collapse(3) schedule(static,1)
for( k = k1; k <= kn; k++) {
for( j = j1; j <= jn; j++) {
for( i = i1; i <= in; i++) {
	a(i,j,k) = anew(i,j,k) ;
}}}

if ( istep%nprint == 0 || error <= epsilon) {
#pragma omp target update from (a[:mxyz])
//	te = clock() ;
	te = omp_get_wtime();
	asum =  0.0 ;
	amax = -2.0 ;
	amin =  2.0 ;
	for ( k = k1 ; k <= kn ; k++ ) {
	for ( j = j1 ; j <= jn ; j++ ) {
	for ( i = i1 ; i <= in ; i++ ) {
		amin = fmin(amin,a(i,j,k)) ;
		amax = fmax(amax,a(i,j,k)) ;
		asum =      asum+a(i,j,k)  ;
	}}}
	asum = asum /(nx*ny*nz) ; 
//	printf ("%%heatx, istep %5d error %15.7f asum %15.7f amin %15.7f amax %15.7f time %10.3f \n",istep,error,asum,amin,amax,(te-ts) / (double) CLOCKS_PER_SEC ) ;
	printf ("%%heatx, istep %5d error %15.7f asum %15.7f amin %15.7f amax %15.7f time %10.3f \n",istep,error,asum,amin,amax,(te-ts) ) ;
	ts = te ;
}

}
//t1 = clock();
t1 = omp_get_wtime();

printf("%%heatx,                                           \n"          ) ;
//printf("%%heatx, total time %10.3f\n",(t1-t0) / (double) CLOCKS_PER_SEC ) ;
printf("%%heatx, total time %10.3f\n",(t1-t0) ) ;

printf("%%heatx, EOP\n") ;

return 0;
}

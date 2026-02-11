#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <mpi.h> //  MPI 

#define MAXVARS		(250)
#define EPSMIN		(1E-6)

extern void mds(double *startpoint, double *endpoint, int n, double *val, double eps, int maxfevals, int maxiter,
         double mu, double theta, double delta, int *ni, int *nf, double *xl, double *xr, int *term);

unsigned long funevals = 0;

double f(double *x, int n) {
    double fv;
    int i;
    funevals++;
    fv = 0.0;
    for (i=0; i<n-1; i++)
        fv = fv + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);
    usleep(100);
    return fv;
}

double get_wtime(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}

int main(int argc, char *argv[]) {
    int nvars = 4;
    int ntrials = 64;
    double lower[MAXVARS], upper[MAXVARS];
    double eps = EPSMIN, mu = 1.0, theta = 0.25, delta = 0.25;
    int maxfevals = 10000, maxiter = 10000;

    double startpt[MAXVARS], endpt[MAXVARS], fx;
    int nt, nf;

    double best_pt[MAXVARS], best_fx = 1e10;
    int best_trial = -1, best_nt = -1, best_nf = -1;

    int rank, size, i, trial;
    double t0, t1;

    // Αρχικοποίηση MPI 
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (i = 0; i < MAXVARS; i++) { lower[i] = -2.0; upper[i] = +2.0; } 

    t0 = get_wtime();

    // Μοίρασμα των trials στις διεργασίες 
    for (trial = rank; trial < ntrials; trial += size) {
        srand48(trial); 
        for (i = 0; i < nvars; i++)
            startpt[i] = lower[i] + (upper[i]-lower[i])*drand48();

        int term = -1;
        mds(startpt, endpt, nvars, &fx, eps, maxfevals, maxiter, mu, theta, delta,
            &nt, &nf, lower, upper, &term);

        if (fx < best_fx) {
            best_trial = trial;
            best_nt = nt;
            best_nf = nf;
            best_fx = fx;
            for (i = 0; i < nvars; i++) best_pt[i] = endpt[i];
        }
    }

    // Εύρεση του παγκόσμιου ελάχιστου (Global Reduction) 
    struct { double val; int rank; } local_res, global_res;
    local_res.val = best_fx;
    local_res.rank = rank;

    // Η MPI_MINLOC βρίσκει την ελάχιστη τιμή και ποιο rank την έχει 
    MPI_Reduce(&local_res, &global_res, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    // Συνολικός αριθμός funevals από όλες τις διεργασίες
    unsigned long total_funevals = 0;
    MPI_Reduce(&funevals, &total_funevals, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    t1 = get_wtime();

    if (rank == 0) {
        // Ο rank 0 λαμβάνει το καλύτερο σημείο από τη διεργασία που το βρήκε
        if (global_res.rank != 0) {
            MPI_Recv(best_pt, nvars, MPI_DOUBLE, global_res.rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&best_trial, 1, MPI_INT, global_res.rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&best_nt, 1, MPI_INT, global_res.rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&best_nf, 1, MPI_INT, global_res.rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        best_fx = global_res.val;

        printf("\n\nFINAL RESULTS (MPI):\n");
        printf("Elapsed time = %.3lf s\n", t1-t0);
        printf("Total number of trials = %d\n", ntrials);
        printf("Total number of function evaluations = %ld\n", total_funevals);
        printf("Best result at trial %d\n", best_trial);
        for (i = 0; i < nvars; i++) printf("x[%3d] = %15.7le \n", i, best_pt[i]);
        printf("f(x) = %15.7le\n", best_fx);
    } else {
        // Οι άλλες διεργασίες στέλνουν τα δεδομένα είναι οί νικητές
        if (rank == global_res.rank) {
            MPI_Send(best_pt, nvars, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&best_trial, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&best_nt, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
            MPI_Send(&best_nf, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
    return 0;

}

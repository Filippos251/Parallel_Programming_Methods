#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

extern double f(double *x, int n);

void initialize_simplex(double *u, int n, double *point, double delta) {
    int i, j;
    for (j = 0; j < n; j++) u[j] = point[j];
    for (i = 1; i < n + 1; i++) {
        for (j = 0; j < n; j++) {
            if (i - 1 == j) u[i * n + j] = point[j] + delta;
            else u[i * n + j] = point[j];
        }
    }
}

int minimum_simplex(double *fu, int n) {
    int i, imin = 0;
    double min = fu[0];
    for (i = 1; i < n + 1; i++) {
        if (min > fu[i]) {
            min = fu[i];
            imin = i;
        }
    }
    return imin;
}

double simplex_size(double *u, int n) {
    int i, j;
    double *mesos, dist, max_dist = -1;
    mesos = (double *) malloc((n + 1) * sizeof(double));
    for (j = 0; j < n; j++) {
        mesos[j] = 0.0;
        for (i = 0; i < n + 1; i++) mesos[j] += u[i * n + j];
        mesos[j] /= (n + 1);
    }
    for (i = 0; i < n + 1; i++) {
        dist = 0.0;
        for (j = 0; j < n; j++) dist += (mesos[j] - u[i * n + j]) * (mesos[j] - u[i * n + j]);
        dist = sqrt(dist);
        if (dist > max_dist) max_dist = dist;
    }
    free(mesos);
    return max_dist;
}

void swap_simplex(double *u, double *fu, int n, int from, int to) {
    int j;
    double *tmp = (double *) malloc(n * sizeof(double));
    double ftmp = fu[from];
    for (j = 0; j < n; j++) tmp[j] = u[from * n + j];
    for (j = 0; j < n; j++) u[from * n + j] = u[to * n + j];
    fu[from] = fu[to];
    for (j = 0; j < n; j++) u[to * n + j] = tmp[j];
    fu[to] = ftmp;
    free(tmp);
}

void assign_simplex(double *s1, double *fs1, double *s2, double *fs2, int n) {
    int i, j;
    for (i = 1; i < n + 1; i++) {
        for (j = 0; j < n; j++) s1[i * n + j] = s2[i * n + j];
        fs1[i] = fs2[i];
    }
}

void mds(double *point, double *endpoint, int n, double *val, double eps, int maxfevals, int maxiter, double mu,
        double theta, double delta, int *nit, int *nf, double *xl, double *xr, int *term) {
    int i, j, k, found_better, iter = 0, kec, terminate = 0, out_of_bounds;
    double *u, *r, *ec, *fu, *fr, *fec;

    u = (double *) malloc(n * (n + 1) * sizeof(double));
    r = (double *) malloc(n * (n + 1) * sizeof(double));
    ec = (double *) malloc(n * (n + 1) * sizeof(double));
    fu = (double *) malloc((n + 1) * sizeof(double));
    fr = (double *) malloc((n + 1) * sizeof(double));
    fec = (double *) malloc((n + 1) * sizeof(double));

    *nf = 0;
    initialize_simplex(u, n, point, delta);

    // Initial Simplex Evaluation (Parallel Tasks)
    for (i = 0; i < n + 1; i++) {
        #pragma omp task firstprivate(i) shared(u, fu, nf)
        {
            fu[i] = f(&u[i * n], n);
            #pragma omp atomic
            (*nf)++;
        }
    }
    #pragma omp taskwait

    while (terminate == 0 && iter < maxiter) {
        k = minimum_simplex(fu, n);
        swap_simplex(u, fu, n, k, 0);

        found_better = 0;
        while (found_better == 0) {
            if (*nf > maxfevals) { *term = 1; terminate = 1; break; }
            if (simplex_size(u, n) < eps) { *term = 2; terminate = 1; break; }

            fr[0] = fu[0];
            found_better = 1;
            for (i = 1; i < n + 1; i++) {
                for (j = 0; j < n; j++) {
                    r[i * n + j] = u[0 * n + j] - (u[i * n + j] - u[0 * n + j]);
                    if (r[i * n + j] > xr[j] || r[i * n + j] < xl[j]) { found_better = 0; break; }
                }
                if (found_better == 0) break;
            }

            if (found_better == 1) { // Rotation Step
                for (i = 1; i < n + 1; i++) {
                    #pragma omp task firstprivate(i) shared(r, u, fr, nf)
                    {
                        for (int jj = 0; jj < n; jj++)
                            r[i * n + jj] = u[0 * n + jj] - (u[i * n + jj] - u[0 * n + jj]);
                        fr[i] = f(&r[i * n], n);
                        #pragma omp atomic
                        (*nf)++;
                    }
                }
                #pragma omp taskwait
                found_better = 0;
                k = minimum_simplex(fr, n);
                if (fr[k] < fu[0]) found_better = 1;
            }

            if (found_better == 1) { // Expand
                out_of_bounds = 0;
                for (i = 1; i < n + 1; i++) {
                    for (j = 0; j < n; j++) {
                        ec[i * n + j] = u[0 * n + j] - mu * ((u[i * n + j] - u[0 * n + j]));
                        if (ec[i * n + j] > xr[j] || ec[i * n + j] < xl[j]) { out_of_bounds = 1; break; }
                    }
                    if (out_of_bounds == 1) break;
                }
                if (out_of_bounds == 0) {
                    fec[0] = fu[0];
                    for (i = 1; i < n + 1; i++) {
                        #pragma omp task firstprivate(i) shared(ec, u, fec, nf)
                        {
                            for (int jj = 0; jj < n; jj++)
                                ec[i * n + jj] = u[0 * n + jj] - mu * ((u[i * n + jj] - u[0 * n + jj]));
                            fec[i] = f(&ec[i * n], n);
                            #pragma omp atomic
                            (*nf)++;
                        }
                    }
                    #pragma omp taskwait
                    kec = minimum_simplex(fec, n);
                    if (fec[kec] < fr[k]) assign_simplex(u, fu, ec, fec, n);
                    else assign_simplex(u, fu, r, fr, n);
                } else assign_simplex(u, fu, r, fr, n);
            } else { // Contract
                fec[0] = fu[0];
                for (i = 1; i < n + 1; i++) {
                    #pragma omp task firstprivate(i) shared(ec, u, fec, nf)
                    {
                        for (int jj = 0; jj < n; jj++)
                            ec[i * n + jj] = u[0 * n + jj] + theta * ((u[i * n + jj] - u[0 * n + jj]));
                        fec[i] = f(&ec[i * n], n);
                        #pragma omp atomic
                        (*nf)++;
                    }
                }
                #pragma omp taskwait
                kec = minimum_simplex(fec, n);
                if (fec[kec] < fu[0]) found_better = 1;
                assign_simplex(u, fu, ec, fec, n);
            }
        }
        iter++;
        if (iter == maxiter) *term = 3;
    }

    k = minimum_simplex(fu, n);
    swap_simplex(u, fu, n, k, 0);
    for (i = 0; i < n; i++) endpoint[i] = u[i];
    *val = fu[0];
    *nit = iter;

    free(u); free(r); free(ec); free(fu); free(fr); free(fec);
}







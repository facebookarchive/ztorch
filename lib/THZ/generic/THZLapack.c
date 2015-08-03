/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZLapack.c"
#else


THZ_EXTERNC void zgesv_(int *n, int *nrhs, double complex *a, int *lda, int *ipiv, double complex *b, int *ldb, int *info);
THZ_EXTERNC void cgesv_(int *n, int *nrhs, float complex *a, int *lda, int *ipiv, float complex *b, int *ldb, int *info);
THZ_EXTERNC void zgels_(char *trans, int *m, int *n, int *nrhs, double complex *a, int *lda, double complex *b, int *ldb, double complex *work, int *lwork, int *info);
THZ_EXTERNC void cgels_(char *trans, int *m, int *n, int *nrhs, float complex *a, int *lda, float complex *b, int *ldb, float complex *work, int *lwork, int *info);
// THZ_EXTERNC void zsyev_(char *jobz, char *uplo, int *n, double complex *a, int *lda, double complex *w, double complex *work, int *lwork, int *info);
// THZ_EXTERNC void csyev_(char *jobz, char *uplo, int *n, float complex *a, int *lda, float complex *w, float complex *work, int *lwork, int *info);
THZ_EXTERNC void zgeev_(char *jobvl, char *jobvr, int *n, double complex *a, int *lda, double complex *wr, double complex *wi, double complex* vl, int *ldvl, double complex *vr, int *ldvr, double complex *work, int *lwork, int *info);
THZ_EXTERNC void cgeev_(char *jobvl, char *jobvr, int *n, float complex *a, int *lda, float complex *wr, float complex *wi, float complex* vl, int *ldvl, float complex *vr, int *ldvr, float complex *work, int *lwork, int *info);
THZ_EXTERNC void zgesvd_(char *jobu, char *jobvt, int *m, int *n, double complex *a, int *lda, double complex *s, double complex *u, int *ldu, double complex *vt, int *ldvt, double complex *work, int *lwork, int *info);
THZ_EXTERNC void cgesvd_(char *jobu, char *jobvt, int *m, int *n, float complex *a, int *lda, float complex *s, float complex *u, int *ldu, float complex *vt, int *ldvt, float complex *work, int *lwork, int *info);
THZ_EXTERNC void zgetrf_(int *m, int *n, double complex *a, int *lda, int *ipiv, int *info);
THZ_EXTERNC void cgetrf_(int *m, int *n, float complex *a, int *lda, int *ipiv, int *info);
THZ_EXTERNC void zgetri_(int *n, double complex *a, int *lda, int *ipiv, double complex *work, int *lwork, int *info);
THZ_EXTERNC void cgetri_(int *n, float complex *a, int *lda, int *ipiv, float complex *work, int *lwork, int *info);
THZ_EXTERNC void zpotrf_(char *uplo, int *n, double complex *a, int *lda, int *info);
THZ_EXTERNC void cpotrf_(char *uplo, int *n, float complex *a, int *lda, int *info);
THZ_EXTERNC void zpotri_(char *uplo, int *n, double complex *a, int *lda, int *info);
THZ_EXTERNC void cpotri_(char *uplo, int *n, float complex *a, int *lda, int *info);
THZ_EXTERNC void zpotrs_(char *uplo, int *n, int *nrhs, double complex *a, int *lda, double complex *b, int *ldb, int *info);
THZ_EXTERNC void cpotrs_(char *uplo, int *n, int *nrhs, float complex *a, int *lda, float complex *b, int *ldb, int *info);


void THZLapack_(gesv)(int n, int nrhs, real *a, int lda, int *ipiv, real *b, int ldb, int* info)
{
#ifdef USE_LAPACK
#if defined(THZ_REAL_IS_DOUBLE)
  zgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#else
  cgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#endif
#else
  THError("gesv : Lapack library not found in compile time\n");
#endif
  return;
}

void THZLapack_(gels)(char trans, int m, int n, int nrhs, real *a, int lda, real *b, int ldb, real *work, int lwork, int *info)
{
#ifdef USE_LAPACK
#if defined(THZ_REAL_IS_DOUBLE)
  zgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#else
  cgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#endif
#else
  THError("gels : Lapack library not found in compile time\n");
#endif
}

void THZLapack_(syev)(char jobz, char uplo, int n, real *a, int lda, real *w, real *work, int lwork, int *info)
{
  /*
#ifdef USE_LAPACK
#if defined(THZ_REAL_IS_DOUBLE)
  zsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
#else
  csyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
#endif
#else
  THError("syev : Lapack library not found in compile time\n");
#endif
  */
  THError("syev : Not defined for complex tensors\n");
}

void THZLapack_(geev)(char jobvl, char jobvr, int n, real *a, int lda, real *wr, real *wi, real* vl, int ldvl, real *vr, int ldvr, real *work, int lwork, int *info)
{
#ifdef USE_LAPACK
#if defined(THZ_REAL_IS_DOUBLE)
  zgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
#else
  cgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
#endif
#else
  THError("geev : Lapack library not found in compile time\n");
#endif
}

void THZLapack_(gesvd)(char jobu, char jobvt, int m, int n, real *a, int lda, real *s, real *u, int ldu, real *vt, int ldvt, real *work, int lwork, int *info)
{
#ifdef USE_LAPACK
#if defined(THZ_REAL_IS_DOUBLE)
  zgesvd_( &jobu,  &jobvt,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  info);
#else
  cgesvd_( &jobu,  &jobvt,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  info);
#endif
#else
  THError("gesvd : Lapack library not found in compile time\n");
#endif
}

/* LU decomposition */
void THZLapack_(getrf)(int m, int n, real *a, int lda, int *ipiv, int *info)
{
#ifdef  USE_LAPACK
#if defined(THZ_REAL_IS_DOUBLE)
  zgetrf_(&m, &n, a, &lda, ipiv, info);
#else
  cgetrf_(&m, &n, a, &lda, ipiv, info);
#endif
#else
  THError("getrf : Lapack library not found in compile time\n");
#endif
}
/* Matrix Inverse */
void THZLapack_(getri)(int n, real *a, int lda, int *ipiv, real *work, int lwork, int* info)
{
#ifdef  USE_LAPACK
#if defined(THZ_REAL_IS_DOUBLE)
  zgetri_(&n, a, &lda, ipiv, work, &lwork, info);
#else
  cgetri_(&n, a, &lda, ipiv, work, &lwork, info);
#endif
#else
  THError("getri : Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization */
void THZLapack_(potrf)(char uplo, int n, real *a, int lda, int *info)
{
#ifdef  USE_LAPACK
#if defined(THZ_REAL_IS_DOUBLE)
  zpotrf_(&uplo, &n, a, &lda, info);
#else
  cpotrf_(&uplo, &n, a, &lda, info);
#endif
#else
  THError("potrf : Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization based Matrix Inverse */
void THZLapack_(potri)(char uplo, int n, real *a, int lda, int *info)
{
#ifdef  USE_LAPACK
#if defined(THZ_REAL_IS_DOUBLE)
  zpotri_(&uplo, &n, a, &lda, info);
#else
  cpotri_(&uplo, &n, a, &lda, info);
#endif
#else
  THError("potri: Lapack library not found in compile time\n");
#endif
}

/* Solve A*X = B with a symmetric positive definite matrix A using the Cholesky factorization */
void THZLapack_(potrs)(char uplo, int n, int nrhs, real *a, int lda, real *b, int ldb, int *info)
{
#ifdef  USE_LAPACK
#if defined(THZ_REAL_IS_DOUBLE)
  zpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
#else
  cpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
#endif
#else
  THError("potrs: Lapack library not found in compile time\n");
#endif
}

#endif

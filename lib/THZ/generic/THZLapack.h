/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZLapack.h"
#else

/* AX=B */
THZ_API void THZLapack_(gesv)(int n, int nrhs, real *a, int lda, int *ipiv, real *b, int ldb, int* info);
/* ||AX-B|| */
THZ_API void THZLapack_(gels)(char trans, int m, int n, int nrhs, real *a, int lda, real *b, int ldb, real *work, int lwork, int *info);
/* Eigenvals */
THZ_API void THZLapack_(syev)(char jobz, char uplo, int n, real *a, int lda, real *w, real *work, int lwork, int *info);
/* Non-sym eigenvals */
THZ_API void THZLapack_(geev)(char jobvl, char jobvr, int n, real *a, int lda, real *wr, real *wi, real* vl, int ldvl, real *vr, int ldvr, real *work, int lwork, int *info);
/* svd */
THZ_API void THZLapack_(gesvd)(char jobu, char jobvt, int m, int n, real *a, int lda, real *s, real *u, int ldu, real *vt, int ldvt, real *work, int lwork, int *info);
/* LU decomposition */
THZ_API void THZLapack_(getrf)(int m, int n, real *a, int lda, int *ipiv, int *info);
/* Matrix Inverse */
THZ_API void THZLapack_(getri)(int n, real *a, int lda, int *ipiv, real *work, int lwork, int* info);

/* Positive Definite matrices */
/* Cholesky factorization */
void THZLapack_(potrf)(char uplo, int n, real *a, int lda, int *info);
/* Matrix inverse based on Cholesky factorization */
void THZLapack_(potri)(char uplo, int n, real *a, int lda, int *info);
/* Solve A*X = B with a symmetric positive definite matrix A using the Cholesky factorization */
void THZLapack_(potrs)(char uplo, int n, int nrhs, real *a, int lda, real *b, int ldb, int *info);

#endif

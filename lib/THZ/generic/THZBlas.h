/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZBlas.h"
#else

/* Level 1 */
THZ_API void THZBlas_(swap)(long n, real *x, long incx, real *y, long incy);
THZ_API void THZBlas_(scal)(long n, real a, real *x, long incx);
THZ_API void THZBlas_(copy)(long n, real *x, long incx, real *y, long incy);
THZ_API void THZBlas_(axpy)(long n, real a, real *x, long incx, real *y, long incy);
THZ_API real THZBlas_(dot)(long n, real *x, long incx, real *y, long incy);

/* Level 2 */
THZ_API void THZBlas_(gemv)(char trans, long m, long n, real alpha, real *a, long lda, real *x, long incx, real beta, real *y, long incy);
THZ_API void THZBlas_(gerc)(long m, long n, real alpha, real *x, long incx, real *y, long incy, real *a, long lda);
THZ_API void THZBlas_(geru)(long m, long n, real alpha, real *x, long incx, real *y, long incy, real *a, long lda);

/* Level 3 */
THZ_API void THZBlas_(gemm)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc);

#endif

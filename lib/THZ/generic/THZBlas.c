/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZBlas.c"
#else

THZ_EXTERNC void zswap_(int *n, double complex *x, int *incx, double complex *y, int *incy);
THZ_EXTERNC void cswap_(int *n, float complex *x, int *incx, float complex *y, int *incy);
THZ_EXTERNC void zscal_(int *n, double complex *a, double complex *x, int *incx);
THZ_EXTERNC void cscal_(int *n, float complex *a, float complex *x, int *incx);
THZ_EXTERNC void zcopy_(int *n, double complex *x, int *incx, double complex *y, int *incy);
THZ_EXTERNC void ccopy_(int *n, float complex *x, int *incx, float complex *y, int *incy);
THZ_EXTERNC void zaxpy_(int *n, double complex *a, double complex *x, int *incx, double complex *y, int *incy);
THZ_EXTERNC void caxpy_(int *n, float complex *a, float complex *x, int *incx, float complex *y, int *incy);
THZ_EXTERNC void zgemv_(char *trans, int *m, int *n, double complex *alpha, double complex *a, int *lda, double complex *x, int *incx, double complex *beta, double complex *y, int *incy);
THZ_EXTERNC void cgemv_(char *trans, int *m, int *n, float complex *alpha, float complex *a, int *lda, float complex *x, int *incx, float complex *beta, float complex *y, int *incy);
THZ_EXTERNC void zger_(int *m, int *n, double complex *alpha, double complex *x, int *incx, double complex *y, int *incy, double complex *a, int *lda);
THZ_EXTERNC void cgerc_(int *m, int *n, float complex *alpha, float complex *x, int *incx, float complex *y, int *incy, float complex *a, int *lda);
THZ_EXTERNC void zgemm_(char *transa, char *transb, int *m, int *n, int *k, double complex *alpha, double complex *a, int *lda, double complex *b, int *ldb, double complex *beta, double complex *c, int *ldc);
THZ_EXTERNC void cgemm_(char *transa, char *transb, int *m, int *n, int *k, float complex *alpha, float complex *a, int *lda, float complex *b, int *ldb, float complex *beta, float complex *c, int *ldc);



void THZBlas_(swap)(long n, real *x, long incx, real *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(THZ_REAL_IS_DOUBLE) || defined(THZ_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(THZ_REAL_IS_DOUBLE)
    zswap_(&i_n, x, &i_incx, y, &i_incy);
#else
    cswap_(&i_n, x, &i_incx, y, &i_incy);
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < n; i++)
    {
      real z = x[i*incx];
      x[i*incx] = y[i*incy];
      y[i*incy] = z;
    }
  }
}

void THZBlas_(scal)(long n, real a, real *x, long incx)
{
  if(n == 1)
    incx = 1;

#if defined(USE_BLAS) && (defined(THZ_REAL_IS_DOUBLE) || defined(THZ_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;

#if defined(THZ_REAL_IS_DOUBLE)
    zscal_(&i_n, &a, x, &i_incx);
#else
    cscal_(&i_n, &a, x, &i_incx);
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < n; i++)
      x[i*incx] *= a;
  }
}

void THZBlas_(copy)(long n, real *x, long incx, real *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(THZ_REAL_IS_DOUBLE) || defined(THZ_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(THZ_REAL_IS_DOUBLE)
    zcopy_(&i_n, x, &i_incx, y, &i_incy);
#else
    ccopy_(&i_n, x, &i_incx, y, &i_incy);
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < n; i++)
      y[i*incy] = x[i*incx];
  }
}

void THZBlas_(axpy)(long n, real a, real *x, long incx, real *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(THZ_REAL_IS_DOUBLE) || defined(THZ_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(THZ_REAL_IS_DOUBLE)
    zaxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
#else
    caxpy_(&i_n, &a, x, &i_incx, y, &i_incy);
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < n; i++)
      y[i*incy] += a*x[i*incx];
  }
}

real THZBlas_(dot)(long n, real *x, long incx, real *y, long incy)
{
  if(n == 1)
  {
    incx = 1;
    incy = 1;
  }

#if defined(USE_BLAS) && (defined(THZ_REAL_IS_DOUBLE) || defined(THZ_REAL_IS_FLOAT))
  if( (n <= INT_MAX) && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_n = (int)n;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

    real result;
#if defined(THZ_REAL_IS_DOUBLE)
    cblas_zdotc_sub(i_n, x, i_incx, y, i_incy, &result);
#else
    cblas_cdotc_sub(i_n, x, i_incx, y, i_incy, &result);
#endif
    return result;
  }
#endif
  {
    long i;
    real sum = 0;
    for(i = 0; i < n; i++) {
      sum += CONJ(x[i*incx])*y[i*incy];
    }
    return sum;
  }
}

void THZBlas_(gemv)(char trans, long m, long n, real alpha, real *a, long lda, real *x, long incx, real beta, real *y, long incy)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && (defined(THZ_REAL_IS_DOUBLE) || defined(THZ_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) &&
      (lda > 0) && (lda <= INT_MAX) &&
      (incx > 0) && (incx <= INT_MAX) &&
      (incy > 0) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(THZ_REAL_IS_DOUBLE)
    zgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
#else
    cgemv_(&trans, &i_m, &i_n, &alpha, a, &i_lda, x, &i_incx, &beta, y, &i_incy);
#endif
    return;
  }
#endif
  {
    long i, j;

    if( (trans == 'T') || (trans == 't') )
    {
      for(i = 0; i < n; i++)
      {
        real sum = 0;
        real *row_ = a+lda*i;
        for(j = 0; j < m; j++)
          sum += x[j*incx]*row_[j];
        y[i*incy] = beta*y[i*incy] + alpha*sum;
      }
    }
    else
    {
      if(beta != 1)
        THZBlas_(scal)(m, beta, y, incy);

      for(j = 0; j < n; j++)
      {
        real *column_ = a+lda*j;
        real z = alpha*x[j*incx];
        for(i = 0; i < m; i++)
          y[i*incy] += z*column_[i];
      }
    }
  }
}

void THZBlas_(gerc)(long m, long n, real alpha, real *x, long incx, real *y, long incy, real *a, long lda)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && (defined(THZ_REAL_IS_DOUBLE) || defined(THZ_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(THZ_REAL_IS_DOUBLE)
    zgerc_(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
#else
    cgerc_(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
#endif
    return;
  }
#endif
  {
    long i, j;
    for(j = 0; j < n; j++)
    {
      real *column_ = a+j*lda;
      real z = alpha*CONJ(y[j*incy]);
      for(i = 0; i < m; i++)
        column_[i] += z*x[i*incx];
    }
  }
}

void THZBlas_(geru)(long m, long n, real alpha, real *x, long incx, real *y, long incy, real *a, long lda)
{
  if(n == 1)
    lda = m;

#if defined(USE_BLAS) && (defined(THZ_REAL_IS_DOUBLE) || defined(THZ_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) && (lda <= INT_MAX)  && (incx <= INT_MAX) && (incy <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_lda = (int)lda;
    int i_incx = (int)incx;
    int i_incy = (int)incy;

#if defined(THZ_REAL_IS_DOUBLE)
    zgeru_(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
#else
    cgeru_(&i_m, &i_n, &alpha, x, &i_incx, y, &i_incy, a, &i_lda);
#endif
    return;
  }
#endif
  {
    long i, j;
    for(j = 0; j < n; j++)
    {
      real *column_ = a+j*lda;
      real z = alpha*y[j*incy];
      for(i = 0; i < m; i++)
        column_[i] += z*x[i*incx];
    }
  }
}

void THZBlas_(gemm)(char transa, char transb, long m, long n, long k, real alpha, real *a, long lda, real *b, long ldb, real beta, real *c, long ldc)
{
  int transa_ = ((transa == 't') || (transa == 'T'));
  int transb_ = ((transb == 't') || (transb == 'T'));

  if(n == 1)
    ldc = m;

  if(transa_)
  {
    if(m == 1)
      lda = k;
  }
  else
  {
    if(k == 1)
      lda = m;
  }

  if(transb_)
  {
    if(k == 1)
      ldb = n;
  }
  else
  {
    if(n == 1)
      ldb = k;
  }

#if defined(USE_BLAS) && (defined(THZ_REAL_IS_DOUBLE) || defined(THZ_REAL_IS_FLOAT))
  if( (m <= INT_MAX) && (n <= INT_MAX) && (k <= INT_MAX) && (lda <= INT_MAX)  && (ldb <= INT_MAX) && (ldc <= INT_MAX) )
  {
    int i_m = (int)m;
    int i_n = (int)n;
    int i_k = (int)k;
    int i_lda = (int)lda;
    int i_ldb = (int)ldb;
    int i_ldc = (int)ldc;

#if defined(THZ_REAL_IS_DOUBLE)
    zgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
#else
    cgemm_(&transa, &transb, &i_m, &i_n, &i_k, &alpha, a, &i_lda, b, &i_ldb, &beta, c, &i_ldc);
#endif
    return;
  }
#endif
  {
    long i, j, l;
    if(!transa_ && !transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l];
          b_ += ldb;
          c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else if(transa_ && !transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l];
          b_ += ldb;
          c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
    else if(!transa_ && transb_)
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l*lda]*b_[l*ldb];
          b_++;
          c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_++;
      }
    }
    else
    {
      real *a_ = a;
      for(i = 0; i < m; i++)
      {
        real *b_ = b;
        for(j = 0; j < n; j++)
        {
          real sum = 0;
          for(l = 0; l < k; l++)
            sum += a_[l]*b_[l*ldb];
          b_++;
          c[j*ldc+i] = beta*c[j*ldc+i]+alpha*sum;
        }
        a_ += lda;
      }
    }
  }
}

#endif

/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorLapack.c"
#else

static int THZTensor_(lapackClone)(THZTensor *r_, THZTensor *m, int forced)
{
  int clone;

  if (!forced && m->stride[0] == 1 && m->stride[1] == m->size[0])
  {
    clone = 0;
    THZTensor_(set)(r_,m);
  }
  else
  {
    clone = 1;
    /* we need to copy */
    THZTensor_(resize2d)(r_,m->size[1],m->size[0]);
    THZTensor_(transpose)(r_,NULL,0,1);
    THZTensor_(copy)(r_,m);
  }
  return clone;
}

THZ_API void THZTensor_(gesv)(THZTensor *rb_, THZTensor *ra_, THZTensor *b, THZTensor *a)
{
  int n, nrhs, lda, ldb, info;
  THIntTensor *ipiv;
  THZTensor *ra__;
  THZTensor *rb__;

  int clonea;
  int cloneb;
  int destroya;
  int destroyb;


  if (a == NULL || ra_ == a) /* possibly destroy the inputs  */
  {
    ra__ = THZTensor_(new)();
    clonea = THZTensor_(lapackClone)(ra__,ra_,0);
    destroya = 1;
  }
  else /*we want to definitely clone and use ra_ and rb_ as computational space*/
  {
    clonea = THZTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroya = 0;
  }
  if (b == NULL || rb_ == b) /* possibly destroy the inputs  */
  {
    rb__ = THZTensor_(new)();
    cloneb = THZTensor_(lapackClone)(rb__,rb_,0);
    destroyb = 1;
  }
  else /*we want to definitely clone and use ra_ and rb_ as computational space*/
  {
    cloneb = THZTensor_(lapackClone)(rb_,b,1);
    rb__ = rb_;
    destroyb = 0;
  }

  THArgCheck(ra__->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(rb__->nDimension == 2, 2, "b should be 2 dimensional");
  THArgCheck(ra__->size[0] == ra__->size[1], 1, "A should be square");
  THArgCheck(rb__->size[0] == ra__->size[0], 2, "A,b size incomptable");

  n    = (int)ra__->size[0];
  nrhs = (int)rb__->size[1];
  lda  = n;
  ldb  = n;

  ipiv = THIntTensor_newWithSize1d((long)n);
  THZLapack_(gesv)(n, nrhs,
		  THZTensor_(data)(ra__), lda, THIntTensor_data(ipiv),
		  THZTensor_(data)(rb__), ldb, &info);

  /* clean up */
  if (destroya)
  {
    if (clonea)
    {
      THZTensor_(copy)(ra_,ra__);
    }
    THZTensor_(free)(ra__);
  }
  if (destroyb)
  {
    if (cloneb)
    {
      THZTensor_(copy)(rb_,rb__);
    }
    THZTensor_(free)(rb__);
  }

  if (info < 0)
  {
    THError("Lapack gesv : Argument %d : illegal value", -info);
  }
  else if (info > 0)
  {
    THError("Lapack gesv : U(%d,%d) is zero, singular U.", info,info);
  }

  THIntTensor_free(ipiv);
}

THZ_API void THZTensor_(gels)(THZTensor *rb_, THZTensor *ra_, THZTensor *b, THZTensor *a)
{
  int m, n, nrhs, lda, ldb, info, lwork;
  THZTensor *work = NULL;
  real wkopt = 0;

  THZTensor *ra__;
  THZTensor *rb__;

  int clonea;
  int cloneb;
  int destroya;
  int destroyb;


  if (a == NULL || ra_ == a) /* possibly destroy the inputs  */
  {
    ra__ = THZTensor_(new)();
    clonea = THZTensor_(lapackClone)(ra__,ra_,0);
    destroya = 1;
  }
  else /*we want to definitely clone and use ra_ and rb_ as computational space*/
  {
    clonea = THZTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroya = 0;
  }
  if (b == NULL || rb_ == b) /* possibly destroy the inputs  */
  {
    rb__ = THZTensor_(new)();
    cloneb = THZTensor_(lapackClone)(rb__,rb_,0);
    destroyb = 1;
  }
  else /*we want to definitely clone and use ra_ and rb_ as computational space*/
  {
    cloneb = THZTensor_(lapackClone)(rb_,b,1);
    rb__ = rb_;
    destroyb = 0;
  }

  THArgCheck(ra__->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(ra_->size[0] == rb__->size[0], 2, "size incompatible A,b");

  m = ra__->size[0];
  n = ra__->size[1];
  nrhs = rb__->size[1];
  lda = m;
  ldb = m;
  info = 0;

  /* get optimal workspace size */
  THZLapack_(gels)('N', m, n, nrhs, THZTensor_(data)(ra__), lda,
		  THZTensor_(data)(rb__), ldb,
		  &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THZTensor_(newWithSize1d)(lwork);
  THZLapack_(gels)('N', m, n, nrhs, THZTensor_(data)(ra__), lda,
		  THZTensor_(data)(rb__), ldb,
		  THZTensor_(data)(work), lwork, &info);

  /* printf("lwork = %d,%g\n",lwork,THZTensor_(data)(work)[0]); */
  if (info != 0)
  {
    THError("Lapack gels : Argument %d : illegal value", -info);
  }
  /* clean up */
  if (destroya)
  {
    if (clonea)
    {
      THZTensor_(copy)(ra_,ra__);
    }
    THZTensor_(free)(ra__);
  }
  if (destroyb)
  {
    if (cloneb)
    {
      THZTensor_(copy)(rb_,rb__);
    }
    THZTensor_(free)(rb__);
  }
  THZTensor_(free)(work);
}

THZ_API void THZTensor_(geev)(THZTensor *re_, THZTensor *rv_, THZTensor *a_, const char *jobvr)
{
  int n, lda, lwork, info, ldvr;
  THZTensor *work, *wi, *wr, *a;
  real wkopt;
  real *rv_data;
  long i;

  THArgCheck(a_->nDimension == 2, 3, "A should be 2 dimensional");
  THArgCheck(a_->size[0] == a_->size[1], 3,"A should be square");

  /* we want to definitely clone */
  a = THZTensor_(new)();
  THZTensor_(lapackClone)(a,a_,1);

  n = a->size[0];
  lda = n;

  wi = THZTensor_(new)();
  wr = THZTensor_(new)();
  THZTensor_(resize2d)(re_,n,2);
  THZTensor_(resize1d)(wi,n);
  THZTensor_(resize1d)(wr,n);

  rv_data = NULL;
  ldvr = 1;
  if (*jobvr == 'V')
  {
    THZTensor_(resize2d)(rv_,n,n);
    rv_data = THZTensor_(data)(rv_);
    ldvr = n;
  }
  /* get optimal workspace size */
  THZLapack_(geev)('N', jobvr[0], n, THZTensor_(data)(a), lda, THZTensor_(data)(wr), THZTensor_(data)(wi),
      NULL, 1, rv_data, ldvr, &wkopt, -1, &info);

  lwork = (int)wkopt;
  work = THZTensor_(newWithSize1d)(lwork);

  THZLapack_(geev)('N', jobvr[0], n, THZTensor_(data)(a), lda, THZTensor_(data)(wr), THZTensor_(data)(wi),
      NULL, 1, rv_data, ldvr, THZTensor_(data)(work), lwork, &info);

  if (info > 0)
  {
    THError(" Lapack geev : Failed to converge. %d off-diagonal elements of an didn't converge to zero",info);
  }
  else if (info < 0)
  {
    THError("Lapack geev : Argument %d : illegal value", -info);
  }

  {
    real *re_data = THZTensor_(data)(re_);
    real *wi_data = THZTensor_(data)(wi);
    real *wr_data = THZTensor_(data)(wr);
    for (i=0; i<n; i++)
    {
      re_data[2*i] = wr_data[i];
      re_data[2*i+1] = wi_data[i];
    }
  }
  if (*jobvr == 'V')
  {
    THZTensor_(transpose)(rv_,NULL,0,1);
  }
  THZTensor_(free)(a);
  THZTensor_(free)(wi);
  THZTensor_(free)(wr);
  THZTensor_(free)(work);
}

THZ_API void THZTensor_(syev)(THZTensor *re_, THZTensor *rv_, THZTensor *a, const char *jobz, const char *uplo)
{
  int n, lda, lwork, info;
  THZTensor *work;
  real wkopt;

  THZTensor *rv__;

  int clonea;
  int destroy;

  if (a == NULL) /* possibly destroy the inputs  */
  {
    rv__ = THZTensor_(new)();
    clonea = THZTensor_(lapackClone)(rv__,rv_,0);
    destroy = 1;
  }
  else /*we want to definitely clone and use ra_ and rb_ as computational space*/
  {
    clonea = THZTensor_(lapackClone)(rv_,a,1);
    rv__ = rv_;
    destroy = 0;
  }

  THArgCheck(rv__->nDimension == 2, 2, "A should be 2 dimensional");

  n = rv__->size[0];
  lda = n;

  THZTensor_(resize1d)(re_,n);

  /* get optimal workspace size */
  THZLapack_(syev)(jobz[0], uplo[0], n, THZTensor_(data)(rv__), lda,
		  THZTensor_(data)(re_), &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THZTensor_(newWithSize1d)(lwork);
  THZLapack_(syev)(jobz[0], uplo[0], n, THZTensor_(data)(rv__), lda,
		  THZTensor_(data)(re_), THZTensor_(data)(work), lwork, &info);

  if (info > 0)
  {
    THError(" Lapack syev : Failed to converge. %d off-diagonal elements of an didn't converge to zero",info);
  }
  else if (info < 0)
  {
    THError("Lapack syev : Argument %d : illegal value", -info);
  }
  /* clean up */
  if (destroy)
  {
    if (clonea)
    {
      THZTensor_(copy)(rv_,rv__);
    }
    THZTensor_(free)(rv__);
  }
  THZTensor_(free)(work);
}

THZ_API void THZTensor_(gesvd)(THZTensor *ru_, THZTensor *rs_, THZTensor *rv_, THZTensor *a, const char* jobu)
{
  THZTensor *ra_ = THZTensor_(new)();
  THZTensor_(gesvd2)(ru_, rs_, rv_,  ra_, a, jobu);
  THZTensor_(free)(ra_);
}

THZ_API void THZTensor_(gesvd2)(THZTensor *ru_, THZTensor *rs_, THZTensor *rv_, THZTensor *ra_, THZTensor *a, const char* jobu)
{
  int k,m, n, lda, ldu, ldvt, lwork, info;
  THZTensor *work;
  real wkopt;

  THZTensor *ra__;

  int clonea;
  int destroy;

  if (a == NULL) /* possibly destroy the inputs  */
  {
    ra__ = THZTensor_(new)();
    clonea = THZTensor_(lapackClone)(ra__,ra_,0);
    destroy = 1;
  }
  else /*we want to definitely clone */
  {
    clonea = THZTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroy = 0;
  }

  THArgCheck(ra__->nDimension == 2, 2, "A should be 2 dimensional");

  m = ra__->size[0];
  n = ra__->size[1];
  k = (m < n ? m : n);

  lda = m;
  ldu = m;
  ldvt = n;
  THZTensor_(resize1d)(rs_,k);
  THZTensor_(resize2d)(rv_,ldvt,n);
  if (*jobu == 'A')
  {
    THZTensor_(resize2d)(ru_,m,ldu);
  }
  else
  {
    THZTensor_(resize2d)(ru_,k,ldu);
  }
  THZTensor_(transpose)(ru_,NULL,0,1);
  /* we want to return V not VT*/
  /*THZTensor_(transpose)(rv_,NULL,0,1);*/

  THZLapack_(gesvd)(jobu[0],jobu[0],
		   m,n,THZTensor_(data)(ra__),lda,
		   THZTensor_(data)(rs_),
		   THZTensor_(data)(ru_),
		   ldu,
		   THZTensor_(data)(rv_), ldvt,
		   &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THZTensor_(newWithSize1d)(lwork);
  THZLapack_(gesvd)(jobu[0],jobu[0],
		   m,n,THZTensor_(data)(ra__),lda,
		   THZTensor_(data)(rs_),
		   THZTensor_(data)(ru_),
		   ldu,
		   THZTensor_(data)(rv_), ldvt,
		   THZTensor_(data)(work),lwork, &info);
  if (info > 0)
  {
    THError(" Lapack gesvd : %d superdiagonals failed to converge.",info);
  }
  else if (info < 0)
  {
    THError("Lapack gesvd : Argument %d : illegal value", -info);
  }

  /* clean up */
  if (destroy)
  {
    if (clonea)
    {
      THZTensor_(copy)(ra_,ra__);
    }
    THZTensor_(free)(ra__);
  }
  THZTensor_(free)(work);
}

THZ_API void THZTensor_(getri)(THZTensor *ra_, THZTensor *a)
{
  int m, n, lda, info, lwork;
  real wkopt;
  THIntTensor *ipiv;
  THZTensor *work;
  THZTensor *ra__;

  int clonea;
  int destroy;

  if (a == NULL) /* possibly destroy the inputs  */
  {
    ra__ = THZTensor_(new)();
    clonea = THZTensor_(lapackClone)(ra__,ra_,0);
    destroy = 1;
  }
  else /*we want to definitely clone */
  {
    clonea = THZTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroy = 0;
  }

  THArgCheck(ra__->nDimension == 2, 2, "A should be 2 dimensional");
  m = ra__->size[0];
  n = ra__->size[1];
  THArgCheck(m == n, 2, "A should be square");
  lda = m;
  ipiv = THIntTensor_newWithSize1d((long)m);

  /* Run LU */
  THZLapack_(getrf)(n, n, THZTensor_(data)(ra__), lda, THIntTensor_data(ipiv), &info);
  if (info > 0)
  {
    THError("Lapack getrf : U(%d,%d) is 0, U is singular",info, info);
  }
  else if (info < 0)
  {
    THError("Lapack getrf : Argument %d : illegal value", -info);
  }

  /* Run inverse */
  THZLapack_(getri)(n, THZTensor_(data)(ra__), lda, THIntTensor_data(ipiv), &wkopt, -1, &info);
  lwork = (int)wkopt;
  work = THZTensor_(newWithSize1d)(lwork);
  THZLapack_(getri)(n, THZTensor_(data)(ra__), lda, THIntTensor_data(ipiv), THZTensor_(data)(work), lwork, &info);
  if (info > 0)
  {
    THError("Lapack getri : U(%d,%d) is 0, U is singular",info, info);
  }
  else if (info < 0)
  {
    THError("Lapack getri : Argument %d : illegal value", -info);
  }

  /* clean up */
  if (destroy)
  {
    if (clonea)
    {
      THZTensor_(copy)(ra_,ra__);
    }
    THZTensor_(free)(ra__);
  }
  THZTensor_(free)(work);
  THIntTensor_free(ipiv);
}

THZ_API void THZTensor_(potrf)(THZTensor *ra_, THZTensor *a)
{
  int n, lda, info;
  char uplo = 'U';
  THZTensor *ra__;

  int clonea;
  int destroy;

  if (a == NULL) /* possibly destroy the inputs  */
  {
    ra__ = THZTensor_(new)();
    clonea = THZTensor_(lapackClone)(ra__,ra_,0);
    destroy = 1;
  }
  else /*we want to definitely clone */
  {
    clonea = THZTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroy = 0;
  }

  THArgCheck(ra__->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(ra__->size[0] == ra__->size[1], 2, "A should be square");
  n = ra__->size[0];
  lda = n;

  /* Run Factorization */
  THZLapack_(potrf)(uplo, n, THZTensor_(data)(ra__), lda, &info);
  if (info > 0)
  {
    THError("Lapack potrf : A(%d,%d) is 0, A cannot be factorized", info, info);
  }
  else if (info < 0)
  {
    THError("Lapack potrf : Argument %d : illegal value", -info);
  }

  /* Build full upper-triangular matrix */
  {
    real *p = THZTensor_(data)(ra__);
    long i,j;
    for (i=0; i<n; i++) {
      for (j=i+1; j<n; j++) {
        p[i*n+j] = 0;
      }
    }
  }

  /* clean up */
  if (destroy)
  {
    if (clonea)
    {
      THZTensor_(copy)(ra_,ra__);
    }
    THZTensor_(free)(ra__);
  }
}

THZ_API void THZTensor_(potri)(THZTensor *ra_, THZTensor *a)
{
  int n, lda, info;
  char uplo = 'U';
  THZTensor *ra__;

  int clonea;
  int destroy;

  if (a == NULL) /* possibly destroy the inputs  */
  {
    ra__ = THZTensor_(new)();
    clonea = THZTensor_(lapackClone)(ra__,ra_,0);
    destroy = 1;
  }
  else /*we want to definitely clone */
  {
    clonea = THZTensor_(lapackClone)(ra_,a,1);
    ra__ = ra_;
    destroy = 0;
  }

  THArgCheck(ra__->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(ra__->size[0] == ra__->size[1], 2, "A should be square");
  n = ra__->size[0];
  lda = n;

  /* Run Factorization */
  THZLapack_(potrf)(uplo, n, THZTensor_(data)(ra__), lda, &info);
  if (info > 0)
  {
    THError("Lapack potrf : A(%d,%d) is 0, A cannot be factorized", info, info);
  }
  else if (info < 0)
  {
    THError("Lapack potrf : Argument %d : illegal value", -info);
  }

  /* Run inverse */
  THZLapack_(potri)(uplo, n, THZTensor_(data)(ra__), lda, &info);
  if (info > 0)
  {
    THError("Lapack potrf : A(%d,%d) is 0, A cannot be factorized", info, info);
  }
  else if (info < 0)
  {
    THError("Lapack potrf : Argument %d : illegal value", -info);
  }

  /* Build full matrix */
  {
    real *p = THZTensor_(data)(ra__);
    long i,j;
    for (i=0; i<n; i++) {
      for (j=i+1; j<n; j++) {
        p[i*n+j] = p[j*n+i];
      }
    }
  }

  /* clean up */
  if (destroy)
  {
    if (clonea)
    {
      THZTensor_(copy)(ra_,ra__);
    }
    THZTensor_(free)(ra__);
  }
}

#endif

/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorMath.c"
#else

#define THZ_OMP_OVERHEAD_THZRESHOLD 100000

void THZTensor_(fill)(THZTensor *r_, real value)
{
  TH_TENSOR_APPLY(real, r_,
                  THZVector_(fill)(r__data, value, r__size); break;);
}

void THZTensor_(zero)(THZTensor *r_)
{
  TH_TENSOR_APPLY(real, r_,
                  THZVector_(fill)(r__data, 0, r__size); break;);
}

void THZTensor_(maskedFill)(THZTensor *tensor, THByteTensor *mask, real value)
{
  TH_TENSOR_APPLY2(real, tensor, unsigned char, mask,
                   if (*mask_data > 1) THError("Mask tensor can take 0 and 1 values only");
                   else if (*mask_data == 1) *tensor_data = value;);
}

void THZTensor_(maskedCopy)(THZTensor *tensor, THByteTensor *mask, THZTensor* src )
{
  THZTensor *srct = THZTensor_(newContiguous)(src);
  real *src_data = THZTensor_(data)(srct);
  long cntr = 0;
  long nelem = THZTensor_(nElement)(srct);
  TH_TENSOR_APPLY2(real, tensor, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     *tensor_data = *src_data;
                     src_data++;
                     cntr++;
                     if (cntr > nelem)
                       THError("Number of elements of src != mask");
                   });
  if (cntr != nelem)
    THError("Number of elements of src != mask");
  THZTensor_(free)(srct);
}

void THZTensor_(maskedSelect)(THZTensor *tensor, THZTensor *src, THByteTensor *mask)
{
  long numel = THByteTensor_sumall(mask);
  real *tensor_data;

  THZTensor_(resize1d)(tensor,numel);
  tensor_data = THZTensor_(data)(tensor);
  TH_TENSOR_APPLY2(real, src, unsigned char, mask,
                   if (*mask_data > 1)
                   {
                     THError("Mask tensor can take 0 and 1 values only");
                   }
                   else if (*mask_data == 1)
                   {
                     *tensor_data = *src_data;
                     tensor_data++;
                   });
}

void THZTensor_(indexSelect)(THZTensor *tensor, THZTensor *src, int dim, THLongTensor *index)
{
  long i, numel;
  THLongStorage *newSize;
  THZTensor *tSlice, *sSlice;
  long *index_data;

  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(src->nDimension > 0,2,"Source tensor is empty");

  numel = THLongTensor_nElement(index);

  newSize = THLongStorage_newWithSize(src->nDimension);
  THLongStorage_rawCopy(newSize,src->size);
  newSize->data[dim] = numel;
  THZTensor_(resize)(tensor,newSize,NULL);
  THLongStorage_free(newSize);

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);
  for (i=0; i<numel; i++)
  {
    if (src->nDimension > 1)
    {
      tSlice = THZTensor_(new)();
      sSlice = THZTensor_(new)();
      THZTensor_(select)(tSlice, tensor, dim, i);
      THZTensor_(select)(sSlice, src, dim, index_data[i]-1);
      THZTensor_(copy)(tSlice, sSlice);
      THZTensor_(free)(tSlice);
      THZTensor_(free)(sSlice);
    }
    else
    {
      THZTensor_(set1d)(tensor,i,THZTensor_(get1d)(src,index_data[i]-1));
    }
  }
  THLongTensor_free(index);
}

void THZTensor_(indexCopy)(THZTensor *tensor, int dim, THLongTensor *index, THZTensor *src)
{
  long i, numel;
  THZTensor *tSlice, *sSlice;
  long *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension,4,"Indexing dim is out of bounds");
  THArgCheck(numel == src->size[dim],4,"Number of indices should be equal to source:size(dim)");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  for (i=0; i<numel; i++)
  {
    if (tensor->nDimension > 1 )
    {
      tSlice = THZTensor_(new)();
      sSlice = THZTensor_(new)();
      THZTensor_(select)(tSlice, tensor, dim, index_data[i]-1);
      THZTensor_(select)(sSlice, src, dim, i);
      THZTensor_(copy)(tSlice, sSlice);
      THZTensor_(free)(tSlice);
      THZTensor_(free)(sSlice);
    }
    else
    {
      THZTensor_(set1d)(tensor,index_data[i]-1,THZTensor_(get1d)(src,i));
    }
  }
  THLongTensor_free(index);
}

void THZTensor_(indexFill)(THZTensor *tensor, int dim, THLongTensor *index, real val)
{
  long i, numel;
  THZTensor *tSlice;
  long *index_data;

  numel = THLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < tensor->nDimension,4,"Indexing dim is out of bounds");

  index = THLongTensor_newContiguous(index);
  index_data = THLongTensor_data(index);

  for (i=0; i<numel; i++)
  {
    if (tensor->nDimension > 1 )
    {
      tSlice = THZTensor_(new)();
      THZTensor_(select)(tSlice, tensor,dim,index_data[i]-1);
      THZTensor_(fill)(tSlice, val);
      THZTensor_(free)(tSlice);
    }
    else
    {
      THZTensor_(set1d)(tensor,index_data[i]-1,val);
    }
  }
  THLongTensor_free(index);
}

accreal THZTensor_(dot)(THZTensor *tensor, THZTensor *src)
{
  accreal sum = 0;
  /* we use a trick here. careful with that. */
  TH_TENSOR_APPLY2(real, tensor, real, src,
                   long sz = (tensor_size-tensor_i < src_size-src_i ? tensor_size-tensor_i : src_size-src_i);
                   sum += THZBlas_(dot)(sz, src_data, src_stride, tensor_data, tensor_stride);
                   tensor_i += sz;
                   src_i += sz;
                   tensor_data += sz*tensor_stride;
                   src_data += sz*src_stride;
                   break;);
  return sum;
}

real THZTensor_(minall)(THZTensor *tensor)
{
  real theMin;
  THArgCheck(tensor->nDimension > 0, 1, "tensor must have one dimension");
  theMin = THZTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor, if(CABS(*tensor_data) < CABS(theMin)) theMin = *tensor_data;);
  return theMin;
}

real THZTensor_(maxall)(THZTensor *tensor)
{
  real theMax;
  THArgCheck(tensor->nDimension > 0, 1, "tensor must have one dimension");
  theMax = THZTensor_(data)(tensor)[0];
  TH_TENSOR_APPLY(real, tensor, if(CABS(*tensor_data) > CABS(theMax)) theMax = *tensor_data;);
  return theMax;
}

accreal THZTensor_(sumall)(THZTensor *tensor)
{
  accreal sum = 0;
  TH_TENSOR_APPLY(real, tensor, sum += *tensor_data;);
  return sum;
}

void THZTensor_(add)(THZTensor *r_, THZTensor *t, real value)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(t)) {
      real *tp = THZTensor_(data)(t);
      real *rp = THZTensor_(data)(r_);
      long sz = THZTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > THZ_OMP_OVERHEAD_THZRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = tp[i] + value;
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data + value;);
  }
}

void THZTensor_(mul)(THZTensor *r_, THZTensor *t, real value)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(t)) {
      real *tp = THZTensor_(data)(t);
      real *rp = THZTensor_(data)(r_);
      long sz = THZTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > THZ_OMP_OVERHEAD_THZRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = tp[i] * value;
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data * value;);
  }
}

void THZTensor_(div)(THZTensor *r_, THZTensor *t, real value)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(t)) {
      real *tp = THZTensor_(data)(t);
      real *rp = THZTensor_(data)(r_);
      long sz = THZTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > THZ_OMP_OVERHEAD_THZRESHOLD) private(i)
      for (i=0; i<sz; i++)
          rp[i] = tp[i] / value;
  } else {
      TH_TENSOR_APPLY2(real, r_, real, t, *r__data = *t_data / value;);
  }
}

void THZTensor_(cadd)(THZTensor *r_, THZTensor *t, real value, THZTensor *src)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(isContiguous)(src) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(src)) {
    if(r_ == t) {
      THZBlas_(axpy)(THZTensor_(nElement)(t), value, THZTensor_(data)(src), 1, THZTensor_(data)(r_), 1);
    } else {
      real *tp = THZTensor_(data)(t);
      real *sp = THZTensor_(data)(src);
      real *rp = THZTensor_(data)(r_);
      long sz = THZTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > THZ_OMP_OVERHEAD_THZRESHOLD) private(i)
      for (i=0; i< sz; i++)
          rp[i] = tp[i] + value * sp[i];
    }
  } else {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data + value * *src_data;);
  }
}

void THZTensor_(cmul)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(isContiguous)(src) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(src)) {
      real *tp = THZTensor_(data)(t);
      real *sp = THZTensor_(data)(src);
      real *rp = THZTensor_(data)(r_);
      long sz = THZTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > THZ_OMP_OVERHEAD_THZRESHOLD) private(i)
      for (i=0; i<sz; i++)
        rp[i] = tp[i] * sp[i];
  } else {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data * *src_data;);
  }
}

void THZTensor_(cdiv)(THZTensor *r_, THZTensor *t, THZTensor *src)
{
  THZTensor_(resizeAs)(r_, t);
  if (THZTensor_(isContiguous)(r_) && THZTensor_(isContiguous)(t) && THZTensor_(isContiguous)(src) && THZTensor_(nElement)(r_) == THZTensor_(nElement)(src)) {
      real *tp = THZTensor_(data)(t);
      real *sp = THZTensor_(data)(src);
      real *rp = THZTensor_(data)(r_);
      long sz = THZTensor_(nElement)(t);
      long i;
      #pragma omp parallel for if(sz > THZ_OMP_OVERHEAD_THZRESHOLD) private(i)
      for (i=0; i<sz; i++)
        rp[i] = tp[i] / sp[i];
  } else {
      TH_TENSOR_APPLY3(real, r_, real, t, real, src, *r__data = *t_data / *src_data;);
  }
}

void THZTensor_(addcmul)(THZTensor *r_, THZTensor *t, real value, THZTensor *src1, THZTensor *src2)
{
  if(r_ != t)
  {
    THZTensor_(resizeAs)(r_, t);
    THZTensor_(copy)(r_, t);
  }

  TH_TENSOR_APPLY3(real, r_, real, src1, real, src2, *r__data += value * *src1_data * *src2_data;);
}


void THZTensor_(addcdiv)(THZTensor *r_, THZTensor *t, real value, THZTensor *src1, THZTensor *src2)
{
  if(r_ != t)
  {
    THZTensor_(resizeAs)(r_, t);
    THZTensor_(copy)(r_, t);
  }

  TH_TENSOR_APPLY3(real, r_, real, src1, real, src2, *r__data += value * *src1_data / *src2_data;);
}

void THZTensor_(addmv)(THZTensor *r_, real beta, THZTensor *t, real alpha, THZTensor *mat, THZTensor *vec)
{
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");

  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if(t->nDimension != 1)
    THError("size mismatch");

  if(t->size[0] != mat->size[0])
    THError("size mismatch");

  if(r_ != t)
  {
    THZTensor_(resizeAs)(r_, t);
    THZTensor_(copy)(r_, t);
  }

  if(mat->stride[0] == 1)
  {
    THZBlas_(gemv)('n', mat->size[0], mat->size[1],
                  alpha, THZTensor_(data)(mat), mat->stride[1],
                  THZTensor_(data)(vec), vec->stride[0],
                  beta, THZTensor_(data)(r_), r_->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THZBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THZTensor_(data)(mat), mat->stride[0],
                  THZTensor_(data)(vec), vec->stride[0],
                  beta, THZTensor_(data)(r_), r_->stride[0]);
  }
  else
  {
    THZTensor *cmat = THZTensor_(newContiguous)(mat);

    THZBlas_(gemv)('t',  mat->size[1], mat->size[0],
                  alpha, THZTensor_(data)(cmat), cmat->stride[0],
                  THZTensor_(data)(vec), vec->stride[0],
                  beta, THZTensor_(data)(r_), r_->stride[0]);

    THZTensor_(free)(cmat);
  }
}

void THZTensor_(addmm)(THZTensor *r_, real beta, THZTensor *t, real alpha, THZTensor *m1, THZTensor *m2)
{
  char transpose_r, transpose_m1, transpose_m2;
  THZTensor *r__, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2) )
    THError("matrix and matrix expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) || (m1->size[1] != m2->size[0]) )
    THError("size mismatch");

  if(t != r_)
  {
    THZTensor_(resizeAs)(r_, t);
    THZTensor_(copy)(r_, t);
  }

/*  printf("%ldx%ld = %ldx%ld X %ldx%ld\n", r_->size[0], r_->size[1], m1->size[0], m1->size[1], m2->size[0], m2->size[1]); */

  /* r_ */
  if(r_->stride[0] == 1)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride[1] == 1)
  {
    THZTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    r__ = THZTensor_(newWithSize2d)(r_->size[1], r_->size[0]);
    THZTensor_(copy)(r__, r_);
    THZTensor_(transpose)(r__, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THZTensor_(newContiguous)(m1);
  }

  /* m2 */
  if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THZTensor_(newContiguous)(m2);
  }

  /* do the operation */
  THZBlas_(gemm)(transpose_m1,
                transpose_m2,
                r__->size[(transpose_r == 'n' ? 0 : 1)],
                r__->size[(transpose_r == 'n' ? 1 : 0)],
                m1_->size[(transpose_r == 'n' ? 1 : 0)],
                alpha,
                THZTensor_(data)(m1_),
                (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                THZTensor_(data)(m2_),
                (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                beta,
                THZTensor_(data)(r__),
                r__->stride[(transpose_r == 'n' ? 1 : 0)]);

  /* free intermediate variables */
  if(m1_ != m1)
    THZTensor_(free)(m1_);

  if(m2_ != m2)
    THZTensor_(free)(m2_);

  if(r__ != r_)
    THZTensor_(freeCopyTo)(r__, r_);
}

void THZTensor_(addr)(THZTensor *r_, real beta, THZTensor *t, real alpha, THZTensor *vec1, THZTensor *vec2)
{
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if(r_ != t)
  {
    THZTensor_(resizeAs)(r_, t);
    THZTensor_(copy)(r_, t);
  }

  if(beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  if(r_->stride[0] == 1)
  {
    THZBlas_(gerc)(vec1->size[0], vec2->size[0],
                  alpha, THZTensor_(data)(vec1), vec1->stride[0],
                  THZTensor_(data)(vec2), vec2->stride[0],
                  THZTensor_(data)(r_), r_->stride[1]);
  }
  else
  {
    THZTensor *cr = r_;
    if(r_->stride[1] != 1)
      cr = THZTensor_(newClone)(r_);

    THZTensor *cvec2 = THZTensor_(new)();
    THZTensor_(conj)(cvec2, vec2);

    THZBlas_(geru)(cvec2->size[0], vec1->size[0],
                   alpha, THZTensor_(data)(cvec2), cvec2->stride[0],
                   THZTensor_(data)(vec1), vec1->stride[0],
                   THZTensor_(data)(cr), cr->stride[0]);

    THZTensor_(free)(cvec2);

    if (cr != r_)
      THZTensor_(freeCopyTo)(cr, r_);
  }
}

void THZTensor_(addru)(THZTensor *r_, real beta, THZTensor *t, real alpha, THZTensor *vec1, THZTensor *vec2)
{
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if(r_ != t)
  {
    THZTensor_(resizeAs)(r_, t);
    THZTensor_(copy)(r_, t);
  }

  if(beta != 1)
    THZTensor_(mul)(r_, r_, beta);

  if(r_->stride[0] == 1)
  {
    THZBlas_(geru)(vec1->size[0], vec2->size[0],
                  alpha, THZTensor_(data)(vec1), vec1->stride[0],
                  THZTensor_(data)(vec2), vec2->stride[0],
                  THZTensor_(data)(r_), r_->stride[1]);
  }
  else if(r_->stride[0] == 1)
  {
    THZBlas_(geru)(vec2->size[0], vec1->size[0],
                   alpha, THZTensor_(data)(vec2), vec2->stride[0],
                   THZTensor_(data)(vec1), vec1->stride[0],
                   THZTensor_(data)(r_), r_->stride[0]);
  }
  else
  {
    THZTensor *cr = THZTensor_(newClone)(r_);

    THZBlas_(geru)(vec2->size[0], vec1->size[0],
                   alpha, THZTensor_(data)(vec2), vec2->stride[0],
                   THZTensor_(data)(vec1), vec1->stride[0],
                   THZTensor_(data)(cr), cr->stride[0]);

    THZTensor_(freeCopyTo)(cr, r_);
  }
}

long THZTensor_(numel)(THZTensor *t)
{
  return THZTensor_(nElement)(t);
}

void THZTensor_(max)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension)
{
  THLongStorage *dim;
  long i;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "dimension out of range");

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY3(real, t, real, values_, long, indices_, dimension,
                       long theIndex = 0;
                       real theMax = t_data[0];
                       for(i = 1; i < t_size; i++)
                       {
                         if(CABS(t_data[i*t_stride]) > CABS(theMax))
                         {
                           theIndex = i;
                           theMax = t_data[i*t_stride];
                         }
                       }
                       *indices__data = theIndex;
                       *values__data = theMax;);

}

void THZTensor_(min)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension)
{
  THLongStorage *dim;
  long i;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "dimension out of range");

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(values_, dim, NULL);
  THLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY3(real, t, real, values_, long, indices_, dimension,
                       long theIndex = 0;
                       real theMin = t_data[0];
                       for(i = 1; i < t_size; i++)
                       {
                         if(CABS(t_data[i*t_stride]) < CABS(theMin))
                         {
                           theIndex = i;
                           theMin = t_data[i*t_stride];
                         }
                       }
                       *indices__data = theIndex;
                       *values__data = theMin;);

}


void THZTensor_(sum)(THZTensor *r_, THZTensor *t, int dimension)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "dimension out of range");

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal sum = 0;
                       long i;
                       for(i = 0; i < t_size; i++)
                         sum += t_data[i*t_stride];
                       *r__data = (real)sum;);
}

void THZTensor_(prod)(THZTensor *r_, THZTensor *t, int dimension)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "dimension out of range");

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal prod = 1;
                       long i;
                       for(i = 0; i < t_size; i++)
                         prod *= t_data[i*t_stride];
                       *r__data = (real)prod;);

}

void THZTensor_(cumsum)(THZTensor *r_, THZTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "dimension out of range");

  THZTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal cumsum = 0;
                       long i;
                       for(i = 0; i < t_size; i++)
                       {
                         cumsum += t_data[i*t_stride];
                         r__data[i*r__stride] = (real)cumsum;
                       });
}

void THZTensor_(cumprod)(THZTensor *r_, THZTensor *t, int dimension)
{
  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "dimension out of range");

  THZTensor_(resizeAs)(r_, t);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal cumprod = 1;
                       long i;
                       for(i = 0; i < t_size; i++)
                       {
                         cumprod *= t_data[i*t_stride];
                         r__data[i*r__stride] = (real)cumprod;
                       });
}


accreal THZTensor_(trace)(THZTensor *t)
{
  real *t_data = THZTensor_(data)(t);
  accreal sum = 0;
  long i = 0;
  long t_stride_0, t_stride_1, t_diag_size;

  THArgCheck(THZTensor_(nDimension)(t) == 2, 1, "not a matrix");

  t_stride_0 = THZTensor_(stride)(t, 0);
  t_stride_1 = THZTensor_(stride)(t, 1);
  t_diag_size = THMin(THZTensor_(size)(t, 0), THZTensor_(size)(t, 1));
  while(i < t_diag_size)
  {
    sum += t_data[i*(t_stride_0+t_stride_1)];
    i++;
  }

  return sum;
}

void THZTensor_(cross)(THZTensor *r_, THZTensor *a, THZTensor *b, int dimension)
{
  int i;

  if(THZTensor_(nDimension)(a) != THZTensor_(nDimension)(b))
    THError("inconsitent tensor sizes");

  for(i = 0; i < THZTensor_(nDimension)(a); i++)
  {
    if(THZTensor_(size)(a, i) != THZTensor_(size)(b, i))
      THError("inconsistent tensor sizes");
  }

  if(dimension < 0)
  {
    for(i = 0; i < THZTensor_(nDimension)(a); i++)
    {
      if(THZTensor_(size)(a, i) == 3)
      {
        dimension = i;
        break;
      }
    }
    if(dimension < 0)
      THError("no dimension of size 3");
  }

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(a), 3, "dimension out of range");
  THArgCheck(THZTensor_(size)(a, dimension) == 3, 3, "dimension size is not 3");

  THZTensor_(resizeAs)(r_, a);

  TH_TENSOR_DIM_APPLY3(real, a, real, b, real, r_, dimension,
                       r__data[0*r__stride] = a_data[1*a_stride]*b_data[2*b_stride] - a_data[2*a_stride]*b_data[1*b_stride];
                       r__data[1*r__stride] = a_data[2*a_stride]*b_data[0*b_stride] - a_data[0*a_stride]*b_data[2*b_stride];
                       r__data[2*r__stride] = a_data[0*a_stride]*b_data[1*b_stride] - a_data[1*a_stride]*b_data[0*b_stride];);
}

void THZTensor_(zeros)(THZTensor *r_, THLongStorage *size)
{
  THZTensor_(resize)(r_, size, NULL);
  THZTensor_(zero)(r_);
}

void THZTensor_(ones)(THZTensor *r_, THLongStorage *size)
{
  THZTensor_(resize)(r_, size, NULL);
  THZTensor_(fill)(r_, 1);
}

void THZTensor_(diag)(THZTensor *r_, THZTensor *t, int k)
{
  THArgCheck(THZTensor_(nDimension)(t) == 1 || THZTensor_(nDimension)(t) == 2, 1, "matrix or a vector expected");

  if(THZTensor_(nDimension)(t) == 1)
  {
    real *t_data = THZTensor_(data)(t);
    long t_stride_0 = THZTensor_(stride)(t, 0);
    long t_size = THZTensor_(size)(t, 0);
    long sz = t_size + (k >= 0 ? k : -k);
    real *r__data;
    long r__stride_0;
    long r__stride_1;
    long i;

    THZTensor_(resize2d)(r_, sz, sz);
    THZTensor_(zero)(r_);
    r__data = THZTensor_(data)(r_);
    r__stride_0 = THZTensor_(stride)(r_, 0);
    r__stride_1 = THZTensor_(stride)(r_, 1);
    r__data += (k >= 0 ? k*r__stride_1 : -k*r__stride_0);

    for(i = 0; i < t_size; i++)
      r__data[i*(r__stride_0+r__stride_1)] = t_data[i*t_stride_0];
  }
  else
  {
    real *t_data = THZTensor_(data)(t);
    long t_stride_0 = THZTensor_(stride)(t, 0);
    long t_stride_1 = THZTensor_(stride)(t, 1);
    long sz;
    real *r__data;
    long r__stride_0;
    long i;

    if(k >= 0)
      sz = THMin(THZTensor_(size)(t, 0), THZTensor_(size)(t, 1)-k);
    else
      sz = THMin(THZTensor_(size)(t, 0)+k, THZTensor_(size)(t, 1));
    THZTensor_(resize1d)(r_, sz);
    r__data = THZTensor_(data)(r_);
    r__stride_0 = THZTensor_(stride)(r_, 0);

    t_data += (k >= 0 ? k*t_stride_1 : -k*t_stride_0);
    for(i = 0; i < sz; i++)
      r__data[i*r__stride_0] = t_data[i*(t_stride_0+t_stride_1)];
  }
}

void THZTensor_(eye)(THZTensor *r_, long n, long m)
{
  real *r__data;
  long i, sz;

  THArgCheck(n > 0, 1, "invalid argument");

  if(m <= 0)
    m = n;

  THZTensor_(resize2d)(r_, n, m);
  THZTensor_(zero)(r_);

  i = 0;
  r__data = THZTensor_(data)(r_);
  sz = THMin(THZTensor_(size)(r_, 0), THZTensor_(size)(r_, 1));
  for(i = 0; i < sz; i++)
    r__data[i*(r_->stride[0]+r_->stride[1])] = 1;
}


void THZTensor_(reshape)(THZTensor *r_, THZTensor *t, THLongStorage *size)
{
  THZTensor_(resize)(r_, size, NULL);
  THZTensor_(copy)(r_, t);
}

/* I cut and pasted (slightly adapted) the quicksort code from
   http://www.alienryderflex.com/quicksort/
   This public-domain C implementation by Darel Rex Finley.
   Thanks man :)

    Updated Oct 16 2013: change choice of pivot to avoid worst-case being a pre-sorted input - Daniel and Julien
    Updated Oct 24 2013: change pivot comparison to strict inequality to avoid worst-case on constant input, see Sedgewick, Algorithms in C, Addison Wesley, 1990, p. 120 - Julien
*/
#define  MAX_LEVELS  300
static void THZTensor_(quicksortascend)(real *arr, long *idx, long elements, long stride)
{
  long beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R, P, swap, pid;
  real rswap, piv;

  beg[0]=0; end[0]=elements;
  while (i>=0) {
    L=beg[i]; R=end[i]-1;
    if (L<R) {
      P=(L+R)>>1; /* Choose pivot as middle element of the current block */
      piv=arr[P*stride];
      pid=idx[P*stride];
      rswap=arr[L*stride];
      swap=idx[L*stride];
      arr[L*stride]=piv;
      idx[L*stride]=pid;
      arr[P*stride]=rswap;
      idx[P*stride]=swap;
      while (L<R) {
        while (CABS(arr[R*stride])>CABS(piv) && L<R)
            R--;
        if (L<R) {
            idx[L*stride]=idx[R*stride];
            arr[L*stride]=arr[R*stride];
            L++;
        }
        while (CABS(arr[L*stride])<CABS(piv) && L<R)
            L++;
        if (L<R) {
            idx[R*stride]=idx[L*stride];
            arr[R*stride]=arr[L*stride];
            R--;
        }
      }
      idx[L*stride]=pid;
      arr[L*stride]=piv;
      beg[i+1]=L+1;
      end[i+1]=end[i];
      end[i++]=L;
      if (end[i]-beg[i]>end[i-1]-beg[i-1]) {
        swap=beg[i]; beg[i]=beg[i-1]; beg[i-1]=swap;
        swap=end[i]; end[i]=end[i-1]; end[i-1]=swap;
      }
    }
    else {
      i--;
    }
  }
}

static void THZTensor_(quicksortdescend)(real *arr, long *idx, long elements, long stride)
{
  long beg[MAX_LEVELS], end[MAX_LEVELS], i=0, L, R, P, swap, pid;
  real rswap, piv;

  beg[0]=0; end[0]=elements;
  while (i>=0) {
    L=beg[i]; R=end[i]-1;
    if (L<R) {
      P=(L+R)>>1; /* Choose pivot as middle element of the current block */
      piv=arr[P*stride];
      pid=idx[P*stride];
      rswap=arr[L*stride];
      swap=idx[L*stride];
      arr[L*stride]=piv;
      idx[L*stride]=pid;
      arr[P*stride]=rswap;
      idx[P*stride]=swap;
      while (L<R) {
        while (CABS(arr[R*stride])<CABS(piv) && L<R)
            R--;
        if (L<R) {
            idx[L*stride]=idx[R*stride];
            arr[L*stride]=arr[R*stride];
            L++;
        }
        while (CABS(arr[L*stride])>CABS(piv) && L<R)
            L++;
        if (L<R) {
            idx[R*stride]=idx[L*stride];
            arr[R*stride]=arr[L*stride];
            R--;
        }
      }
      idx[L*stride]=pid;
      arr[L*stride]=piv;
      beg[i+1]=L+1;
      end[i+1]=end[i];
      end[i++]=L;
      if (end[i]-beg[i]>end[i-1]-beg[i-1]) {
        swap=beg[i]; beg[i]=beg[i-1]; beg[i-1]=swap;
        swap=end[i]; end[i]=end[i-1]; end[i-1]=swap;
      }
    }
    else {
      i--;
    }
  }
}

void THZTensor_(sort)(THZTensor *rt_, THLongTensor *ri_, THZTensor *t, int dimension, int descendingOrder)
{
  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "invalid dimension");

  THZTensor_(resizeAs)(rt_, t);
  THZTensor_(copy)(rt_, t);

  {
    THLongStorage *size = THZTensor_(newSizeOf)(t);
    THLongTensor_resize(ri_, size, NULL);
    THLongStorage_free(size);
  }

  if(descendingOrder)
  {
    TH_TENSOR_DIM_APPLY2(real, rt_, long, ri_, dimension,
                         long i;
                         for(i = 0; i < ri__size; i++)
                           ri__data[i*ri__stride] = i;
                         THZTensor_(quicksortdescend)(rt__data, ri__data, rt__size, rt__stride);)
      }
  else
  {
    TH_TENSOR_DIM_APPLY2(real, rt_, long, ri_, dimension,
                         long i;
                         for(i = 0; i < ri__size; i++)
                           ri__data[i*ri__stride] = i;
                         THZTensor_(quicksortascend)(rt__data, ri__data, rt__size, rt__stride);)
      }
}

void THZTensor_(tril)(THZTensor *r_, THZTensor *t, long k)
{
  long t_size_0, t_size_1;
  long t_stride_0, t_stride_1;
  long r__stride_0, r__stride_1;
  real *t_data, *r__data;
  long r, c;

  THArgCheck(THZTensor_(nDimension)(t) == 2, 1, "not a matrix");

  THZTensor_(resizeAs)(r_, t);

  t_size_0 = THZTensor_(size)(t, 0);
  t_size_1 = THZTensor_(size)(t, 1);
  t_stride_0 = THZTensor_(stride)(t, 0);
  t_stride_1 = THZTensor_(stride)(t, 1);
  r__stride_0 = THZTensor_(stride)(r_, 0);
  r__stride_1 = THZTensor_(stride)(r_, 1);
  r__data = THZTensor_(data)(r_);
  t_data = THZTensor_(data)(t);

  for(r = 0; r < t_size_0; r++)
  {
    long sz = THMin(r+k+1, t_size_1);
    for(c = THMax(0, r+k); c < t_size_1; c++)
      r__data[r*r__stride_0+c*r__stride_1] = 0;
    for(c = 0; c < sz; c++)
      r__data[r*r__stride_0+c*r__stride_1] = t_data[r*t_stride_0+c*t_stride_1];
  }
}

void THZTensor_(triu)(THZTensor *r_, THZTensor *t, long k)
{
  long t_size_0, t_size_1;
  long t_stride_0, t_stride_1;
  long r__stride_0, r__stride_1;
  real *t_data, *r__data;
  long r, c;

  THArgCheck(THZTensor_(nDimension)(t) == 2, 1, "not a matrix");

  THZTensor_(resizeAs)(r_, t);

  t_size_0 = THZTensor_(size)(t, 0);
  t_size_1 = THZTensor_(size)(t, 1);
  t_stride_0 = THZTensor_(stride)(t, 0);
  t_stride_1 = THZTensor_(stride)(t, 1);
  r__stride_0 = THZTensor_(stride)(r_, 0);
  r__stride_1 = THZTensor_(stride)(r_, 1);
  r__data = THZTensor_(data)(r_);
  t_data = THZTensor_(data)(t);

  for(r = 0; r < t_size_0; r++)
  {
    long sz = THMin(r+k, t_size_1);
    for(c = THMax(0, r+k); c < t_size_1; c++)
      r__data[r*r__stride_0+c*r__stride_1] = t_data[r*t_stride_0+c*t_stride_1];
    for(c = 0; c < sz; c++)
      r__data[r*r__stride_0+c*r__stride_1] = 0;
  }
}

void THZTensor_(cat)(THZTensor *r_, THZTensor *ta, THZTensor *tb, int dimension)
{
  THLongStorage *size;
  int i;
  int ndim = THMax(ta->nDimension, tb->nDimension);
  ndim = THMax(ndim, dimension+1);

  THArgCheck(dimension >= 0, 4, "invalid dimension");

  size = THLongStorage_newWithSize(ndim);
  for(i = 0; i < ndim; i++)
  {
    int tadi = (i < ta->nDimension ? ta->size[i] : 1);
    int tbdi = (i < tb->nDimension ? tb->size[i] : 1);

    if(i == dimension)
      size->data[i] = tadi+tbdi;
    else
    {
      if(tadi != tbdi)
      {
        THLongStorage_free(size);
        THError("inconsistent tensor sizes");
      }
      size->data[i] = tadi;
    }
  }

  THZTensor_(resize)(r_, size, NULL);
  THLongStorage_free(size);

  {
    THZTensor *nta = THZTensor_(newWithTensor)(r_);
    THZTensor_(narrow)(nta, NULL, dimension, 0, (dimension < ta->nDimension ? ta->size[dimension] : 1));
    THZTensor_(copy)(nta, ta);
    THZTensor_(free)(nta);
  }

  {
    THZTensor *ntb = THZTensor_(newWithTensor)(r_);
    THZTensor_(narrow)(ntb, NULL, dimension, (dimension < ta->nDimension ? ta->size[dimension] : 1), (dimension < tb->nDimension ? tb->size[dimension] : 1));
    THZTensor_(copy)(ntb, tb);
    THZTensor_(free)(ntb);
  }
}



#define TENSOR_IMPLEMENT_LOGICAL(NAME,OP)                               \
  void THZTensor_(NAME##Value)(THByteTensor *r_, THZTensor* t, real value) \
  {                                                                     \
    THLongStorage *tsz = THZTensor_(newSizeOf)(t);                      \
    THByteTensor_resize(r_, tsz, NULL);                                 \
    THLongStorage_free(tsz);                                            \
    THByteTensor_zero(r_);                                              \
    TH_TENSOR_APPLY2(unsigned char, r_, real, t,                        \
                     if (CABS(*t_data) OP CABS(value)) *r__data = 1;); \
  }                                                                     \
  void THZTensor_(NAME##Tensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb) \
  {                                                                     \
    THLongStorage *tsz = THZTensor_(newSizeOf)(ta);                     \
    THByteTensor_resize(r_, tsz, NULL);                                 \
    THLongStorage_free(tsz);                                            \
    THByteTensor_zero(r_);                                              \
    TH_TENSOR_APPLY3(unsigned char, r_, real, ta, real, tb,             \
                     if(CABS(*ta_data) OP CABS(*tb_data)) *r__data = 1;); \
  }                                                                     \


TENSOR_IMPLEMENT_LOGICAL(lt,<)
TENSOR_IMPLEMENT_LOGICAL(gt,>)
TENSOR_IMPLEMENT_LOGICAL(le,<=)
TENSOR_IMPLEMENT_LOGICAL(ge,>=)
TENSOR_IMPLEMENT_LOGICAL(eq,==)
TENSOR_IMPLEMENT_LOGICAL(ne,!=)

#define LAB_IMPLEMENT_BASIC_FUNCTION(NAME, CFUNC)                       \
  void THZTensor_(NAME)(THZTensor *r_, THZTensor *t)                    \
  {                                                                     \
    THZTensor_(resizeAs)(r_, t);                                        \
    TH_TENSOR_APPLY2(real, t, real, r_, *r__data = CFUNC(*t_data););    \
  }                                                                     \

#define LAB_IMPLEMENT_BASIC_FUNCTION_VALUE(NAME, CFUNC)                 \
  void THZTensor_(NAME)(THZTensor *r_, THZTensor *t, real value)        \
  {                                                                     \
    THZTensor_(resizeAs)(r_, t);                                        \
    TH_TENSOR_APPLY2(real, t, real, r_, *r__data = CFUNC(*t_data, value);); \
  }                                                                     \

LAB_IMPLEMENT_BASIC_FUNCTION(log,CLOG)
LAB_IMPLEMENT_BASIC_FUNCTION(exp,CEXP)
LAB_IMPLEMENT_BASIC_FUNCTION(cos,CCOS)
LAB_IMPLEMENT_BASIC_FUNCTION(acos,CACOS)
LAB_IMPLEMENT_BASIC_FUNCTION(cosh,CACOSH)
LAB_IMPLEMENT_BASIC_FUNCTION(sin,CSIN)
LAB_IMPLEMENT_BASIC_FUNCTION(asin,CASIN)
LAB_IMPLEMENT_BASIC_FUNCTION(sinh,CSINH)
LAB_IMPLEMENT_BASIC_FUNCTION(tan,CTAN)
LAB_IMPLEMENT_BASIC_FUNCTION(atan,CATAN)
LAB_IMPLEMENT_BASIC_FUNCTION(tanh,CTANH)
LAB_IMPLEMENT_BASIC_FUNCTION_VALUE(pow,CPOW)
LAB_IMPLEMENT_BASIC_FUNCTION(sqrt,CSQRT)
LAB_IMPLEMENT_BASIC_FUNCTION(conj,CONJ)
LAB_IMPLEMENT_BASIC_FUNCTION(proj,CPROJ)
LAB_IMPLEMENT_BASIC_FUNCTION(zabs,CABS)
LAB_IMPLEMENT_BASIC_FUNCTION(zarg,CARG)
LAB_IMPLEMENT_BASIC_FUNCTION(zre,CREAL)
LAB_IMPLEMENT_BASIC_FUNCTION(zim,CIMAG)

#define LAB_IMPLEMENT_BASIC_FUNCTION_RETURN_FLOAT(NAME, CFUNC)          \
  void THZTensor_(NAME)(THFloatTensor *r, THZTensor *t)                 \
  {                                                                     \
    THLongStorage *tsz = THZTensor_(newSizeOf)(t);                      \
    THFloatTensor_resize(r, tsz, NULL);                                 \
    THLongStorage_free(tsz);                                            \
    TH_TENSOR_APPLY2(real, t, float, r, *r_data = CFUNC(*t_data););     \
  }
#define LAB_IMPLEMENT_BASIC_FUNCTION_RETURN_DOUBLE(NAME, CFUNC)         \
  void THZTensor_(NAME)(THDoubleTensor *r, THZTensor *t)                \
  {                                                                     \
    THLongStorage *tsz = THZTensor_(newSizeOf)(t);                      \
    THDoubleTensor_resize(r, tsz, NULL);                                \
    THLongStorage_free(tsz);                                            \
    TH_TENSOR_APPLY2(real, t, double, r, *r_data = CFUNC(*t_data););    \
  }

LAB_IMPLEMENT_BASIC_FUNCTION_RETURN_FLOAT(Float_abs,CABS)
LAB_IMPLEMENT_BASIC_FUNCTION_RETURN_FLOAT(Float_arg,CARG)
LAB_IMPLEMENT_BASIC_FUNCTION_RETURN_FLOAT(Float_re,CREAL)
LAB_IMPLEMENT_BASIC_FUNCTION_RETURN_FLOAT(Float_im,CIMAG)
LAB_IMPLEMENT_BASIC_FUNCTION_RETURN_DOUBLE(Double_abs,CABS)
LAB_IMPLEMENT_BASIC_FUNCTION_RETURN_DOUBLE(Double_arg,CARG)
LAB_IMPLEMENT_BASIC_FUNCTION_RETURN_DOUBLE(Double_re,CREAL)
LAB_IMPLEMENT_BASIC_FUNCTION_RETURN_DOUBLE(Double_im,CIMAG)


void THZTensor_(mean)(THZTensor *r_, THZTensor *t, int dimension)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 2, "invalid dimension");

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal sum = 0;
                       long i;
                       for(i = 0; i < t_size; i++)
                         sum += t_data[i*t_stride];
                       *r__data = (real)sum/t_size;);
}

void THZTensor_(std)(THZTensor *r_, THZTensor *t, int dimension, int flag)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 3, "invalid dimension");

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal sum = 0;
                       accreal sum2 = 0;
                       long i;
                       for(i = 0; i < t_size; i++)
                       {
                         real z = t_data[i*t_stride];
                         sum += z;
                         sum2 += z*z;
                       }

                       if(flag)
                       {
                         sum /= t_size;
                         sum2 /= t_size;
                         sum2 -= sum*sum;
                         sum2 = (cabs(sum2) < 0 ? 0 : sum2);
                         *r__data = (real)csqrt(sum2);
                       }
                       else
                       {
                         sum /= t_size;
                         sum2 /= t_size-1;
                         sum2 -= ((real)t_size)/((real)(t_size-1))*sum*sum;
                         sum2 = (cabs(sum2) < 0 ? 0 : sum2);
                         *r__data = (real)csqrt(sum2);
                       });
}

void THZTensor_(var)(THZTensor *r_, THZTensor *t, int dimension, int flag)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 3, "invalid dimension");

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                       accreal sum = 0;
                       accreal sum2 = 0;
                       long i;
                       for(i = 0; i < t_size; i++)
                       {
                         real z = t_data[i*t_stride];
                         sum += z;
                         sum2 += z*z;
                       }

                       if(flag)
                       {
                         sum /= t_size;
                         sum2 /= t_size;
                         sum2 -= sum*sum;
                         sum2 = (cabs(sum2) < 0 ? 0 : sum2);
                         *r__data = sum2;
                       }
                       else
                       {
                         sum /= t_size;
                         sum2 /= t_size-1;
                         sum2 -= ((real)t_size)/((real)(t_size-1))*sum*sum;
                         sum2 = (cabs(sum2) < 0 ? 0 : sum2);
                         *r__data = (real)sum2;
                       });
}

void THZTensor_(norm)(THZTensor *r_, THZTensor *t, real value, int dimension)
{
  THLongStorage *dim;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(t), 3, "invalid dimension");

  dim = THZTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THZTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  if(value == 0) {
    TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                         accreal sum = 0;
                         long i;
                         for(i = 0; i < t_size; i++)
                           sum += t_data[i*t_stride] != 0.0;
                         *r__data = sum;)
  } else {
    TH_TENSOR_DIM_APPLY2(real, t, real, r_, dimension,
                         accreal sum = 0;
                         long i;
                         for(i = 0; i < t_size; i++)
                           sum += pow(CABS(t_data[i*t_stride]), value);
                         *r__data = (real)cpow(sum, 1.0/value);)
  }
}

accreal THZTensor_(normall)(THZTensor *tensor, real value)
{
  accreal sum = 0;
  if(value == 0) {
    TH_TENSOR_APPLY(real, tensor, sum += *tensor_data != 0.0;);
    return sum;
  } else if(value == 1) {
    TH_TENSOR_APPLY(real, tensor, sum += CABS(*tensor_data););
    return sum;
  } else if(value == 2) {
    TH_TENSOR_APPLY(real, tensor, accreal z = *tensor_data; sum += z*z;);
    return csqrt(sum);
  } else {
    TH_TENSOR_APPLY(real, tensor, sum += pow(CABS(*tensor_data), value););
    return cpow(sum, 1.0/value);
  }
}

void THZTensor_(renorm)(THZTensor *res, THZTensor *src, real value, int dimension, real maxnorm)
{
  int i;
  THZTensor *rowR, *rowS;

  THArgCheck(dimension >= 0 && dimension < THZTensor_(nDimension)(src), 3, "invalid dimension");
  THArgCheck(CABS(value) > 0, 2, "non-positive-norm not supported");
  THArgCheck(THZTensor_(nDimension)(src) > 1, 1, "need at least 2 dimensions");

  rowR = THZTensor_(new)();
  rowS = THZTensor_(new)();

  THZTensor_(resizeAs)(res, src);

  for (i=0; i<src->size[dimension]; i++)
  {
    real norm = 0;
    real new_norm;

    THZTensor_(select)(rowS, src, dimension, i);
    THZTensor_(select)(rowR, res, dimension, i);
    if (value == 1) {
      TH_TENSOR_APPLY(real, rowS, norm += CABS(*rowS_data););
    } else if (value == 2) {
      TH_TENSOR_APPLY(real, rowS, accreal z = *rowS_data; norm += z*z;);
    } else {
      TH_TENSOR_APPLY(real, rowS, norm += pow(CABS(*rowS_data), value););
    }

    norm = CPOW(norm, 1/value);

    if (CABS(norm) > CABS(maxnorm))
    {
      new_norm = maxnorm / (norm + 1e-7);

      TH_TENSOR_APPLY2(
        real, rowR, real, rowS,
        *rowR_data = (*rowS_data) * new_norm;
      )
    }
    else
      THZTensor_(copy)(rowR, rowS);
  }

  THZTensor_(free)(rowR);
  THZTensor_(free)(rowS);
}

accreal THZTensor_(dist)(THZTensor *tensor, THZTensor *src, real value)
{
  real sum = 0;
  TH_TENSOR_APPLY2(real, tensor, real, src,
        sum += pow(CABS(*tensor_data - *src_data), value);)
  return CPOW(sum, 1.0/value);
}

accreal THZTensor_(meanall)(THZTensor *tensor)
{
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");
  return THZTensor_(sumall)(tensor)/THZTensor_(nElement)(tensor);
}

accreal THZTensor_(varall)(THZTensor *tensor)
{
  accreal mean = THZTensor_(meanall)(tensor);
  accreal sum = 0;
  TH_TENSOR_APPLY(real, tensor, sum += (*tensor_data - mean)*(*tensor_data - mean););
  sum /= (THZTensor_(nElement)(tensor)-1);
  return sum;
}

accreal THZTensor_(stdall)(THZTensor *tensor)
{
  return csqrt(THZTensor_(varall)(tensor));
}

#endif

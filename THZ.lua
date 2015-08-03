--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.

local ffi = require 'ffi'

local function cdef(template)
  local floatdefs = template:gsub('accreal', 'double _Complex'):gsub('Real', 'Float'):gsub('real', 'float _Complex')
  ffi.cdef(floatdefs)

  local doubledefs = template:gsub('accreal', 'double _Complex'):gsub('Real', 'Double'):gsub('real', 'double _Complex')
  ffi.cdef(doubledefs)
end

cdef([[
typedef struct THZRealStorage
{
    real *__data;
    long __size;
    int __refcount;
    char __flag;
    THAllocator* __allocator;
    void* __allocatorContext;
} THZRealStorage;

typedef struct THZRealTensor
{
    long *__size;
    long *__stride;
    int __nDimension;

    THZRealStorage *__storage;
    long __storageOffset;
    int __refcount;

    char __flag;

} THZRealTensor;
]])

cdef([[
real* THZRealStorage_data(const THZRealStorage*);
long THZRealStorage_size(const THZRealStorage*);


void THZRealStorage_set(THZRealStorage*, long, real);
real THZRealStorage_get(const THZRealStorage*, long);

THZRealStorage& THZRealStorage_new(void);
THZRealStorage& THZRealStorage_newWithSize(long size);
THZRealStorage& THZRealStorage_newWithSize1(real);
THZRealStorage& THZRealStorage_newWithSize2(real, real);
THZRealStorage& THZRealStorage_newWithSize3(real, real, real);
THZRealStorage& THZRealStorage_newWithSize4(real, real, real, real);
THZRealStorage& THZRealStorage_newWithMapping(const char *filename, long size, int shared);
THZRealStorage& THZRealStorage_newWithData(real *data, long size);
THZRealStorage& THZRealStorage_newWithAllocator(long size, THAllocator* allocator, void *allocatorContext);
THZRealStorage& THZRealStorage_newWithDataAndAllocator(real* data, long size, THAllocator* allocator, void *allocatorContext);


void THZRealStorage_setFlag(THZRealStorage *storage, const char flag);
void THZRealStorage_clearFlag(THZRealStorage *storage, const char flag);
void THZRealStorage_retain(THZRealStorage *storage);


void THZRealStorage_free(THZRealStorage *storage);
void THZRealStorage_resize(THZRealStorage *storage, long size);
void THZRealStorage_fill(THZRealStorage *storage, real value);

void THZRealStorage_rawCopy(THZRealStorage *storage, real *src);
void THZRealStorage_copyZFloat(THZRealStorage *storage, THZFloatStorage *src);
void THZRealStorage_copyZDouble(THZRealStorage *storage, THZDoubleStorage *src);
void THZRealStorage_copyByte(THZRealStorage *storage, struct THByteStorage *src);
void THZRealStorage_copyChar(THZRealStorage *storage, struct THCharStorage *src);
void THZRealStorage_copyShort(THZRealStorage *storage, struct THShortStorage *src);
void THZRealStorage_copyInt(THZRealStorage *storage, struct THIntStorage *src);
void THZRealStorage_copyLong(THZRealStorage *storage, struct THLongStorage *src);
void THZRealStorage_copyFloat(THZRealStorage *storage, struct THFloatStorage *src);
void THZRealStorage_copyDouble(THZRealStorage *storage, struct THDoubleStorage *src);


THZRealStorage& THZRealTensor_storage(const THZRealTensor *self);
long THZRealTensor_storageOffset(const THZRealTensor *self);
int THZRealTensor_nDimension(const THZRealTensor *self);
long THZRealTensor_size(const THZRealTensor *self, int dim);
long THZRealTensor_stride(const THZRealTensor *self, int dim);
THLongStorage *THZRealTensor_newSizeOf(THZRealTensor *self);
THLongStorage *THZRealTensor_newStrideOf(THZRealTensor *self);
real *THZRealTensor_data(const THZRealTensor *self);

void THZRealTensor_setFlag(THZRealTensor *self, const char flag);
void THZRealTensor_clearFlag(THZRealTensor *self, const char flag);



THZRealTensor& THZRealTensor_new(void);
THZRealTensor& THZRealTensor_newWithTensor(THZRealTensor *tensor);

THZRealTensor& THZRealTensor_newWithStorage(THZRealStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THZRealTensor& THZRealTensor_newWithStorage1d(THZRealStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
THZRealTensor& THZRealTensor_newWithStorage2d(THZRealStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
THZRealTensor& THZRealTensor_newWithStorage3d(THZRealStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
THZRealTensor& THZRealTensor_newWithStorage4d(THZRealStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);


THZRealTensor &THZRealTensor_newWithSize(THLongStorage *size_, THLongStorage *stride_);
THZRealTensor &THZRealTensor_newWithSize1d(long size0_);
THZRealTensor &THZRealTensor_newWithSize2d(long size0_, long size1_);
THZRealTensor &THZRealTensor_newWithSize3d(long size0_, long size1_, long size2_);
THZRealTensor &THZRealTensor_newWithSize4d(long size0_, long size1_, long size2_, long size3_);

THZRealTensor &THZRealTensor_newClone(THZRealTensor *self);
THZRealTensor &THZRealTensor_newContiguous(THZRealTensor *tensor);
THZRealTensor &THZRealTensor_newSelect(THZRealTensor *tensor, int dimension_, long sliceIndex_);
THZRealTensor &THZRealTensor_newNarrow(THZRealTensor *tensor, int dimension_, long firstIndex_, long size_);
THZRealTensor &THZRealTensor_newTranspose(THZRealTensor *tensor, int dimension1_, int dimension2_);
THZRealTensor &THZRealTensor_newUnfold(THZRealTensor *tensor, int dimension_, long size_, long step_);

void THZRealTensor_resize(THZRealTensor *tensor, THLongStorage *size, THLongStorage *stride);
void THZRealTensor_resizeAs(THZRealTensor *tensor, THZRealTensor *src);
void THZRealTensor_resize1d(THZRealTensor *tensor, long size0_);
void THZRealTensor_resize2d(THZRealTensor *tensor, long size0_, long size1_);
void THZRealTensor_resize3d(THZRealTensor *tensor, long size0_, long size1_, long size2_);
void THZRealTensor_resize4d(THZRealTensor *tensor, long size0_, long size1_, long size2_, long size3_);
void THZRealTensor_resize5d(THZRealTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

void THZRealTensor_set(THZRealTensor *self, THZRealTensor *src);
void THZRealTensor_setStorage(THZRealTensor *self, THZRealStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
void THZRealTensor_setStorage1d(THZRealTensor *self, THZRealStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
void THZRealTensor_setStorage2d(THZRealTensor *self, THZRealStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
void THZRealTensor_setStorage3d(THZRealTensor *self, THZRealStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
void THZRealTensor_setStorage4d(THZRealTensor *self, THZRealStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

void THZRealTensor_narrow(THZRealTensor *self, THZRealTensor *src, int dimension_, long firstIndex_, long size_);
void THZRealTensor_select(THZRealTensor *self, THZRealTensor *src, int dimension_, long sliceIndex_);
void THZRealTensor_transpose(THZRealTensor *self, THZRealTensor *src, int dimension1_, int dimension2_);
void THZRealTensor_unfold(THZRealTensor *self, THZRealTensor *src, int dimension_, long size_, long step_);

void THZRealTensor_squeeze(THZRealTensor *self, THZRealTensor *src);
void THZRealTensor_squeeze1d(THZRealTensor *self, THZRealTensor *src, int dimension_);

int THZRealTensor_isContiguous(const THZRealTensor *self);
int THZRealTensor_isSameSizeAs(const THZRealTensor *self, const THZRealTensor *src);
long THZRealTensor_nElement(const THZRealTensor *self);

void THZRealTensor_retain(THZRealTensor *self);
void THZRealTensor_free(THZRealTensor *self);
void THZRealTensor_freeCopyTo(THZRealTensor *self, THZRealTensor *dst);


void THZRealTensor_set1d(THZRealTensor *tensor, long x0, real value);
void THZRealTensor_set2d(THZRealTensor *tensor, long x0, long x1, real value);
void THZRealTensor_set3d(THZRealTensor *tensor, long x0, long x1, long x2, real value);
void THZRealTensor_set4d(THZRealTensor *tensor, long x0, long x1, long x2, long x3, real value);

real THZRealTensor_get1d(const THZRealTensor *tensor, long x0);
real THZRealTensor_get2d(const THZRealTensor *tensor, long x0, long x1);
real THZRealTensor_get3d(const THZRealTensor *tensor, long x0, long x1, long x2);
real THZRealTensor_get4d(const THZRealTensor *tensor, long x0, long x1, long x2, long x3);

void THZRealTensor_copy(THZRealTensor *tensor, THZRealTensor *src);
void THZRealTensor_copyByte(THZRealTensor *tensor, struct THByteTensor *src);
void THZRealTensor_copyChar(THZRealTensor *tensor, struct THCharTensor *src);
void THZRealTensor_copyShort(THZRealTensor *tensor, struct THShortTensor *src);
void THZRealTensor_copyInt(THZRealTensor *tensor, struct THIntTensor *src);
void THZRealTensor_copyLong(THZRealTensor *tensor, struct THLongTensor *src);
void THZRealTensor_copyFloat(THZRealTensor *tensor, struct THFloatTensor *src);
void THZRealTensor_copyDouble(THZRealTensor *tensor, struct THDoubleTensor *src);


void THZRealTensor_fill(THZRealTensor *r_, real value);
void THZRealTensor_zero(THZRealTensor *r_);

void THZRealTensor_maskedFill(THZRealTensor *tensor, THByteTensor *mask, real value);
void THZRealTensor_maskedCopy(THZRealTensor *tensor, THByteTensor *mask, THZRealTensor* src);
void THZRealTensor_maskedSelect(THZRealTensor *tensor, THZRealTensor* src, THByteTensor *mask);

void THZRealTensor_indexSelect(THZRealTensor *tensor, THZRealTensor *src, int dim, THLongTensor *index);
void THZRealTensor_indexCopy(THZRealTensor *tensor, int dim, THLongTensor *index, THZRealTensor *src);
void THZRealTensor_indexFill(THZRealTensor *tensor, int dim, THLongTensor *index, real val);

accreal THZRealTensor_dot(THZRealTensor *t, THZRealTensor *src);

real THZRealTensor_minall(THZRealTensor *t);
real THZRealTensor_maxall(THZRealTensor *t);
accreal THZRealTensor_sumall(THZRealTensor *t);

void THZRealTensor_add(THZRealTensor *r_, THZRealTensor *t, real value);
void THZRealTensor_mul(THZRealTensor *r_, THZRealTensor *t, real value);
void THZRealTensor_div(THZRealTensor *r_, THZRealTensor *t, real value);

void THZRealTensor_cadd(THZRealTensor *r_, THZRealTensor *t, real value, THZRealTensor *src);
void THZRealTensor_cmul(THZRealTensor *r_, THZRealTensor *t, THZRealTensor *src);
void THZRealTensor_cdiv(THZRealTensor *r_, THZRealTensor *t, THZRealTensor *src);

void THZRealTensor_addcmul(THZRealTensor *r_, THZRealTensor *t, real value, THZRealTensor *src1, THZRealTensor *src2);
void THZRealTensor_addcdiv(THZRealTensor *r_, THZRealTensor *t, real value, THZRealTensor *src1, THZRealTensor *src2);

void THZRealTensor_addmv(THZRealTensor *r_, real beta, THZRealTensor *t, real alpha, THZRealTensor *mat, THZRealTensor *vec);
void THZRealTensor_addmm(THZRealTensor *r_, real beta, THZRealTensor *t, real alpha, THZRealTensor *mat1, THZRealTensor *mat2);
void THZRealTensor_addr(THZRealTensor *r_, real beta, THZRealTensor *t, real alpha, THZRealTensor *vec1, THZRealTensor *vec2);
void THZRealTensor_addru(THZRealTensor *r_, real beta, THZRealTensor *t, real alpha, THZRealTensor *vec1, THZRealTensor *vec2);

void THZRealTensor_match(THZRealTensor *r_, THZRealTensor *m1, THZRealTensor *m2, real gain);

long THZRealTensor_numel(THZRealTensor *t);
void THZRealTensor_max(THZRealTensor *values_, THLongTensor *indices_, THZRealTensor *t, int dimension);
void THZRealTensor_min(THZRealTensor *values_, THLongTensor *indices_, THZRealTensor *t, int dimension);
void THZRealTensor_sum(THZRealTensor *r_, THZRealTensor *t, int dimension);
void THZRealTensor_prod(THZRealTensor *r_, THZRealTensor *t, int dimension);
void THZRealTensor_cumsum(THZRealTensor *r_, THZRealTensor *t, int dimension);
void THZRealTensor_cumprod(THZRealTensor *r_, THZRealTensor *t, int dimension);
accreal THZRealTensor_trace(THZRealTensor *t);
void THZRealTensor_cross(THZRealTensor *r_, THZRealTensor *a, THZRealTensor *b, int dimension);

void THZRealTensor_zeros(THZRealTensor *r_, THLongStorage *size);
void THZRealTensor_ones(THZRealTensor *r_, THLongStorage *size);
void THZRealTensor_diag(THZRealTensor *r_, THZRealTensor *t, int k);
void THZRealTensor_eye(THZRealTensor *r_, long n, long m);

void THZRealTensor_reshape(THZRealTensor *r_, THZRealTensor *t, THLongStorage *size);
void THZRealTensor_sort(THZRealTensor *rt_, THLongTensor *ri_, THZRealTensor *t, int dimension, int descendingOrder);
void THZRealTensor_tril(THZRealTensor *r_, THZRealTensor *t, long k);
void THZRealTensor_triu(THZRealTensor *r_, THZRealTensor *t, long k);
void THZRealTensor_cat(THZRealTensor *r_, THZRealTensor *ta, THZRealTensor *tb, int dimension);

void THZRealTensor_ltValue(THByteTensor *r_, THZRealTensor* t, real value);
void THZRealTensor_leValue(THByteTensor *r_, THZRealTensor* t, real value);
void THZRealTensor_gtValue(THByteTensor *r_, THZRealTensor* t, real value);
void THZRealTensor_geValue(THByteTensor *r_, THZRealTensor* t, real value);
void THZRealTensor_neValue(THByteTensor *r_, THZRealTensor* t, real value);
void THZRealTensor_eqValue(THByteTensor *r_, THZRealTensor* t, real value);

void THZRealTensor_ltTensor(THByteTensor *r_, THZRealTensor *ta, THZRealTensor *tb);
void THZRealTensor_leTensor(THByteTensor *r_, THZRealTensor *ta, THZRealTensor *tb);
void THZRealTensor_gtTensor(THByteTensor *r_, THZRealTensor *ta, THZRealTensor *tb);
void THZRealTensor_geTensor(THByteTensor *r_, THZRealTensor *ta, THZRealTensor *tb);
void THZRealTensor_neTensor(THByteTensor *r_, THZRealTensor *ta, THZRealTensor *tb);
void THZRealTensor_eqTensor(THByteTensor *r_, THZRealTensor *ta, THZRealTensor *tb);

void THZRealTensor_log(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_exp(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_cos(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_acos(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_cosh(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_sin(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_asin(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_sinh(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_tan(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_atan(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_tanh(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_pow(THZRealTensor *r_, THZRealTensor *t, real value);
void THZRealTensor_sqrt(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_conj(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_proj(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_zabs(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_zarg(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_zre(THZRealTensor *r_, THZRealTensor *t);
void THZRealTensor_zim(THZRealTensor *r_, THZRealTensor *t);


void THZRealTensor_mean(THZRealTensor *r_, THZRealTensor *t, int dimension);
void THZRealTensor_std(THZRealTensor *r_, THZRealTensor *t, int dimension, int flag);
void THZRealTensor_var(THZRealTensor *r_, THZRealTensor *t, int dimension, int flag);
void THZRealTensor_norm(THZRealTensor *r_, THZRealTensor *t, real value, int dimension);
void THZRealTensor_renorm(THZRealTensor *r_, THZRealTensor *t, real value, int dimension, real maxnorm);
accreal THZRealTensor_dist(THZRealTensor *a, THZRealTensor *b, real value);

accreal THZRealTensor_meanall(THZRealTensor *self);
accreal THZRealTensor_varall(THZRealTensor *self);
accreal THZRealTensor_stdall(THZRealTensor *self);
accreal THZRealTensor_normall(THZRealTensor *t, real value);

void THZRealTensor_Float_abs(THFloatTensor *r_, THZRealTensor *t);
void THZRealTensor_Float_arg(THFloatTensor *r_, THZRealTensor *t);
void THZRealTensor_Float_re(THFloatTensor *r_, THZRealTensor *t);
void THZRealTensor_Float_im(THFloatTensor *r_, THZRealTensor *t);
void THZRealTensor_Double_abs(THDoubleTensor *r_, THZRealTensor *t);
void THZRealTensor_Double_arg(THDoubleTensor *r_, THZRealTensor *t);
void THZRealTensor_Double_re(THDoubleTensor *r_, THZRealTensor *t);
void THZRealTensor_Double_im(THDoubleTensor *r_, THZRealTensor *t);

void THZRealTensor_validXCorr2Dptr(real *r_,
                                    real alpha,
                                    real *t_, long ir, long ic,
                                    real *k_, long kr, long kc,
                                    long sr, long sc);

void THZRealTensor_validConv2Dptr(real *r_,
                                   real alpha,
                                   real *t_, long ir, long ic,
                                   real *k_, long kr, long kc,
                                   long sr, long sc);

void THZRealTensor_fullXCorr2Dptr(real *r_,
                                   real alpha,
                                   real *t_, long ir, long ic,
                                   real *k_, long kr, long kc,
                                   long sr, long sc);

void THZRealTensor_fullConv2Dptr(real *r_,
                                  real alpha,
                                  real *t_, long ir, long ic,
                                  real *k_, long kr, long kc,
                                  long sr, long sc);

void THZRealTensor_validXCorr2DRevptr(real *r_,
                                       real alpha,
                                       real *t_, long ir, long ic,
                                       real *k_, long kr, long kc,
                                       long sr, long sc);

void THZRealTensor_conv2DRevger(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long srow, long scol);
void THZRealTensor_conv2DRevgerm(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long srow, long scol);
void THZRealTensor_conv2Dger(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long srow, long scol, const char *vf, const char *xc);
void THZRealTensor_conv2Dmv(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long srow, long scol, const char *vf, const char *xc);
void THZRealTensor_conv2Dmm(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long srow, long scol, const char *vf, const char *xc);
void THZRealTensor_conv2Dmul(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long srow, long scol, const char *vf, const char *xc);
void THZRealTensor_conv2Dcmul(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long srow, long scol, const char *vf, const char *xc);

void THZRealTensor_validXCorr3Dptr(real *r_,
                                    real alpha,
                                    real *t_, long it, long ir, long ic,
                                    real *k_, long kt, long kr, long kc,
                                    long st, long sr, long sc);

void THZRealTensor_validConv3Dptr(real *r_,
                                   real alpha,
                                   real *t_, long it, long ir, long ic,
                                   real *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

void THZRealTensor_fullXCorr3Dptr(real *r_,
                                   real alpha,
                                   real *t_, long it, long ir, long ic,
                                   real *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

void THZRealTensor_fullConv3Dptr(real *r_,
                                  real alpha,
                                  real *t_, long it, long ir, long ic,
                                  real *k_, long kt, long kr, long kc,
                                  long st, long sr, long sc);

void THZRealTensor_validXCorr3DRevptr(real *r_,
                                       real alpha,
                                       real *t_, long it, long ir, long ic,
                                       real *k_, long kt, long kr, long kc,
                                       long st, long sr, long sc);

void THZRealTensor_conv3DRevger(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long sdepth, long srow, long scol);
void THZRealTensor_conv3Dger(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
void THZRealTensor_conv3Dmv(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
void THZRealTensor_conv3Dmul(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
void THZRealTensor_conv3Dcmul(THZRealTensor *r_, real beta, real alpha, THZRealTensor *t_, THZRealTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
void THZRealTensor_gesv(THZRealTensor *rb_, THZRealTensor *ra_, THZRealTensor *b_, THZRealTensor *a_);
void THZRealTensor_gels(THZRealTensor *rb_, THZRealTensor *ra_, THZRealTensor *b_, THZRealTensor *a_);
void THZRealTensor_syev(THZRealTensor *re_, THZRealTensor *rv_, THZRealTensor *a_, const char *jobz, const char *uplo);
void THZRealTensor_geev(THZRealTensor *re_, THZRealTensor *rv_, THZRealTensor *a_, const char *jobvr);
void THZRealTensor_gesvd(THZRealTensor *ru_, THZRealTensor *rs_, THZRealTensor *rv_, THZRealTensor *a, const char *jobu);
void THZRealTensor_gesvd2(THZRealTensor *ru_, THZRealTensor *rs_, THZRealTensor *rv_, THZRealTensor *ra_, THZRealTensor *a, const char *jobu);
void THZRealTensor_getri(THZRealTensor *ra_, THZRealTensor *a);
void THZRealTensor_potri(THZRealTensor *ra_, THZRealTensor *a);
void THZRealTensor_potrf(THZRealTensor *ra_, THZRealTensor *a);
]])

local ok, C = pcall(ffi.load, 'torch_oss_THZ')
if not ok then
  C = ffi.load('THZ')
end
return C

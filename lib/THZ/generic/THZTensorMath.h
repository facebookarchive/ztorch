/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorMath.h"
#else

THZ_API void THZTensor_(fill)(THZTensor *r_, real value);
THZ_API void THZTensor_(zero)(THZTensor *r_);

THZ_API void THZTensor_(maskedFill)(THZTensor *tensor, THByteTensor *mask, real value);
THZ_API void THZTensor_(maskedCopy)(THZTensor *tensor, THByteTensor *mask, THZTensor* src);
THZ_API void THZTensor_(maskedSelect)(THZTensor *tensor, THZTensor* src, THByteTensor *mask);

THZ_API void THZTensor_(indexSelect)(THZTensor *tensor, THZTensor *src, int dim, THLongTensor *index);
THZ_API void THZTensor_(indexCopy)(THZTensor *tensor, int dim, THLongTensor *index, THZTensor *src);
THZ_API void THZTensor_(indexFill)(THZTensor *tensor, int dim, THLongTensor *index, real val);

THZ_API accreal THZTensor_(dot)(THZTensor *t, THZTensor *src);

THZ_API real THZTensor_(minall)(THZTensor *t);
THZ_API real THZTensor_(maxall)(THZTensor *t);
THZ_API accreal THZTensor_(sumall)(THZTensor *t);

THZ_API void THZTensor_(add)(THZTensor *r_, THZTensor *t, real value);
THZ_API void THZTensor_(mul)(THZTensor *r_, THZTensor *t, real value);
THZ_API void THZTensor_(div)(THZTensor *r_, THZTensor *t, real value);

THZ_API void THZTensor_(cadd)(THZTensor *r_, THZTensor *t, real value, THZTensor *src);
THZ_API void THZTensor_(cmul)(THZTensor *r_, THZTensor *t, THZTensor *src);
THZ_API void THZTensor_(cdiv)(THZTensor *r_, THZTensor *t, THZTensor *src);

THZ_API void THZTensor_(addcmul)(THZTensor *r_, THZTensor *t, real value, THZTensor *src1, THZTensor *src2);
THZ_API void THZTensor_(addcdiv)(THZTensor *r_, THZTensor *t, real value, THZTensor *src1, THZTensor *src2);

THZ_API void THZTensor_(addmv)(THZTensor *r_, real beta, THZTensor *t, real alpha, THZTensor *mat,  THZTensor *vec);
THZ_API void THZTensor_(addmm)(THZTensor *r_, real beta, THZTensor *t, real alpha, THZTensor *mat1, THZTensor *mat2);
THZ_API void THZTensor_(addr)(THZTensor *r_,  real beta, THZTensor *t, real alpha, THZTensor *vec1, THZTensor *vec2);

THZ_API void THZTensor_(match)(THZTensor *r_, THZTensor *m1, THZTensor *m2, real gain);

THZ_API long THZTensor_(numel)(THZTensor *t);
THZ_API void THZTensor_(max)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension);
THZ_API void THZTensor_(min)(THZTensor *values_, THLongTensor *indices_, THZTensor *t, int dimension);
THZ_API void THZTensor_(sum)(THZTensor *r_, THZTensor *t, int dimension);
THZ_API void THZTensor_(prod)(THZTensor *r_, THZTensor *t, int dimension);
THZ_API void THZTensor_(cumsum)(THZTensor *r_, THZTensor *t, int dimension);
THZ_API void THZTensor_(cumprod)(THZTensor *r_, THZTensor *t, int dimension);
THZ_API accreal THZTensor_(trace)(THZTensor *t);
THZ_API void THZTensor_(cross)(THZTensor *r_, THZTensor *a, THZTensor *b, int dimension);

THZ_API void THZTensor_(zeros)(THZTensor *r_, THLongStorage *size);
THZ_API void THZTensor_(ones)(THZTensor *r_, THLongStorage *size);
THZ_API void THZTensor_(diag)(THZTensor *r_, THZTensor *t, int k);
THZ_API void THZTensor_(eye)(THZTensor *r_, long n, long m);

THZ_API void THZTensor_(reshape)(THZTensor *r_, THZTensor *t, THLongStorage *size);
THZ_API void THZTensor_(sort)(THZTensor *rt_, THLongTensor *ri_, THZTensor *t, int dimension, int descendingOrder);
THZ_API void THZTensor_(tril)(THZTensor *r_, THZTensor *t, long k);
THZ_API void THZTensor_(triu)(THZTensor *r_, THZTensor *t, long k);
THZ_API void THZTensor_(cat)(THZTensor *r_, THZTensor *ta, THZTensor *tb, int dimension);

THZ_API void THZTensor_(ltValue)(THByteTensor *r_, THZTensor* t, real value);
THZ_API void THZTensor_(leValue)(THByteTensor *r_, THZTensor* t, real value);
THZ_API void THZTensor_(gtValue)(THByteTensor *r_, THZTensor* t, real value);
THZ_API void THZTensor_(geValue)(THByteTensor *r_, THZTensor* t, real value);
THZ_API void THZTensor_(neValue)(THByteTensor *r_, THZTensor* t, real value);
THZ_API void THZTensor_(eqValue)(THByteTensor *r_, THZTensor* t, real value);

THZ_API void THZTensor_(ltTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);
THZ_API void THZTensor_(leTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);
THZ_API void THZTensor_(gtTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);
THZ_API void THZTensor_(geTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);
THZ_API void THZTensor_(neTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);
THZ_API void THZTensor_(eqTensor)(THByteTensor *r_, THZTensor *ta, THZTensor *tb);

THZ_API void THZTensor_(log)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(exp)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(cos)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(acos)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(cosh)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(sin)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(asin)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(sinh)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(tan)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(atan)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(tanh)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(pow)(THZTensor *r_, THZTensor *t, real value);
THZ_API void THZTensor_(sqrt)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(conj)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(proj)(THZTensor *r_, THZTensor *t);


THZ_API void THZTensor_(mean)(THZTensor *r_, THZTensor *t, int dimension);
THZ_API void THZTensor_(std)(THZTensor *r_, THZTensor *t, int dimension, int flag);
THZ_API void THZTensor_(var)(THZTensor *r_, THZTensor *t, int dimension, int flag);
THZ_API void THZTensor_(norm)(THZTensor *r_, THZTensor *t, real value, int dimension);
THZ_API void THZTensor_(renorm)(THZTensor *r_, THZTensor *t, real value, int dimension, real maxnorm);
THZ_API accreal THZTensor_(dist)(THZTensor *a, THZTensor *b, real value);

THZ_API accreal THZTensor_(meanall)(THZTensor *self);
THZ_API accreal THZTensor_(varall)(THZTensor *self);
THZ_API accreal THZTensor_(stdall)(THZTensor *self);
THZ_API accreal THZTensor_(normall)(THZTensor *t, real value);

THZ_API void THZTensor_(zabs)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(zarg)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(zre)(THZTensor *r_, THZTensor *t);
THZ_API void THZTensor_(zim)(THZTensor *r_, THZTensor *t);

THZ_API void THZTensor_(Float_abs)(THFloatTensor *r_, THZTensor *t);
THZ_API void THZTensor_(Float_arg)(THFloatTensor *r_, THZTensor *t);
THZ_API void THZTensor_(Float_re)(THFloatTensor *r_, THZTensor *t);
THZ_API void THZTensor_(Float_im)(THFloatTensor *r_, THZTensor *t);
THZ_API void THZTensor_(Double_abs)(THDoubleTensor *r_, THZTensor *t);
THZ_API void THZTensor_(Double_arg)(THDoubleTensor *r_, THZTensor *t);
THZ_API void THZTensor_(Double_re)(THDoubleTensor *r_, THZTensor *t);
THZ_API void THZTensor_(Double_im)(THDoubleTensor *r_, THZTensor *t);

#endif

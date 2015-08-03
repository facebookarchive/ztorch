/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorCopy.c"
#else

void THZTensor_(copy)(THZTensor *tensor, THZTensor *src)
{
  TH_TENSOR_APPLY2(real, tensor, real, src, *tensor_data = (real)(*src_data);)
}

#define IMPLEMENT_THZTensor_COPY(TYPENAMESRC, TYPE_SRC)			\
void THZTensor_(copy##TYPENAMESRC)(THZTensor *tensor, TH##TYPENAMESRC##Tensor *src) \
{									\
 int srcdims = TH##TYPENAMESRC##Tensor_nDimension(src);			\
 int selfdims = THZTensor_(nDimension)(tensor);				\
 int copyimag = 0;							\
 int i;									\
 if (srcdims == (selfdims + 1))  {					\
   for (i=0; i < selfdims; i++) {						\
     if (THZTensor_(size)(tensor, i) != TH##TYPENAMESRC##Tensor_size(src, i)) { \
       break;								\
     }									\
     if (i == selfdims-1) {						\
       copyimag = 1;							\
     }									\
   }									\
 }									\
									\
if (copyimag == 1) {							\
  src = TH##TYPENAMESRC##Tensor_newContiguous(src);			\
  TYPE_SRC *src_data = TH##TYPENAMESRC##Tensor_data(src);		\
    TH_TENSOR_APPLY(real, tensor,					\
		    *tensor_data = (*src_data + *(src_data+1) * 1i);	\
		    src_data +=2;);					\
  TH##TYPENAMESRC##Tensor_free(src);					\
 } else {								\
  TH_TENSOR_APPLY2(real, tensor, TYPE_SRC, src, *tensor_data = (real)(*src_data);) \
    }									\
}

IMPLEMENT_THZTensor_COPY(Byte, unsigned char)
IMPLEMENT_THZTensor_COPY(Char, char)
IMPLEMENT_THZTensor_COPY(Short, short)
IMPLEMENT_THZTensor_COPY(Int, int)
IMPLEMENT_THZTensor_COPY(Long, long)
IMPLEMENT_THZTensor_COPY(Float, float)
IMPLEMENT_THZTensor_COPY(Double, double)

#endif

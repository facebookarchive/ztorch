/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorCopy.h"
#else

/* Support for copy between different Tensor types */

THZ_API void THZTensor_(copy)(THZTensor *tensor, THZTensor *src);
THZ_API void THZTensor_(copyByte)(THZTensor *tensor, struct THByteTensor *src);
THZ_API void THZTensor_(copyChar)(THZTensor *tensor, struct THCharTensor *src);
THZ_API void THZTensor_(copyShort)(THZTensor *tensor, struct THShortTensor *src);
THZ_API void THZTensor_(copyInt)(THZTensor *tensor, struct THIntTensor *src);
THZ_API void THZTensor_(copyLong)(THZTensor *tensor, struct THLongTensor *src);
THZ_API void THZTensor_(copyFloat)(THZTensor *tensor, struct THFloatTensor *src);
THZ_API void THZTensor_(copyDouble)(THZTensor *tensor, struct THDoubleTensor *src);

#endif

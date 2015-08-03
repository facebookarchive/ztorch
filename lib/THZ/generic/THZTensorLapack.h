/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorLapack.h"
#else

THZ_API void THZTensor_(gesv)(THZTensor *rb_, THZTensor *ra_, THZTensor *b_, THZTensor *a_);
THZ_API void THZTensor_(gels)(THZTensor *rb_, THZTensor *ra_, THZTensor *b_, THZTensor *a_);
THZ_API void THZTensor_(syev)(THZTensor *re_, THZTensor *rv_, THZTensor *a_, const char *jobz, const char *uplo);
THZ_API void THZTensor_(geev)(THZTensor *re_, THZTensor *rv_, THZTensor *a_, const char *jobvr);
THZ_API void THZTensor_(gesvd)(THZTensor *ru_, THZTensor *rs_, THZTensor *rv_, THZTensor *a, const char *jobu);
THZ_API void THZTensor_(gesvd2)(THZTensor *ru_, THZTensor *rs_, THZTensor *rv_, THZTensor *ra_, THZTensor *a, const char *jobu);
THZ_API void THZTensor_(getri)(THZTensor *ra_, THZTensor *a);
THZ_API void THZTensor_(potri)(THZTensor *ra_, THZTensor *a);
THZ_API void THZTensor_(potrf)(THZTensor *ra_, THZTensor *a);

#endif

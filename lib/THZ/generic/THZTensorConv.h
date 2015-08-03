/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensorConv.h"
#else


THZ_API void THZTensor_(validXCorr2Dptr)(real *r_,
                                    real alpha,
                                    real *t_, long ir, long ic,
                                    real *k_, long kr, long kc,
                                    long sr, long sc);

THZ_API void THZTensor_(validConv2Dptr)(real *r_,
                                   real alpha,
                                   real *t_, long ir, long ic,
                                   real *k_, long kr, long kc,
                                   long sr, long sc);

THZ_API void THZTensor_(fullXCorr2Dptr)(real *r_,
                                   real alpha,
                                   real *t_, long ir, long ic,
                                   real *k_, long kr, long kc,
                                   long sr, long sc);

THZ_API void THZTensor_(fullConv2Dptr)(real *r_,
                                  real alpha,
                                  real *t_, long ir, long ic,
                                  real *k_, long kr, long kc,
                                  long sr, long sc);

THZ_API void THZTensor_(validXCorr2DRevptr)(real *r_,
                                       real alpha,
                                       real *t_, long ir, long ic,
                                       real *k_, long kr, long kc,
                                       long sr, long sc);

THZ_API void THZTensor_(conv2DRevger)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol);
THZ_API void THZTensor_(conv2DRevgerm)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol);
THZ_API void THZTensor_(conv2Dger)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol, const char *vf, const char *xc);
THZ_API void THZTensor_(conv2Dmv)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol, const char *vf, const char *xc);
THZ_API void THZTensor_(conv2Dmm)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol, const char *vf, const char *xc);
THZ_API void THZTensor_(conv2Dmul)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol, const char *vf, const char *xc);
THZ_API void THZTensor_(conv2Dcmul)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long srow, long scol, const char *vf, const char *xc);

THZ_API void THZTensor_(validXCorr3Dptr)(real *r_,
                                    real alpha,
                                    real *t_, long it, long ir, long ic,
                                    real *k_, long kt, long kr, long kc,
                                    long st, long sr, long sc);

THZ_API void THZTensor_(validConv3Dptr)(real *r_,
                                   real alpha,
                                   real *t_, long it, long ir, long ic,
                                   real *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

THZ_API void THZTensor_(fullXCorr3Dptr)(real *r_,
                                   real alpha,
                                   real *t_, long it, long ir, long ic,
                                   real *k_, long kt, long kr, long kc,
                                   long st, long sr, long sc);

THZ_API void THZTensor_(fullConv3Dptr)(real *r_,
                                  real alpha,
                                  real *t_, long it, long ir, long ic,
                                  real *k_, long kt, long kr, long kc,
                                  long st, long sr, long sc);

THZ_API void THZTensor_(validXCorr3DRevptr)(real *r_,
                                       real alpha,
                                       real *t_, long it, long ir, long ic,
                                       real *k_, long kt, long kr, long kc,
                                       long st, long sr, long sc);

THZ_API void THZTensor_(conv3DRevger)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long sdepth, long srow, long scol);
THZ_API void THZTensor_(conv3Dger)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
THZ_API void THZTensor_(conv3Dmv)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
THZ_API void THZTensor_(conv3Dmul)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);
THZ_API void THZTensor_(conv3Dcmul)(THZTensor *r_, real beta, real alpha, THZTensor *t_, THZTensor *k_, long sdepth, long srow, long scol, const char *vf, const char *xc);

#endif

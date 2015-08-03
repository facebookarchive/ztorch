/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensor.h"
#else

/* a la lua? dim, storageoffset, ...  et les methodes ? */

#define THZ_TENSOR_REFCOUNTED 1

typedef struct THZTensor
{
    long *size;
    long *stride;
    int nDimension;

    THZStorage *storage;
    long storageOffset;
    int refcount;

    char flag;

} THZTensor;


/**** access methods ****/
THZ_API THZStorage* THZTensor_(storage)(const THZTensor *self);
THZ_API long THZTensor_(storageOffset)(const THZTensor *self);
THZ_API int THZTensor_(nDimension)(const THZTensor *self);
THZ_API long THZTensor_(size)(const THZTensor *self, int dim);
THZ_API long THZTensor_(stride)(const THZTensor *self, int dim);
THZ_API THLongStorage *THZTensor_(newSizeOf)(THZTensor *self);
THZ_API THLongStorage *THZTensor_(newStrideOf)(THZTensor *self);
THZ_API real *THZTensor_(data)(const THZTensor *self);

THZ_API void THZTensor_(setFlag)(THZTensor *self, const char flag);
THZ_API void THZTensor_(clearFlag)(THZTensor *self, const char flag);


/**** creation methods ****/
THZ_API THZTensor *THZTensor_(new)(void);
THZ_API THZTensor *THZTensor_(newWithTensor)(THZTensor *tensor);
/* stride might be NULL */
THZ_API THZTensor *THZTensor_(newWithStorage)(THZStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THZ_API THZTensor *THZTensor_(newWithStorage1d)(THZStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
THZ_API THZTensor *THZTensor_(newWithStorage2d)(THZStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
THZ_API THZTensor *THZTensor_(newWithStorage3d)(THZStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
THZ_API THZTensor *THZTensor_(newWithStorage4d)(THZStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);

/* stride might be NULL */
THZ_API THZTensor *THZTensor_(newWithSize)(THLongStorage *size_, THLongStorage *stride_);
THZ_API THZTensor *THZTensor_(newWithSize1d)(long size0_);
THZ_API THZTensor *THZTensor_(newWithSize2d)(long size0_, long size1_);
THZ_API THZTensor *THZTensor_(newWithSize3d)(long size0_, long size1_, long size2_);
THZ_API THZTensor *THZTensor_(newWithSize4d)(long size0_, long size1_, long size2_, long size3_);

THZ_API THZTensor *THZTensor_(newClone)(THZTensor *self);
THZ_API THZTensor *THZTensor_(newContiguous)(THZTensor *tensor);
THZ_API THZTensor *THZTensor_(newSelect)(THZTensor *tensor, int dimension_, long sliceIndex_);
THZ_API THZTensor *THZTensor_(newNarrow)(THZTensor *tensor, int dimension_, long firstIndex_, long size_);
THZ_API THZTensor *THZTensor_(newTranspose)(THZTensor *tensor, int dimension1_, int dimension2_);
THZ_API THZTensor *THZTensor_(newUnfold)(THZTensor *tensor, int dimension_, long size_, long step_);

THZ_API void THZTensor_(resize)(THZTensor *tensor, THLongStorage *size, THLongStorage *stride);
THZ_API void THZTensor_(resizeAs)(THZTensor *tensor, THZTensor *src);
THZ_API void THZTensor_(resize1d)(THZTensor *tensor, long size0_);
THZ_API void THZTensor_(resize2d)(THZTensor *tensor, long size0_, long size1_);
THZ_API void THZTensor_(resize3d)(THZTensor *tensor, long size0_, long size1_, long size2_);
THZ_API void THZTensor_(resize4d)(THZTensor *tensor, long size0_, long size1_, long size2_, long size3_);
THZ_API void THZTensor_(resize5d)(THZTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);

THZ_API void THZTensor_(set)(THZTensor *self, THZTensor *src);
THZ_API void THZTensor_(setStorage)(THZTensor *self, THZStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THZ_API void THZTensor_(setStorage1d)(THZTensor *self, THZStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
THZ_API void THZTensor_(setStorage2d)(THZTensor *self, THZStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
THZ_API void THZTensor_(setStorage3d)(THZTensor *self, THZStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
THZ_API void THZTensor_(setStorage4d)(THZTensor *self, THZStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

THZ_API void THZTensor_(narrow)(THZTensor *self, THZTensor *src, int dimension_, long firstIndex_, long size_);
THZ_API void THZTensor_(select)(THZTensor *self, THZTensor *src, int dimension_, long sliceIndex_);
THZ_API void THZTensor_(transpose)(THZTensor *self, THZTensor *src, int dimension1_, int dimension2_);
THZ_API void THZTensor_(unfold)(THZTensor *self, THZTensor *src, int dimension_, long size_, long step_);

THZ_API void THZTensor_(squeeze)(THZTensor *self, THZTensor *src);
THZ_API void THZTensor_(squeeze1d)(THZTensor *self, THZTensor *src, int dimension_);

THZ_API int THZTensor_(isContiguous)(const THZTensor *self);
THZ_API int THZTensor_(isSameSizeAs)(const THZTensor *self, const THZTensor *src);
THZ_API long THZTensor_(nElement)(const THZTensor *self);

THZ_API void THZTensor_(retain)(THZTensor *self);
THZ_API void THZTensor_(free)(THZTensor *self);
THZ_API void THZTensor_(freeCopyTo)(THZTensor *self, THZTensor *dst);

/* Slow access methods [check everything] */
THZ_API void THZTensor_(set1d)(THZTensor *tensor, long x0, real value);
THZ_API void THZTensor_(set2d)(THZTensor *tensor, long x0, long x1, real value);
THZ_API void THZTensor_(set3d)(THZTensor *tensor, long x0, long x1, long x2, real value);
THZ_API void THZTensor_(set4d)(THZTensor *tensor, long x0, long x1, long x2, long x3, real value);

THZ_API real THZTensor_(get1d)(const THZTensor *tensor, long x0);
THZ_API real THZTensor_(get2d)(const THZTensor *tensor, long x0, long x1);
THZ_API real THZTensor_(get3d)(const THZTensor *tensor, long x0, long x1, long x2);
THZ_API real THZTensor_(get4d)(const THZTensor *tensor, long x0, long x1, long x2, long x3);

#endif

/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZStorage.h"
#else

#define THZ_STORAGE_REFCOUNTED 1
#define THZ_STORAGE_RESIZABLE  2
#define THZ_STORAGE_FREEMEM    4

typedef struct THZStorage
{
    real *data;
    long size;
    int refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
} THZStorage;

THZ_API real* THZStorage_(data)(const THZStorage*);
THZ_API long THZStorage_(size)(const THZStorage*);

/* slow access -- checks everything */
THZ_API void THZStorage_(set)(THZStorage*, long, real);
THZ_API real THZStorage_(get)(const THZStorage*, long);

THZ_API THZStorage* THZStorage_(new)(void);
THZ_API THZStorage* THZStorage_(newWithSize)(long size);
THZ_API THZStorage* THZStorage_(newWithSize1)(real);
THZ_API THZStorage* THZStorage_(newWithSize2)(real, real);
THZ_API THZStorage* THZStorage_(newWithSize3)(real, real, real);
THZ_API THZStorage* THZStorage_(newWithSize4)(real, real, real, real);
THZ_API THZStorage* THZStorage_(newWithMapping)(const char *filename, long size, int shared);

/* takes ownership of data */
THZ_API THZStorage* THZStorage_(newWithData)(real *data, long size);

THZ_API THZStorage* THZStorage_(newWithAllocator)(long size,
                                               THAllocator* allocator,
                                               void *allocatorContext);
THZ_API THZStorage* THZStorage_(newWithDataAndAllocator)(
    real* data, long size, THAllocator* allocator, void *allocatorContext);

/* should not differ with API */
THZ_API void THZStorage_(setFlag)(THZStorage *storage, const char flag);
THZ_API void THZStorage_(clearFlag)(THZStorage *storage, const char flag);
THZ_API void THZStorage_(retain)(THZStorage *storage);

/* might differ with other API (like CUDA) */
THZ_API void THZStorage_(free)(THZStorage *storage);
THZ_API void THZStorage_(resize)(THZStorage *storage, long size);
THZ_API void THZStorage_(fill)(THZStorage *storage, real value);

#endif

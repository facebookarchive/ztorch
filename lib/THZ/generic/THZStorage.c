/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZStorage.c"
#else

real* THZStorage_(data)(const THZStorage *self)
{
  return self->data;
}

long THZStorage_(size)(const THZStorage *self)
{
  return self->size;
}

THZStorage* THZStorage_(new)(void)
{
  return THZStorage_(newWithSize)(0);
}

THZStorage* THZStorage_(newWithSize)(long size)
{
  return THZStorage_(newWithAllocator)(size, &THDefaultAllocator, NULL);
}

THZStorage* THZStorage_(newWithAllocator)(long size,
                                        THAllocator *allocator,
                                        void *allocatorContext)
{
  THZStorage *storage = THAlloc(sizeof(THZStorage));
  storage->data = allocator->malloc(allocatorContext, sizeof(real)*size);
  storage->size = size;
  storage->refcount = 1;
  storage->flag = THZ_STORAGE_REFCOUNTED | THZ_STORAGE_RESIZABLE | THZ_STORAGE_FREEMEM;
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  return storage;
}

THZStorage* THZStorage_(newWithMapping)(const char *filename, long size, int shared)
{
  THMapAllocatorContext *ctx = THMapAllocatorContext_new(filename, shared);

  THZStorage *storage = THZStorage_(newWithAllocator)(size,
                                                    &THMapAllocator,
                                                    ctx);

  if(size <= 0)
    storage->size = THMapAllocatorContext_size(ctx)/sizeof(real);

  THZStorage_(clearFlag)(storage, THZ_STORAGE_RESIZABLE);

  return storage;
}

THZStorage* THZStorage_(newWithSize1)(real data0)
{
  THZStorage *self = THZStorage_(newWithSize)(1);
  self->data[0] = data0;
  return self;
}

THZStorage* THZStorage_(newWithSize2)(real data0, real data1)
{
  THZStorage *self = THZStorage_(newWithSize)(2);
  self->data[0] = data0;
  self->data[1] = data1;
  return self;
}

THZStorage* THZStorage_(newWithSize3)(real data0, real data1, real data2)
{
  THZStorage *self = THZStorage_(newWithSize)(3);
  self->data[0] = data0;
  self->data[1] = data1;
  self->data[2] = data2;
  return self;
}

THZStorage* THZStorage_(newWithSize4)(real data0, real data1, real data2, real data3)
{
  THZStorage *self = THZStorage_(newWithSize)(4);
  self->data[0] = data0;
  self->data[1] = data1;
  self->data[2] = data2;
  self->data[3] = data3;
  return self;
}

void THZStorage_(setFlag)(THZStorage *storage, const char flag)
{
  storage->flag |= flag;
}

void THZStorage_(clearFlag)(THZStorage *storage, const char flag)
{
  storage->flag &= ~flag;
}

void THZStorage_(retain)(THZStorage *storage)
{
  if(storage && (storage->flag & THZ_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&storage->refcount);
}

void THZStorage_(free)(THZStorage *storage)
{
  if(!storage)
    return;

  if((storage->flag & THZ_STORAGE_REFCOUNTED) && (THAtomicGet(&storage->refcount) > 0))
  {
    if(THAtomicDecrementRef(&storage->refcount))
    {
      if(storage->flag & THZ_STORAGE_FREEMEM)
        storage->allocator->free(storage->allocatorContext, storage->data);
      THFree(storage);
    }
  }
}

THZStorage* THZStorage_(newWithData)(real *data, long size)
{
  return THZStorage_(newWithDataAndAllocator)(data, size,
                                             &THDefaultAllocator, NULL);
}

THZStorage* THZStorage_(newWithDataAndAllocator)(real* data, long size,
                                               THAllocator* allocator,
                                               void* allocatorContext) {
  THZStorage *storage = THAlloc(sizeof(THZStorage));
  storage->data = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = THZ_STORAGE_REFCOUNTED | THZ_STORAGE_RESIZABLE | THZ_STORAGE_FREEMEM;
  storage->allocator = allocator;
  storage->allocatorContext = allocatorContext;
  return storage;
}

void THZStorage_(resize)(THZStorage *storage, long size)
{
  if(storage->flag & THZ_STORAGE_RESIZABLE)
  {
    storage->data = storage->allocator->realloc(
        storage->allocatorContext,
        storage->data,
        sizeof(real)*size);
    storage->size = size;
  }
}

void THZStorage_(fill)(THZStorage *storage, real value)
{
  long i;
  for(i = 0; i < storage->size; i++)
    storage->data[i] = value;
}

void THZStorage_(set)(THZStorage *self, long idx, real value)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  self->data[idx] = value;
}

real THZStorage_(get)(const THZStorage *self, long idx)
{
  THArgCheck((idx >= 0) && (idx < self->size), 2, "out of bounds");
  return self->data[idx];
}

#endif

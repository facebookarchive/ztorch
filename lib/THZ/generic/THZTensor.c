/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZTensor.c"
#else

/**** access methods ****/
THZStorage *THZTensor_(storage)(const THZTensor *self)
{
  return self->storage;
}

long THZTensor_(storageOffset)(const THZTensor *self)
{
  return self->storageOffset;
}

int THZTensor_(nDimension)(const THZTensor *self)
{
  return self->nDimension;
}

long THZTensor_(size)(const THZTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->size[dim];
}

long THZTensor_(stride)(const THZTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->stride[dim];
}

THLongStorage *THZTensor_(newSizeOf)(THZTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THZTensor_(newStrideOf)(THZTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

real *THZTensor_(data)(const THZTensor *self)
{
  if(self->storage)
    return (self->storage->data+self->storageOffset);
  else
    return NULL;
}

void THZTensor_(setFlag)(THZTensor *self, const char flag)
{
  self->flag |= flag;
}

void THZTensor_(clearFlag)(THZTensor *self, const char flag)
{
  self->flag &= ~flag;
}

/**** creation methods ****/

static void THZTensor_(rawInit)(THZTensor *self);
static void THZTensor_(rawSet)(THZTensor *self, THZStorage *storage, long storageOffset, int nDimension, long *size, long *stride);
static void THZTensor_(rawResize)(THZTensor *self, int nDimension, long *size, long *stride);


/* Empty init */
THZTensor *THZTensor_(new)(void)
{
  THZTensor *self = THAlloc(sizeof(THZTensor));
  THZTensor_(rawInit)(self);
  return self;
}

/* Pointer-copy init */
THZTensor *THZTensor_(newWithTensor)(THZTensor *tensor)
{
  THZTensor *self = THAlloc(sizeof(THZTensor));
  THZTensor_(rawInit)(self);
  THZTensor_(rawSet)(self,
                    tensor->storage,
                    tensor->storageOffset,
                    tensor->nDimension,
                    tensor->size,
                    tensor->stride);
  return self;
}

/* Storage init */
THZTensor *THZTensor_(newWithStorage)(THZStorage *storage, long storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THZTensor *self = THAlloc(sizeof(THZTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THZTensor_(rawInit)(self);
  THZTensor_(rawSet)(self,
                    storage,
                    storageOffset,
                    (size ? size->size : (stride ? stride->size : 0)),
                    (size ? size->data : NULL),
                    (stride ? stride->data : NULL));

  return self;
}
THZTensor *THZTensor_(newWithStorage1d)(THZStorage *storage, long storageOffset,
                               long size0, long stride0)
{
  return THZTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THZTensor *THZTensor_(newWithStorage2d)(THZStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1)
{
  return THZTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THZTensor *THZTensor_(newWithStorage3d)(THZStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2)
{
  return THZTensor_(newWithStorage4d)(storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THZTensor *THZTensor_(newWithStorage4d)(THZStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2,
                               long size3, long stride3)
{
  long size[4] = {size0, size1, size2, size3};
  long stride[4] = {stride0, stride1, stride2, stride3};

  THZTensor *self = THAlloc(sizeof(THZTensor));
  THZTensor_(rawInit)(self);
  THZTensor_(rawSet)(self, storage, storageOffset, 4, size, stride);

  return self;
}

THZTensor *THZTensor_(newWithSize)(THLongStorage *size, THLongStorage *stride)
{
  return THZTensor_(newWithStorage)(NULL, 0, size, stride);
}

THZTensor *THZTensor_(newWithSize1d)(long size0)
{
  return THZTensor_(newWithSize4d)(size0, -1, -1, -1);
}

THZTensor *THZTensor_(newWithSize2d)(long size0, long size1)
{
  return THZTensor_(newWithSize4d)(size0, size1, -1, -1);
}

THZTensor *THZTensor_(newWithSize3d)(long size0, long size1, long size2)
{
  return THZTensor_(newWithSize4d)(size0, size1, size2, -1);
}

THZTensor *THZTensor_(newWithSize4d)(long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THZTensor *self = THAlloc(sizeof(THZTensor));
  THZTensor_(rawInit)(self);
  THZTensor_(rawResize)(self, 4, size, NULL);

  return self;
}

THZTensor *THZTensor_(newClone)(THZTensor *self)
{
  THZTensor *tensor = THZTensor_(new)();
  THZTensor_(resizeAs)(tensor, self);
  THZTensor_(copy)(tensor, self);
  return tensor;
}

THZTensor *THZTensor_(newContiguous)(THZTensor *self)
{
  if(!THZTensor_(isContiguous)(self))
    return THZTensor_(newClone)(self);
  else
  {
    THZTensor_(retain)(self);
    return self;
  }
}

THZTensor *THZTensor_(newSelect)(THZTensor *tensor, int dimension_, long sliceIndex_)
{
  THZTensor *self = THZTensor_(newWithTensor)(tensor);
  THZTensor_(select)(self, NULL, dimension_, sliceIndex_);
  return self;
}

THZTensor *THZTensor_(newNarrow)(THZTensor *tensor, int dimension_, long firstIndex_, long size_)
{
  THZTensor *self = THZTensor_(newWithTensor)(tensor);
  THZTensor_(narrow)(self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THZTensor *THZTensor_(newTranspose)(THZTensor *tensor, int dimension1_, int dimension2_)
{
  THZTensor *self = THZTensor_(newWithTensor)(tensor);
  THZTensor_(transpose)(self, NULL, dimension1_, dimension2_);
  return self;
}

THZTensor *THZTensor_(newUnfold)(THZTensor *tensor, int dimension_, long size_, long step_)
{
  THZTensor *self = THZTensor_(newWithTensor)(tensor);
  THZTensor_(unfold)(self, NULL, dimension_, size_, step_);
  return self;
}

/* Resize */
void THZTensor_(resize)(THZTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

  THZTensor_(rawResize)(self, size->size, size->data, (stride ? stride->data : NULL));
}

void THZTensor_(resizeAs)(THZTensor *self, THZTensor *src)
{
  THArgCheck(self != NULL, 1, "self is NULL pointer");
  THArgCheck(src != NULL, 2, "src is NULL pointer");
  int isSame = 0;
  int d;
  if(self->nDimension == src->nDimension)
  {
    isSame = 1;
    for(d = 0; d < self->nDimension; d++)
    {
      if(self->size[d] != src->size[d])
      {
        isSame = 0;
        break;
      }
    }
  }
  if(!isSame)
    THZTensor_(rawResize)(self, src->nDimension, src->size, NULL);
}

void THZTensor_(resize1d)(THZTensor *tensor, long size0)
{
  THZTensor_(resize4d)(tensor, size0, -1, -1, -1);
}

void THZTensor_(resize2d)(THZTensor *tensor, long size0, long size1)
{
  THZTensor_(resize4d)(tensor, size0, size1, -1, -1);
}

void THZTensor_(resize3d)(THZTensor *tensor, long size0, long size1, long size2)
{
  THZTensor_(resize4d)(tensor, size0, size1, size2, -1);
}

void THZTensor_(resize4d)(THZTensor *self, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THZTensor_(rawResize)(self, 4, size, NULL);
}

void THZTensor_(resize5d)(THZTensor *self, long size0, long size1, long size2, long size3, long size4)
{
    long size[5] = {size0, size1, size2, size3, size4};

  THZTensor_(rawResize)(self, 5, size, NULL);
}

void THZTensor_(set)(THZTensor *self, THZTensor *src)
{
  if(self != src)
    THZTensor_(rawSet)(self,
                      src->storage,
                      src->storageOffset,
                      src->nDimension,
                      src->size,
                      src->stride);
}

void THZTensor_(setStorage)(THZTensor *self, THZStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

  THZTensor_(rawSet)(self,
                    storage_,
                    storageOffset_,
                    (size_ ? size_->size : (stride_ ? stride_->size : 0)),
                    (size_ ? size_->data : NULL),
                    (stride_ ? stride_->data : NULL));
}

void THZTensor_(setStorage1d)(THZTensor *self, THZStorage *storage_, long storageOffset_,
                             long size0_, long stride0_)
{
  THZTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          -1, -1,
                          -1, -1,
                          -1, -1);
}

void THZTensor_(setStorage2d)(THZTensor *self, THZStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_)
{
  THZTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          size1_, stride1_,
                          -1, -1,
                          -1, -1);
}

void THZTensor_(setStorage3d)(THZTensor *self, THZStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_)
{
  THZTensor_(setStorage4d)(self, storage_, storageOffset_,
                          size0_, stride0_,
                          size1_, stride1_,
                          size2_, stride2_,
                          -1, -1);
}

void THZTensor_(setStorage4d)(THZTensor *self, THZStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_,
                             long size3_, long stride3_)
{

  long size[4] = {size0_, size1_, size2_, size3_};
  long stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THZTensor_(rawSet)(self, storage_, storageOffset_, 4, size, stride);
}


void THZTensor_(narrow)(THZTensor *self, THZTensor *src, int dimension, long firstIndex, long size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 4, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= src->size[dimension]), 5, "out of range");

  THZTensor_(set)(self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THZTensor_(select)(THZTensor *self, THZTensor *src, int dimension, long sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 4, "out of range");

  THZTensor_(set)(self, src);
  THZTensor_(narrow)(self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->nDimension-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;
}

void THZTensor_(transpose)(THZTensor *self, THZTensor *src, int dimension1, int dimension2)
{
  long z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 2, "out of range");

  THZTensor_(set)(self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride[dimension1];
  self->stride[dimension1] = self->stride[dimension2];
  self->stride[dimension2] = z;
  z = self->size[dimension1];
  self->size[dimension1] = self->size[dimension2];
  self->size[dimension2] = z;
}

void THZTensor_(unfold)(THZTensor *self, THZTensor *src, int dimension, long size, long step)
{
  long *newSize;
  long *newStride;
  int d;

  if(!src)
    src = self;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THZTensor_(set)(self, src);

  newSize = THAlloc(sizeof(long)*(self->nDimension+1));
  newStride = THAlloc(sizeof(long)*(self->nDimension+1));

  newSize[self->nDimension] = size;
  newStride[self->nDimension] = self->stride[dimension];
  for(d = 0; d < self->nDimension; d++)
  {
    if(d == dimension)
    {
      newSize[d] = (self->size[d] - size) / step + 1;
      newStride[d] = step*self->stride[d];
    }
    else
    {
      newSize[d] = self->size[d];
      newStride[d] = self->stride[d];
    }
  }

  THFree(self->size);
  THFree(self->stride);

  self->size = newSize;
  self->stride = newStride;
  self->nDimension++;
}

/* we have to handle the case where the result is a number */
void THZTensor_(squeeze)(THZTensor *self, THZTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THZTensor_(set)(self, src);

  for(d = 0; d < src->nDimension; d++)
  {
    if(src->size[d] != 1)
    {
      if(d != ndim)
      {
        self->size[ndim] = src->size[d];
        self->stride[ndim] = src->stride[d];
      }
      ndim++;
    }
  }

  /* right now, we do not handle 0-dimension tensors */
  if(ndim == 0 && src->nDimension > 0)
  {
    self->size[0] = 1;
    self->stride[0] = 1;
    ndim = 1;
  }
  self->nDimension = ndim;
}

void THZTensor_(squeeze1d)(THZTensor *self, THZTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < src->nDimension, 3, "dimension out of range");

  THZTensor_(set)(self, src);

  if(src->size[dimension] == 1 && src->nDimension > 1)
  {
    for(d = dimension; d < self->nDimension-1; d++)
    {
      self->size[d] = self->size[d+1];
      self->stride[d] = self->stride[d+1];
    }
    self->nDimension--;
  }
}

int THZTensor_(isContiguous)(const THZTensor *self)
{
  long z = 1;
  int d;
  for(d = self->nDimension-1; d >= 0; d--)
  {
    if(self->size[d] != 1)
    {
      if(self->stride[d] == z)
        z *= self->size[d];
      else
        return 0;
    }
  }
  return 1;
}

int THZTensor_(isSameSizeAs)(const THZTensor *self, const THZTensor* src)
{
  int d;
  if (self->nDimension != src->nDimension)
    return 0;
  for(d = 0; d < self->nDimension; ++d)
  {
    if(self->size[d] != src->size[d])
      return 0;
  }
  return 1;
}

long THZTensor_(nElement)(const THZTensor *self)
{
  if(self->nDimension == 0)
    return 0;
  else
  {
    long nElement = 1;
    int d;
    for(d = 0; d < self->nDimension; d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THZTensor_(retain)(THZTensor *self)
{
  if(self->flag & THZ_TENSOR_REFCOUNTED)
    THAtomicIncrementRef(&self->refcount);
}

void THZTensor_(free)(THZTensor *self)
{
  if(!self)
    return;

  if(self->flag & THZ_TENSOR_REFCOUNTED)
  {
    if(THAtomicDecrementRef(&self->refcount))
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THZStorage_(free)(self->storage);
      THFree(self);
    }
  }
}

void THZTensor_(freeCopyTo)(THZTensor *self, THZTensor *dst)
{
  if(self != dst)
    THZTensor_(copy)(dst, self);

  THZTensor_(free)(self);
}

/*******************************************************************************/

static void THZTensor_(rawInit)(THZTensor *self)
{
  self->refcount = 1;
  self->storage = NULL;
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = THZ_TENSOR_REFCOUNTED;
}

static void THZTensor_(rawSet)(THZTensor *self, THZStorage *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THZStorage_(free)(self->storage);

    if(storage)
    {
      self->storage = storage;
      THZStorage_(retain)(self->storage);
    }
    else
      self->storage = NULL;
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THZTensor_(rawResize)(self, nDimension, size, stride);
}

static void THZTensor_(rawResize)(THZTensor *self, int nDimension, long *size, long *stride)
{
  int d;
  int nDimension_;
  long totalSize;
  int hascorrectsize = 1;

  nDimension_ = 0;
  for(d = 0; d < nDimension; d++)
  {
    if(size[d] > 0)
    {
      nDimension_++;
      if((self->nDimension > d) && (size[d] != self->size[d]))
        hascorrectsize = 0;

      if((self->nDimension > d) && stride && (stride[d] >= 0) && (stride[d] != self->stride[d]))
        hascorrectsize = 0;
    }
    else
      break;
  }
  nDimension = nDimension_;

  if(nDimension != self->nDimension)
    hascorrectsize = 0;

  if(hascorrectsize)
    return;

  if(nDimension > 0)
  {
    if(nDimension != self->nDimension)
    {
      self->size = THRealloc(self->size, sizeof(long)*nDimension);
      self->stride = THRealloc(self->stride, sizeof(long)*nDimension);
      self->nDimension = nDimension;
    }

    totalSize = 1;
    for(d = self->nDimension-1; d >= 0; d--)
    {
      self->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        self->stride[d] = stride[d];
      else
      {
        if(d == self->nDimension-1)
          self->stride[d] = 1;
        else
          self->stride[d] = self->size[d+1]*self->stride[d+1];
      }
      totalSize += (self->size[d]-1)*self->stride[d];
    }

    if(totalSize+self->storageOffset > 0)
    {
      if(!self->storage)
        self->storage = THZStorage_(new)();
      if(totalSize+self->storageOffset > self->storage->size)
        THZStorage_(resize)(self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

void THZTensor_(set1d)(THZTensor *tensor, long x0, real value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THZStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

real THZTensor_(get1d)(const THZTensor *tensor, long x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THZStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THZTensor_(set2d)(THZTensor *tensor, long x0, long x1, real value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THZStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

real THZTensor_(get2d)(const THZTensor *tensor, long x0, long x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THZStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THZTensor_(set3d)(THZTensor *tensor, long x0, long x1, long x2, real value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THZStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

real THZTensor_(get3d)(const THZTensor *tensor, long x0, long x1, long x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THZStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THZTensor_(set4d)(THZTensor *tensor, long x0, long x1, long x2, long x3, real value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THZStorage_(set)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

real THZTensor_(get4d)(const THZTensor *tensor, long x0, long x1, long x2, long x3)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  return THZStorage_(get)(tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]);
}

#endif

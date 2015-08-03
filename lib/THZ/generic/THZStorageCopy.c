/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZStorageCopy.c"
#else

void THZStorage_(rawCopy)(THZStorage *storage, real *src)
{
  long i;
  for(i = 0; i < storage->size; i++)
    storage->data[i] = src[i];
}

#define IMPLEMENT_THZStorage_COPY(TYPENAMESRC) \
void THZStorage_(copy##TYPENAMESRC)(THZStorage *storage, TH##TYPENAMESRC##Storage *src) \
{ \
  long i; \
  THArgCheck(storage->size == src->size, 2, "size mismatch");   \
  for(i = 0; i < storage->size; i++)                            \
    storage->data[i] = (real)src->data[i];                      \
}

IMPLEMENT_THZStorage_COPY(ZFloat)
IMPLEMENT_THZStorage_COPY(ZDouble)
IMPLEMENT_THZStorage_COPY(Byte)
IMPLEMENT_THZStorage_COPY(Char)
IMPLEMENT_THZStorage_COPY(Short)
IMPLEMENT_THZStorage_COPY(Int)
IMPLEMENT_THZStorage_COPY(Long)
IMPLEMENT_THZStorage_COPY(Float)
IMPLEMENT_THZStorage_COPY(Double)

#endif

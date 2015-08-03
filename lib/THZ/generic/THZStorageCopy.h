/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant 
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#ifndef THZ_GENERIC_FILE
#define THZ_GENERIC_FILE "generic/THZStorageCopy.h"
#else

/* Support for copy between different Storage types */

THZ_API void THZStorage_(rawCopy)(THZStorage *storage, real *src);
THZ_API void THZStorage_(copyZFloat)(THZStorage *storage, THZFloatStorage *src);
THZ_API void THZStorage_(copyZDouble)(THZStorage *storage, THZDoubleStorage *src);
THZ_API void THZStorage_(copyByte)(THZStorage *storage, struct THByteStorage *src);
THZ_API void THZStorage_(copyChar)(THZStorage *storage, struct THCharStorage *src);
THZ_API void THZStorage_(copyShort)(THZStorage *storage, struct THShortStorage *src);
THZ_API void THZStorage_(copyInt)(THZStorage *storage, struct THIntStorage *src);
THZ_API void THZStorage_(copyLong)(THZStorage *storage, struct THLongStorage *src);
THZ_API void THZStorage_(copyFloat)(THZStorage *storage, struct THFloatStorage *src);
THZ_API void THZStorage_(copyDouble)(THZStorage *storage, struct THDoubleStorage *src);

#endif

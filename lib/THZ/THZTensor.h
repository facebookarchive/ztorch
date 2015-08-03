#ifndef THZ_TENSOR_INC
#define THZ_TENSOR_INC

#include "THZStorage.h"

#define THZTensor          TH_CONCAT_3(THZ,Real,Tensor)
#define THZTensor_(NAME)   TH_CONCAT_4(THZ,Real,Tensor_,NAME)

/* basics */
#include "generic/THZTensor.h"
#include "THZGenerateAllTypes.h"

#include "generic/THZTensorCopy.h"
#include "THZGenerateAllTypes.h"

/* maths */
#include "generic/THZTensorMath.h"
#include "THZGenerateAllTypes.h"

/* convolutions */
#include "generic/THZTensorConv.h"
#include "THZGenerateAllTypes.h"

/* lapack support */
#include "generic/THZTensorLapack.h"
#include "THZGenerateAllTypes.h"

#endif

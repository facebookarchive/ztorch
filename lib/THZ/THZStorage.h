#ifndef THZ_STORAGE_INC
#define THZ_STORAGE_INC

#include "THZGeneral.h"
#include "THAllocator.h"

#define THZStorage        TH_CONCAT_3(THZ,Real,Storage)
#define THZStorage_(NAME) TH_CONCAT_4(THZ,Real,Storage_,NAME)

#include "generic/THZStorage.h"
#include "THZGenerateAllTypes.h"

#include "generic/THZStorageCopy.h"
#include "THZGenerateAllTypes.h"

#endif

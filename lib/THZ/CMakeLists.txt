cmake_minimum_required(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)
SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# OpenMP support?
SET(WITH_OPENMP ON CACHE BOOL "OpenMP support if available?")
IF (APPLE AND CMAKE_COMPILER_IS_GNUCC)
  EXEC_PROGRAM (uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
  STRING (REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  MESSAGE (STATUS "MAC OS Darwin Version: ${DARWIN_VERSION}")
  IF (DARWIN_VERSION GREATER 9)
    SET(APPLE_OPENMP_SUCKS 1)
  ENDIF (DARWIN_VERSION GREATER 9)
  EXECUTE_PROCESS (COMMAND ${CMAKE_C_COMPILER} -dumpversion
    OUTPUT_VARIABLE GCC_VERSION)
  IF (APPLE_OPENMP_SUCKS AND GCC_VERSION VERSION_LESS 4.6.2)
    MESSAGE(STATUS "Warning: Disabling OpenMP (unstable with this version of GCC)")
    MESSAGE(STATUS " Install GCC >= 4.6.2 or change your OS to enable OpenMP")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
    SET(WITH_OPENMP OFF CACHE BOOL "OpenMP support if available?" FORCE)
  ENDIF ()
ENDIF ()

IF (WITH_OPENMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    MESSAGE(STATUS "Compiling with OpenMP support")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  ENDIF(OPENMP_FOUND)
ENDIF (WITH_OPENMP)

IF(UNIX)
  INCLUDE(CheckFunctionExists)
  SET(CMAKE_EXTRA_INCLUDE_FILES "sys/mman.h")
  CHECK_FUNCTION_EXISTS(mmap HAVE_MMAP)
  IF(HAVE_MMAP)
    ADD_DEFINITIONS(-DHAVE_MMAP=1)
  ENDIF(HAVE_MMAP)
ENDIF(UNIX)

SET(hdr
  THZGeneral.h THZStorage.h THZTensor.h THZBlas.h
  THZLapack.h THZVector.h)

SET(src
  THZGeneral.c THZStorage.c THZTensor.c THZBlas.c THZLapack.c)

SET(src ${src} ${hdr})
ADD_LIBRARY(THZ SHARED ${src})

TARGET_LINK_LIBRARIES(THZ TH)

FIND_PACKAGE(BLAS)
IF(BLAS_FOUND)
  SET(USE_BLAS 1)
  TARGET_LINK_LIBRARIES(THZ ${BLAS_LIBRARIES})
ENDIF(BLAS_FOUND)

FIND_PACKAGE(LAPACK)
IF(LAPACK_FOUND)
  SET(USE_LAPACK 1)
  TARGET_LINK_LIBRARIES(THZ ${LAPACK_LIBRARIES})
ENDIF(LAPACK_FOUND)

SET(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})
FOREACH(KEYWORD "inline" "__inline__" "__inline")
  IF(NOT DEFINED C_INLINE)

    SET(CMAKE_REQUIRED_FLAGS "-Dinline=${KEYWORD} ${CMAKE_C_FLAGS}")
    CHECK_C_SOURCE_RUNS("
       static inline int static_foo()
       {
         return 0;
       }

       int main(int argc, char *argv[])
       {
         static_foo();
         return 0;
       }" C_HAS_${KEYWORD})

    IF(C_HAS_${KEYWORD})
      SET(C_INLINE TRUE)
      SET(THZ_INLINE ${KEYWORD})
      MESSAGE(STATUS "C inline is supported (${KEYWORD})")
    ENDIF(C_HAS_${KEYWORD})
  ENDIF(NOT DEFINED C_INLINE)
ENDFOREACH(KEYWORD)
SET(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

IF(NOT DEFINED C_INLINE)
  MESSAGE(STATUS "C inline seems not supported")
  SET(THZ_INLINE "")
ENDIF(NOT DEFINED C_INLINE)

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_BINARY_DIR}")
CONFIGURE_FILE(THZGeneral.h.in "${CMAKE_CURRENT_BINARY_DIR}/THZGeneral.h")

INSTALL(TARGETS THZ
		LIBRARY DESTINATION "${Torch_DIR}/../../../lib")

INSTALL(FILES
  THZ.h
  ${CMAKE_CURRENT_BINARY_DIR}/THZGeneral.h
  THZBlas.h
  THZGenerateAllTypes.h
  THZLapack.h
  THZStorage.h
  THZTensor.h
  THZVector.h
  DESTINATION "${Torch_DIR}/../../../include/TH"
)

INSTALL(FILES
  generic/THZBlas.c
  generic/THZBlas.h
  generic/THZLapack.c
  generic/THZLapack.h
  generic/THZStorage.c
  generic/THZStorage.h
  generic/THZStorageCopy.c
  generic/THZStorageCopy.h
  generic/THZTensor.c
  generic/THZTensor.h
  generic/THZTensorConv.c
  generic/THZTensorConv.h
  generic/THZTensorCopy.c
  generic/THZTensorCopy.h
  generic/THZTensorLapack.c
  generic/THZTensorLapack.h
  generic/THZTensorMath.c
  generic/THZTensorMath.h
  generic/THZVector.c
  DESTINATION "${Torch_DIR}/../../../include/TH/generic"
)

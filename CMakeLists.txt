cmake_minimum_required(VERSION 3.2)
project(KronMat)
add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CUDA)

# Additional 'FindX.cmake'
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake/Modules)

# Google test framework
include("test/gtest.cmake")

find_package(OpenBLAS)
find_package(CUDA)
find_package(TBB)
find_package(Boost)
message("CUDA version ${CUDA_VERSION}")
set(CUSOLVER_LIBRARY, "")
if(${CUDA_VERSION} LESS 7) 
    message("CUDA version < 7 - using homebrew Cholesky")
else()
    message("CUDA version >= 7 - using CUSolver Cholesky")
    add_definitions("-DCUSOLVER")
    set(CUDA_LIBRARIES "${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY}")
endif()

# Compiler flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3") 

# Includes
include_directories(${CUDA_INCLUDE_DIRS} ${OpenBLAS_INCLUDE_DIR} ${GTEST_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} src inc /Users/tom/Code/KronMat/inc)

## Source files for executable
set(SOURCE_FILES src/play.cu)
CUDA_ADD_EXECUTABLE(Blitz ${SOURCE_FILES})
CUDA_ADD_CUBLAS_TO_TARGET(Blitz)

target_link_libraries(Blitz ${OpenBLAS_LIB} ${TBB_LIBRARY} ${GTEST_BOTH_LIBRARIES}  ${CUDA_LIBRARIES})
# Tests
enable_testing()

if(CUDA_FOUND)
    CUDA_ADD_EXECUTABLE(TestHostSqDist test/TestHostSqDist.cu)
    CUDA_ADD_CUBLAS_TO_TARGET(TestHostSqDist)
    target_link_libraries(TestHostSqDist ${OpenBLAS_LIB}  ${TBB_LIBRARY} ${GTEST_LIBRARY} ${CUDA_LIBRARIES} )
    add_test(TestHostSqDist TestHostSqDist)
endif()
if(CUDA_FOUND)
    CUDA_ADD_EXECUTABLE(TestCUDASqDist test/TestCUDASqDist.cu)
    CUDA_ADD_CUBLAS_TO_TARGET(TestCUDASqDist)
    target_link_libraries(TestCUDASqDist ${OpenBLAS_LIB}  ${TBB_LIBRARY} ${GTEST_LIBRARY} ${CUDA_LIBRARIES} )
    add_test(TestCUDASqDist TestCUDASqDist)
endif()
if(CUDA_FOUND)
    CUDA_ADD_EXECUTABLE(TestTBBSqDist test/TestTBBSqDist.cu)
    CUDA_ADD_CUBLAS_TO_TARGET(TestTBBSqDist)
    target_link_libraries(TestTBBSqDist ${OpenBLAS_LIB}  ${TBB_LIBRARY} ${GTEST_LIBRARY} ${CUDA_LIBRARIES} )
    add_test(TestTBBSqDist TestTBBSqDist)
endif()

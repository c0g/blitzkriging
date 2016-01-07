//
// Created by Thomas Nickson on 05/07/2015.
//

#include "kronlib.h"
#include "blitzkriging.h"
#include <iostream>
#include "gtest/gtest.h"
using namespace kronlib;
using namespace blitzkriging;
TEST(SqDist, MM)
{
   CUDAMatrix<float> m0{3,1}, m1{2,1}, m2{3,1};
   m0 = 1,2,3;
   m1 = 1,2;
   m2 = 1,2,3;
   std::vector<CUDAMatrix<float>> m{{ m0, m1, m2 }};
   
   Dummy<CUDAMatrix<float>> k{3};
   k.setM(m);
   auto distMM = k.sqdistMM();
   CUDAMatrix<float> sd2{2,2}, sd3{3,3};
   sd2 = 0, 1,
         1, 0;
   sd3 = 0, 1, 4,
         1, 0, 1,
         4, 1, 0;
   Kronecker<CUDAMatrix<float>> ans{{ sd3, sd2, sd3 }};
   ASSERT_EQ(ans, distMM);
}
TEST(SqDist, MX) 
{
   CUDAMatrix<float> m0{3,1}, m1{2,1}, m2{3,1}, x0{2,1}, x1{2,1}, x2{2,1};
   m0 = 1,2,3;
   m1 = 1,2;
   m2 = 1,2,3;
   x0 = 1,2;
   x1 = 2,3;
   x2 = 3,4;
   std::vector<CUDAMatrix<float>> m{{ m0, m1, m2 }};
   std::vector<CUDAMatrix<float>> x{{ x0, x1, x2 }};
   
   Dummy<CUDAMatrix<float>> k{3};
   k.setM(m);
   k.setX(x);
   auto distMX = k.sqdistMX();
   CUDAMatrix<float> sd0{3,2}, sd1{2,2}, sd2{3,2};
   sd0 = 0, 1,
         1, 0,
         4, 1;
   sd1 = 1, 0,
         4, 1;
   sd2 = 4, 9,
         1, 4,
         0, 1;
   KroneckerVectorStack<CUDAMatrix<float>> ans{{ sd0, sd1, sd2 }};
   ASSERT_EQ(ans, distMX);
}
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

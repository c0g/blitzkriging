#ifndef kernel_h
#define kernel_h
#include "kronlib.h"

namespace blitzkriging {
using namespace kronlib;
template <typename MatrixType>
class Kernel {
protected:
    using Storage = typename MatrixType::Storage;
    using T = typename Storage::value_type;


public:
    Kernel(int D) : D{D} {}
    void setX(const std::vector<MatrixType> & newX) { checkD(newX); X = newX; }
    void setM(const std::vector<MatrixType> & newM) { checkD(newM); M = newM; }
    void setP(const std::vector<MatrixType> & newP) { checkD(newP);  P = newP; }
    void setHyp(const Storage & newHyp) { checkHyp(newHyp); rawHyp = newHyp; }

    const Kronecker<Cholesky<MatrixType>> & Lmm() const { return mLmm; }
    const Kronecker<MatrixType> & Kmm() const { return mKmm; }
    const std::vector<Kronecker<MatrixType>> & dKmm() const { return mdKmm; }
    const KroneckerVectorStack<MatrixType> & Kmx() const { return mKmx; }
    const std::vector<KroneckerVectorStack<MatrixType>> & dKmx() const { return mdKmx; }
    const KroneckerVectorStack<MatrixType> & Kmp() const { return mKmp; }
    int getD() { return D; }

    virtual void updateInferenceMatrices() = 0;
    virtual void updateInferenceMatricesGradients() = 0;
    virtual void updatePredictionMatrices() = 0;
    virtual void updatePredictionMatricesGradients() = 0;
    virtual Storage suggestHyp() = 0;

protected:
    int D;
    // Parameters of the kernel
    std::vector<MatrixType> M;
    std::vector<MatrixType> X;
    std::vector<MatrixType> P;
    Storage rawHyp;

    // Matrices for use in the likelihoods
    Kronecker<Cholesky<MatrixType>> mLmm;
    Kronecker<MatrixType> mKmm;
    std::vector<Kronecker<MatrixType>> mdKmm;
    KroneckerVectorStack<MatrixType> mKmx;
    std::vector<KroneckerVectorStack<MatrixType>> mdKmx;
    KroneckerVectorStack<MatrixType> mKmp;

    // Internal matrices to reduce uneeded calculations
    Kronecker<MatrixType> mmsqdist;
    KroneckerVectorStack<MatrixType> mxsqdist;

    void checkD( const std::vector<MatrixType> & S ) const  
    {
        if (S.size() != D) 
        {
            std::cout << "Kernel expects " << D << " dimensions, you gave " << S.size() << " around " << __LINE__ << " in " << __FILE__ << std::endl;
            std::cout << "Program will now exit \n";
            exit(0);
        }
    }
    void checkD( const MatrixType & S ) const  
    {
        if (S.nC() != D) 
        {
            std::cout << "Kernel expects " << D << " dimensions, you gave " << S.nC() << " around " << __LINE__ << " in " << __FILE__ << std::endl;
            std::cout << "Program will now exit \n";
            exit(0);
        }
    }

    void sqdist(const std::vector<MatrixType> & M) // Calculates square distance for Kmm
    {
        mmsqdist = Kronecker<MatrixType>{};
        for (const auto & dimM : M)
        {
            // Pairwise distance between matrices is: sumsq_cols(x1) + sumsq_cols(x2)' - 2 x1 x2', using Python style broadcasting
            MatrixType sumsq_cols = dimM.sumsq_cols(); 
            MatrixType mat = dimM.dot(None, dimM, Trans);
            mat *= -2;
            mat.col_wise_tiled_add_inplace(sumsq_cols);
            mat.row_wise_tiled_add_inplace(sumsq_cols);
            mmsqdist.push_back(MatrixType{mat});
        }
    }
    void sqdist(const std::vector<MatrixType> & M, const std::vector<MatrixType> & X)   
    {
        mxsqdist = KroneckerVectorStack<MatrixType>{};
        auto mit = M.begin();
        auto xit = X.begin();
        for ( ; (mit < M.end()) && (xit < X.end()); ++mit, ++xit)
        {
            // Pairwise distance between matrices is: sumsq_cols(x1) + sumsq_cols(x2)' - 2 x1 x2', using Python style broadcasting
            MatrixType sumsq_cols_M = mit->sumsq_cols(); 
            MatrixType sumsq_cols_X = xit->sumsq_cols(); 
            MatrixType mat = mit->dot(None, *xit, Trans);
            mat *= -2;
            mat.col_wise_tiled_add_inplace(sumsq_cols_M);
            mat.row_wise_tiled_add_inplace(sumsq_cols_X);
            mxsqdist.push_back(mat);
        }
    }
};
} // namespace blitzkriging
#endif // kernel_h include guard

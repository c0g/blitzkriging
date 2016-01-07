#ifndef dummy_h
#define dummy_h
#include "kernel/kernel.h"
namespace blitzkriging{
template <typename MatrixType>
class Dummy : public Kernel<MatrixType> {

    public:
    Dummy(int D) : Kernel<MatrixType>{D} {}
    virtual void updateInferenceMatrices() override {}
    virtual void updateInferenceMatricesGradients() override {}
    virtual void updatePredictionMatrices() override {}
    virtual void updatePredictionMatricesGradients() override {}
    virtual typename Kernel<MatrixType>::Storage suggestHyp() override {
        return this->rawHyp;
    }
    
    const Kronecker<MatrixType> & sqdistMM() {this->sqdist(this->M); return this->mmsqdist; }
    const KroneckerVectorStack<MatrixType> & sqdistMX() { this->sqdist(this->M, this->X); return this->mxsqdist; }
};
}//namespace blitz
#endif // dummy include guard

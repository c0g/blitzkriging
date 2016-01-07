#ifndef sqexp_h
#define sqexp_h
#include "kernel/kernel.h"
namespace blitzkriging{
template <typename MatrixType>
class SqExp : public Kernel<MatrixType> {

    public:
    void updateInferenceMatrices() override {}
    void updateInferenceMatricesGradients() override {}
    void updatePredictionMatrices() override {}
    void updatePredictionMatricesGradients() override {}
    typename Kernel<MatrixType>::Storage suggestHyp() override {
        return this->rawHyp;
    }
};
}//namespace blitz
#endif // sqexp include guard

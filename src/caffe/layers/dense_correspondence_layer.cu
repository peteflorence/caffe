#include "caffe/layers/dense_correspondence_layer_impl.hpp"

namespace caffe {

template <typename Dtype,
          template <typename> class LossFunctorT,
          template <typename> class PixelwiseWeightingT>
__global__ void DCLBackwardPositives(const int N,
                                     const int width,
                                     const int height,
                                     const int channels,
                                     const Dtype * posDiffData,
                                     const Dtype * posSampleAData,
                                     const Dtype * posSampleBData,
                                     const Dtype posAlpha,
                                     const LossFunctorT<Dtype> lossFunctor,
                                     PixelwiseWeightingT<Dtype> weighting,
                                     Dtype * gradA,
                                     Dtype * gradB) {

    CUDA_KERNEL_LOOP(i, N) {

        const int uA = floor(posSampleAData[0 + 2*i] + 0.5);
        const int vA = floor(posSampleAData[1 + 2*i] + 0.5);
        const Dtype uB = posSampleBData[0 + 2*i];
        const Dtype vB = posSampleBData[1 + 2*i];

        const Dtype * thisDiff = posDiffData + i*channels;

        const Dtype weightA = weighting.positiveWeightA(uA,vA);

        const Dtype weightB = weighting.positiveWeightB(uB,vB);

        const Dtype thisAlpha = weightA*weightB*posAlpha;

//        printf("%f*%f*%f (%d,%d -- %f,%f) [%f]\n",weightA,weightB,posAlpha,uA,vA,uB,vB,lossFunctor.loss(thisDiff,channels));

        lossFunctor.template differentiateLoss<CudaAtomicAddition>(thisDiff,width,height,channels,
                                                                   uA,vA, thisAlpha,gradA);

        lossFunctor.template differentiateLoss<CudaAtomicAddition>(thisDiff,width,height,channels,
                                                                   uB,vB,-thisAlpha,gradB);

//        weighting.template backpropWeightA<CudaAtomicAddition>(uA,vA,posAlpha*weightB*lossFunctor.loss(thisDiff,channels));

//        weighting.template backpropWeightB<CudaAtomicAddition>(uB,vB,posAlpha*weightA*lossFunctor.loss(thisDiff,channels));

        weighting.template backpropPositiveWeightA<CudaAtomicAddition>(uA,vA,posAlpha*(weightB*lossFunctor.loss(thisDiff,channels) + 0.2*(Dtype(2)*weightA - Dtype(2))));

        weighting.template backpropPositiveWeightB<CudaAtomicAddition>(uB,vB,posAlpha*(weightA*lossFunctor.loss(thisDiff,channels) + 0.2*(Dtype(2)*weightB - Dtype(2))));

//        weighting.template backpropWeightA<CudaAtomicAddition>(uA,vA,posAlpha*weightB*0.5*(1/sqrt(weightA*weightB))*lossFunctor.loss(thisDiff,channels));

//        weighting.template backpropWeightB<CudaAtomicAddition>(uB,vB,posAlpha*weightA*0.5*(1/sqrt(weightA*weightB))*lossFunctor.loss(thisDiff,channels));

    }

}

//template __global__ void DCLBackwardPositives<float, SquaredLossFunctor>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const SquaredLossFunctor<float>,float *, float *);
//template __global__ void DCLBackwardPositives<float, HuberLossFunctor>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const HuberLossFunctor<float>,float *, float *);


template <typename Dtype,
          template <typename> class LossFunctorT,
          template <typename> class PixelwiseWeightingT>
__global__ void DCLBackwardNegatives(const int N,
                                     const int width,
                                     const int height,
                                     const int channels,
                                     const Dtype * negDiffData,
                                     const Dtype * negSampleAData,
                                     const Dtype * negSampleBData,
                                     const Dtype negAlpha,
                                     const LossFunctorT<Dtype> lossFunctor,
                                     PixelwiseWeightingT<Dtype> weighting,
                                     Dtype * gradA,
                                     Dtype * gradB) {

    CUDA_KERNEL_LOOP(i, N) {

        const int uA = floor(negSampleAData[0 + 2*i] + 0.5);
        const int vA = floor(negSampleAData[1 + 2*i] + 0.5);
        const int uB = floor(negSampleBData[0 + 2*i] + 0.5);
        const int vB = floor(negSampleBData[1 + 2*i] + 0.5);

        const Dtype * thisDiff = negDiffData + i*channels;

        const Dtype weightA = weighting.negativeWeightA(uA,vA);

        const Dtype weightB = weighting.negativeWeightB(uB,vB);

        const Dtype thisAlpha = weightA*weightB*negAlpha;

        lossFunctor.template differentiateLoss<CudaAtomicAddition>(thisDiff,width,height,channels,
                                                                   uA,vA,thisAlpha,gradA);

        lossFunctor.template differentiateLoss<CudaAtomicAddition>(thisDiff,width,height,channels,
                                                                   uB,vB,-thisAlpha,gradB);

        weighting.template backpropNegativeWeightA<CudaAtomicAddition>(uA,vA,negAlpha*(weightB*lossFunctor.loss(thisDiff,channels) + 0.2*(Dtype(2)*weightA - Dtype(2))));

        weighting.template backpropNegativeWeightB<CudaAtomicAddition>(uB,vB,negAlpha*(weightA*lossFunctor.loss(thisDiff,channels) + 0.2*(Dtype(2)*weightB - Dtype(2))));

    }

}

//template __global__ void DCLBackwardNegatives<float, HingeLossFunctor>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const HingeLossFunctor<float>,float *, float *);
//template __global__ void DCLBackwardNegatives<float, HuberHingeLossFunctor>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const HuberHingeLossFunctor<float>,float *, float *);
//template __global__ void DCLBackwardNegatives<float, NegativeExponentialLossFunctor>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const NegativeExponentialLossFunctor<float>,float *, float *);

template <typename Dtype,
          template <typename> class LossFunctorT,
          template <typename> class PixelwiseWeightingT>
void backwardPositiveWrapper(const int N,
                             const int width,
                             const int height,
                             const int channels,
                             const Dtype * posDiffData,
                             const Dtype * posSampleAData,
                             const Dtype * posSampleBData,
                             const Dtype posAlpha,
                             const LossFunctorT<Dtype> lossFunctor,
                             Dtype * gradA,
                             Dtype * gradB,
                             PixelwiseWeightingT<Dtype> weighting) {

    if (N == 0) { return; }

    DCLBackwardPositives<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, width, height, channels, posDiffData, posSampleAData, posSampleBData,
        posAlpha, lossFunctor, weighting, gradA, gradB);
    CUDA_POST_KERNEL_CHECK;

}

template void backwardPositiveWrapper<float, SquaredLossFunctor,NoWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const SquaredLossFunctor<float>,float *, float *, NoWeighting<float>);
//template void backwardPositiveWrapper<float, HuberLossFunctor,NoWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const HuberLossFunctor<float>,float *, float *, NoWeighting<float>);
//template void backwardPositiveWrapper<float, TukeyLossFunctor,NoWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const TukeyLossFunctor<float>,float *, float *, NoWeighting<float>);

template void backwardPositiveWrapper<double, SquaredLossFunctor,NoWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const SquaredLossFunctor<double>,double *, double *, NoWeighting<double>);
//template void backwardPositiveWrapper<double, HuberLossFunctor,NoWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const HuberLossFunctor<double>,double *, double *, NoWeighting<double>);
//template void backwardPositiveWrapper<double, TukeyLossFunctor,NoWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const TukeyLossFunctor<double>,double *, double *, NoWeighting<double>);


template void backwardPositiveWrapper<float, SquaredLossFunctor,InputWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const SquaredLossFunctor<float>,float *, float *, InputWeighting<float>);
//template void backwardPositiveWrapper<float, HuberLossFunctor,InputWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const HuberLossFunctor<float>,float *, float *, InputWeighting<float>);
//template void backwardPositiveWrapper<float, TukeyLossFunctor,InputWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const TukeyLossFunctor<float>,float *, float *, InputWeighting<float>);

template void backwardPositiveWrapper<double, SquaredLossFunctor,InputWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const SquaredLossFunctor<double>,double *, double *, InputWeighting<double>);
//template void backwardPositiveWrapper<double, HuberLossFunctor,InputWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const HuberLossFunctor<double>,double *, double *, InputWeighting<double>);
//template void backwardPositiveWrapper<double, TukeyLossFunctor,InputWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const TukeyLossFunctor<double>,double *, double *, InputWeighting<double>);


template <typename Dtype,
          template <typename> class LossFunctorT,
          template <typename> class PixelwiseWeightingT>
void backwardNegativeWrapper(const int N,
                             const int width,
                             const int height,
                             const int channels,
                             const Dtype * negDiffData,
                             const Dtype * negSampleAData,
                             const Dtype * negSampleBData,
                             const Dtype negAlpha,
                             const LossFunctorT<Dtype> lossFunctor,
                             Dtype * gradA,
                             Dtype * gradB,
                             PixelwiseWeightingT<Dtype> weighting) {

    if (N == 0) { return; }

    DCLBackwardNegatives<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
        N, width, height, channels, negDiffData, negSampleAData, negSampleBData,
        negAlpha, lossFunctor, weighting, gradA, gradB);
    CUDA_POST_KERNEL_CHECK;

}

template void backwardNegativeWrapper<float, HingeLossFunctor,NoWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const HingeLossFunctor<float>,float *, float *, NoWeighting<float>);
//template void backwardNegativeWrapper<float, HuberHingeLossFunctor,NoWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const HuberHingeLossFunctor<float>,float *, float *, NoWeighting<float>);
//template void backwardNegativeWrapper<float, NegativeExponentialLossFunctor,NoWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const NegativeExponentialLossFunctor<float>,float *, float *, NoWeighting<float>);

template void backwardNegativeWrapper<double, HingeLossFunctor,NoWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const HingeLossFunctor<double>,double *, double *, NoWeighting<double>);
//template void backwardNegativeWrapper<double, HuberHingeLossFunctor,NoWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const HuberHingeLossFunctor<double>,double *, double *, NoWeighting<double>);
//template void backwardNegativeWrapper<double, NegativeExponentialLossFunctor,NoWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const NegativeExponentialLossFunctor<double>,double *, double *, NoWeighting<double>);


template void backwardNegativeWrapper<float, HingeLossFunctor,InputWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const HingeLossFunctor<float>,float *, float *, InputWeighting<float>);
//template void backwardNegativeWrapper<float, HuberHingeLossFunctor,InputWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const HuberHingeLossFunctor<float>,float *, float *, InputWeighting<float>);
//template void backwardNegativeWrapper<float, NegativeExponentialLossFunctor,InputWeighting>(const int, const int, const int, const int, const float *, const float *, const float *, const float, const NegativeExponentialLossFunctor<float>,float *, float *, InputWeighting<float>);

template void backwardNegativeWrapper<double, HingeLossFunctor,InputWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const HingeLossFunctor<double>,double *, double *, InputWeighting<double>);
//template void backwardNegativeWrapper<double, HuberHingeLossFunctor,InputWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const HuberHingeLossFunctor<double>,double *, double *, InputWeighting<double>);
//template void backwardNegativeWrapper<double, NegativeExponentialLossFunctor,InputWeighting>(const int, const int, const int, const int, const double *, const double *, const double *, const double, const NegativeExponentialLossFunctor<double>,double *, double *, InputWeighting<double>);



} // namespace caffe

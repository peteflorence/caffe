#include "caffe/layers/dense_correspondence_layer.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/transformMatrix.hpp"

#include <cmath>
#include <numeric>

namespace caffe {

// -=-=-=- creation code -=-=-=-
template <typename Dtype,
          template <typename> class PositiveLossFunctorT,
          template <typename> class NegativeLossFunctorT,
          template <typename> class PositiveMatchSelectorT,
          template <typename> class NegativeMatchSelectorT,
          template <typename> class LossBalancingFunctorT>
inline DenseCorrespondenceLayerImplBase<Dtype> * createImplementation(const DenseCorrespondenceParameter & param,
                                                                      const int width, const int height,
                                                                      PositiveLossFunctorT<Dtype> posLossFunctor,
                                                                      NegativeLossFunctorT<Dtype> negLossFunctor,
                                                                      PositiveMatchSelectorT<Dtype> positiveSelector,
                                                                      NegativeMatchSelectorT<Dtype> negativeSelector,
                                                                      LossBalancingFunctorT<Dtype> lossBalancingFunctor) {

#define MATCH_FINDING_CASE(protoName,matchFinder) \
    case DenseCorrespondenceParameter_MatchFinding_##protoName: \
        return new DenseCorrespondenceLayerImpl<Dtype, \
                                                matchFinder, \
                                                PositiveMatchSelectorT, \
                                                PositiveLossFunctorT, \
                                                NegativeMatchSelectorT, \
                                                NegativeLossFunctorT, \
                                                LossBalancingFunctorT>(positiveSelector, \
                                                                       posLossFunctor, \
                                                                       negativeSelector, \
                                                                       negLossFunctor, \
                                                                       lossBalancingFunctor, \
                                                                       param.focal_length_x(), \
                                                                       param.focal_length_y(), \
                                                                       param.principal_point_x(), \
                                                                       param.principal_point_y(), \
                                                                       param.enable_matchless())

    switch (param.match_finding()) {

        MATCH_FINDING_CASE(RIGID_MATCHING,RigidMatchFinder);
        MATCH_FINDING_CASE(FLANN_MATCHING,FLANNMatchFinder);
        MATCH_FINDING_CASE(RANDOM_MATCHING,RandomMatchFinder);

//    case DenseCorrespondenceParameter_MatchFinding_RIGID_MATCHING:
//        return new DenseCorrespondenceLayerImpl<Dtype,
//                                                RigidMatchFinder,
//                                                PositiveMatchSelectorT,
//                                                PositiveLossFunctorT,
//                                                NegativeMatchSelectorT,
//                                                NegativeLossFunctorT,
//                                                LossBalancingFunctorT>(positiveSelector,
//                                                                       posLossFunctor,
//                                                                       negativeSelector,
//                                                                       negLossFunctor,
//                                                                       lossBalancingFunctor,
//                                                                       param.focal_length_x(),
//                                                                       param.focal_length_y(),
//                                                                       param.principal_point_x(),
//                                                                       param.principal_point_y(),
//                                                                       param.enable_matchless());
//    case DenseCorrespondenceParameter_MatchFinding_FLANN_MATCHING:
//        return new DenseCorrespondenceLayerImpl<Dtype,
//                                                FLANNMatchFinder,
//                                                PositiveMatchSelectorT,
//                                                PositiveLossFunctorT,
//                                                NegativeMatchSelectorT,
//                                                NegativeLossFunctorT,
//                                                LossBalancingFunctorT>(positiveSelector,
//                                                                       posLossFunctor,
//                                                                       negativeSelector,
//                                                                       negLossFunctor,
//                                                                       lossBalancingFunctor,
//                                                                       param.focal_length_x(),
//                                                                       param.focal_length_y(),
//                                                                       param.principal_point_x(),
//                                                                       param.principal_point_y(),
//                                                                       param.enable_matchless());
//    case DenseCorrespondenceParameter_MatchFinding_RANDOM_MATCHING:
//        return new DenseCorrespondenceLayerImpl<Dtype,
//                                                RandomMatchFinder,
//                                                PositiveMatchSelectorT,
//                                                PositiveLossFunctorT,
//                                                NegativeMatchSelectorT,
//                                                NegativeLossFunctorT,
//                                                LossBalancingFunctorT>(positiveSelector,
//                                                                       posLossFunctor,
//                                                                       negativeSelector,
//                                                                       negLossFunctor,
//                                                                       lossBalancingFunctor,
//                                                                       param.focal_length_x(),
//                                                                       param.focal_length_y(),
//                                                                       param.principal_point_x(),
//                                                                       param.principal_point_y(),
//                                                                       param.enable_matchless());
    }

    return 0;
}

template <typename Dtype,
          template <typename> class PositiveLossFunctorT,
          template <typename> class NegativeLossFunctorT,
          template <typename> class PositiveMatchSelectorT,
          template <typename> class NegativeMatchSelectorT>
inline DenseCorrespondenceLayerImplBase<Dtype> * createImplementation(const DenseCorrespondenceParameter & param,
                                                                      const int width, const int height,
                                                                      PositiveLossFunctorT<Dtype> posLossFunctor,
                                                                      NegativeLossFunctorT<Dtype> negLossFunctor,
                                                                      PositiveMatchSelectorT<Dtype> positiveSelector,
                                                                      NegativeMatchSelectorT<Dtype> negativeSelector) {

#define LOSS_BALANCING_CASE(protoName,instantiation) \
    case DenseCorrespondenceParameter_LossBalancing_##protoName: \
        return createImplementation(param,width,height,posLossFunctor,negLossFunctor,positiveSelector,negativeSelector, \
                                    instantiation); \
        break

    switch(param.loss_balancing()) {

//        LOSS_BALANCING_CASE(NUM_POSITIVES,NormalizeByNumPositivesFunctor<Dtype>());
//        LOSS_BALANCING_CASE(TOTAL_SAMPLES,NormalizeTotalFunctor<Dtype>());
//        LOSS_BALANCING_CASE(UNIFORM,NormalizeUniformFunctor<Dtype>());
        LOSS_BALANCING_CASE(REWEIGHTED_UNIFORM,ReweightedNormalizeUniformFunctor<Dtype>(param.positive_weight(),param.negative_weight()));

//    case DenseCorrespondenceParameter_LossBalancing_NUM_POSITIVES:
//        return createImplementation(param,width,height,posLossFunctor,negLossFunctor,positiveSelector,negativeSelector,
//                                    NormalizeByNumPositivesFunctor<Dtype>());
//        break;
//    case DenseCorrespondenceParameter_LossBalancing_TOTAL_SAMPLES:
//        return createImplementation(param,width,height,posLossFunctor,negLossFunctor,positiveSelector,negativeSelector,
//                                    NormalizeTotalFunctor<Dtype>());
//        break;
//    case DenseCorrespondenceParameter_LossBalancing_UNIFORM:
//        return createImplementation(param,width,height,posLossFunctor,negLossFunctor,positiveSelector,negativeSelector,
//                                    NormalizeUniformFunctor<Dtype>());
//        break;
//    case DenseCorrespondenceParameter_LossBalancing_REWEIGHTED_UNIFORM:
//        return createImplementation(param,width,height,posLossFunctor,negLossFunctor,positiveSelector,negativeSelector,
//                                    ReweightedNormalizeUniformFunctor<Dtype>(param.positive_weight(),param.negative_weight()));
//        break;
    default:
        throw std::runtime_error("some options have been temporarily disabled for faster builds");
    }
    return 0;

}

template <typename Dtype,
          template <typename> class PositiveLossFunctorT,
          template <typename> class NegativeLossFunctorT,
          template <typename> class PositiveMatchSelectorT>
inline DenseCorrespondenceLayerImplBase<Dtype> * createImplementation(const DenseCorrespondenceParameter & param,
                                                                      const int width, const int height,
                                                                      PositiveLossFunctorT<Dtype> posLossFunctor,
                                                                      NegativeLossFunctorT<Dtype> negLossFunctor,
                                                                      PositiveMatchSelectorT<Dtype> positiveSelector) {
#define NEGATIVE_SELECTION_CASE(protoName,instantiation)   \
    case DenseCorrespondenceParameter_NegativeSelection_##protoName: \
        return createImplementation(param,width,height,posLossFunctor,negLossFunctor,positiveSelector, \
                                    instantiation); \
        break

    switch (param.negative_selection()) {

//        NEGATIVE_SELECTION_CASE(ALL_NEGATIVES,AllNegativesSelector<Dtype>(width,height));
        NEGATIVE_SELECTION_CASE(RANDOM_NEGATIVES,RandomNegativesSelector<Dtype>(param.negative_samples()));
//        NEGATIVE_SELECTION_CASE(HARD_NEGATIVES,HardNegativesSelector<Dtype>(param.negative_samples()));

//    case DenseCorrespondenceParameter_NegativeSelection_ALL_NEGATIVES:
//        return createImplementation(param,width,height,posLossFunctor,negLossFunctor,positiveSelector,
//                                    AllNegativesSelector<Dtype>(width,height));
//        break;
//    case DenseCorrespondenceParameter_NegativeSelection_RANDOM_NEGATIVES:
//        return createImplementation(param,width,height,posLossFunctor,negLossFunctor,positiveSelector,
//                                    RandomNegativesSelector<Dtype>(param.negative_samples()));
//        break;
//    case DenseCorrespondenceParameter_NegativeSelection_HARD_NEGATIVES:
//        return createImplementation(param,width,height,posLossFunctor,negLossFunctor,positiveSelector,
//                                    HardNegativesSelector<Dtype>(param.negative_samples()));
//        break;
    default:
        throw std::runtime_error("some options have been temporarily disabled for faster builds");
    }
    return 0;

}

template <typename Dtype,
          template <typename> class PositiveLossFunctorT,
          template <typename> class NegativeLossFunctorT>
inline DenseCorrespondenceLayerImplBase<Dtype> * createImplementation(const DenseCorrespondenceParameter & param,
                                                                      const int width, const int height,
                                                                      PositiveLossFunctorT<Dtype> posLossFunctor,
                                                                      NegativeLossFunctorT<Dtype> negLossFunctor) {

#define POSITIVE_SELECTION_CASE(protoName,instantiation) \
    case DenseCorrespondenceParameter_PositiveSelection_##protoName: \
        return createImplementation(param,width,height,posLossFunctor,negLossFunctor, \
                                    instantiation); \
        break

    switch (param.positive_selection()) {
//        POSITIVE_SELECTION_CASE(ALL_POSITIVES,AllPositiveMatchesSelector<Dtype>(width,height));
        POSITIVE_SELECTION_CASE(RANDOM_POSITIVES,RandomPositiveMatchesSelector<Dtype>(param.positive_samples()));

//    case DenseCorrespondenceParameter_PositiveSelection_ALL_POSITIVES:
//        return createImplementation(param,width,height,posLossFunctor,negLossFunctor,
//                                    AllPositiveMatchesSelector<Dtype>(width,height));
//        break;
//    case DenseCorrespondenceParameter_PositiveSelection_RANDOM_POSITIVES:
//        return createImplementation(param,width,height,posLossFunctor,negLossFunctor,
//                                    RandomPositiveMatchesSelector<Dtype>(param.positive_samples()));
    default:
        throw std::runtime_error("some options have been temporarily disabled for faster builds");
        break;
    }

    return 0;

}


template <typename Dtype,
          template <typename> class PositiveLossFunctorT>
inline DenseCorrespondenceLayerImplBase<Dtype> * createImplementation(const DenseCorrespondenceParameter & param,
                                                                      const int width, const int height,
                                                                      PositiveLossFunctorT<Dtype> posLossFunctor) {

#define NEGATIVE_LOSS_CASE(protoName,instantiation) \
    case DenseCorrespondenceParameter_NegativeLoss_##protoName: \
        return createImplementation(param,width,height,posLossFunctor, \
                                    instantiation); \
        break

    switch (param.negative_loss()) {

        NEGATIVE_LOSS_CASE(HINGE,HingeLossFunctor<Dtype>(param.margin()));
//        NEGATIVE_LOSS_CASE(HUBER_HINGE,HuberHingeLossFunctor<Dtype>(param.margin(),param.negative_huber_delta()));
//        NEGATIVE_LOSS_CASE(NEGATIVE_EXPONENTIAL,NegativeExponentialLossFunctor<Dtype>(param.negative_exp_sigma()));

//    case DenseCorrespondenceParameter_NegativeLoss_HINGE:
//        return createImplementation(param,width,height,posLossFunctor,
//                                    HingeLossFunctor<Dtype>(param.margin()));
//        break;
//    case DenseCorrespondenceParameter_NegativeLoss_HUBER_HINGE:
//        return createImplementation(param,width,height,posLossFunctor,
//                                    HuberHingeLossFunctor<Dtype>(param.margin(),param.negative_huber_delta()));
//        break;
//    case DenseCorrespondenceParameter_NegativeLoss_NEGATIVE_EXPONENTIAL:
//        return createImplementation(param,width,height,posLossFunctor,
//                                    NegativeExponentialLossFunctor<Dtype>(param.negative_exp_sigma()));
//        break;
    default:
        throw std::runtime_error("some options have been temporarily disabled for faster builds");
    }
    return 0;

}

template <typename Dtype>
inline DenseCorrespondenceLayerImplBase<Dtype> * createImplementation(const DenseCorrespondenceParameter & param,
                                                                      const int width, const int height) {

#define POSITIVE_LOSS_CASE(protoName,instantiation) \
    case DenseCorrespondenceParameter_PositiveLoss_##protoName: \
        return createImplementation(param,width,height,instantiation); \
        break

    switch (param.positive_loss()) {

        POSITIVE_LOSS_CASE(L2,SquaredLossFunctor<Dtype>());
//        POSITIVE_LOSS_CASE(HUBER,HuberLossFunctor<Dtype>(param.positive_huber_delta()));
//        POSITIVE_LOSS_CASE(TUKEY,TukeyLossFunctor<Dtype>(param.positive_tukey_c()));

//    case DenseCorrespondenceParameter_PositiveLoss_L2:
//        return createImplementation(param,width,height,SquaredLossFunctor<Dtype>());
//        break;
//    case DenseCorrespondenceParameter_PositiveLoss_HUBER:
//        return createImplementation(param,width,height,HuberLossFunctor<Dtype>(param.positive_huber_delta()));
//        break;
//    case DenseCorrespondenceParameter_PositiveLoss_TUKEY:
//        return createImplementation(param,width,height,TukeyLossFunctor<Dtype>(param.positive_tukey_c()));
//        break;
    default:
        throw std::runtime_error("some options have been temporarily disabled for faster builds");
    }
    return 0;

}


//template <typename Dtype>
//inline void interpolateVertex(const Blob<Dtype> * vertMap, const Dtype u, const Dtype v, Dtype * vert) {

//    const Dtype * vertData = vertMap->cpu_data();

//    const int width = vertMap->width();
//    const int height = vertMap->height();

//    const int baseU = u;
//    const int baseV = v;
//    const int nextU = 1 + baseU;
//    const int nextV = 1 + baseV;
//    const Dtype offU = u - baseU;
//    const Dtype offV = v - baseV;

//    vert[0] = vert[1] = vert[2] = Dtype(0);
//    Dtype weight = 0;
//    if (!std::isnan(vertData[baseU + width*baseV])) {
//        const Dtype w = (1-offU)*(1-offV);
//        weight += w;
//        for (int c=0; c<3; ++c) {
//            vert[c] += w*vertData[baseU + width*(baseV + height*c)];
//        }
//    }
//    if (!std::isnan(vertData[baseU + width*nextV])) {
//        const Dtype w = (1-offU)*offV;
//        weight += w;
//        for (int c=0; c<3; ++c) {
//            vert[c] += w*vertData[baseU + width*(nextV + height*c)];
//        }
//    }
//    if (!std::isnan(vertData[nextU + width*baseV])) {
//        const Dtype w = offU*(1-offV);
//        weight += w;
//        for (int c=0; c<3; ++c) {
//            vert[c] += w*vertData[nextU + width*(baseV + height*c)];
//        }
//    }
//    if (!std::isnan(vertData[nextU + width*nextV])) {
//        const float w = offU*offV;
//        weight += w;
//        for (int c=0; c<3; ++c) {
//            vert[c] += w*vertData[nextU + width*(nextV + height*c)];
//        }
//    }

//    const float oneOverWeight = 1./weight;
//    for (int c=0; c<3; ++c) {
//        vert[c] *= oneOverWeight;
//    }

//}

//template <typename Dtype>
//void interpolateRepresentation(const Blob<Dtype> * denseRepresentation, const Dtype u, const Dtype v, Dtype * sampledRepresentation) {

//    const Dtype * denseData = denseRepresentation->cpu_data();

//    const int width = denseRepresentation->width();
//    const int height = denseRepresentation->height();
//    const int channels = denseRepresentation->channels();

//    const int baseU = u;
//    const int baseV = v;
//    const int nextU = 1 + baseU;
//    const int nextV = 1 + baseV;
//    const Dtype offU = u - baseU;
//    const Dtype offV = v - baseV;

//    const Dtype wbb = (1-offU)*(1-offV);
//    const Dtype wbn = (1-offU)*( offV );
//    const Dtype wnb = ( offU )*(1-offV);
//    const Dtype wnn = ( offU )*( offV );

//    for (int c=0; c<channels; ++c) {
//        sampledRepresentation[c] =
//                wbb*denseData[baseU + width*(baseV + height*c)] +
//                wbn*denseData[baseU + width*(nextV + height*c)] +
//                wnb*denseData[nextU + width*(baseV + height*c)] +
//                wnn*denseData[nextU + width*(nextV + height*c)];
//    }

//}

//template <typename Dtype>
//void deInterpolateGradient(Blob<Dtype> * denseRepresentation, const Dtype u, const Dtype v, const Dtype * interpolatedGradient, const Dtype alpha=Dtype(1)) {

//    Dtype * denseDiff = denseRepresentation->mutable_cpu_diff();

//    const int width = denseRepresentation->width();
//    const int height = denseRepresentation->height();
//    const int channels = denseRepresentation->channels();

//    const int baseU = u;
//    const int baseV = v;
//    const int nextU = 1 + baseU;
//    const int nextV = 1 + baseV;
//    const Dtype offU = u - baseU;
//    const Dtype offV = v - baseV;

//    const Dtype wbb = (1-offU)*(1-offV);
//    const Dtype wbn = (1-offU)*( offV );
//    const Dtype wnb = ( offU )*(1-offV);
//    const Dtype wnn = ( offU )*( offV );

//    for (int c=0; c<channels; ++c) {
//        denseDiff[baseU + width*(baseV + height*c)] += alpha*wbb*interpolatedGradient[c];
//        denseDiff[baseU + width*(nextV + height*c)] += alpha*wbn*interpolatedGradient[c];
//        denseDiff[nextU + width*(baseV + height*c)] += alpha*wnb*interpolatedGradient[c];
//        denseDiff[nextU + width*(nextV + height*c)] += alpha*wnn*interpolatedGradient[c];
//    }

//}

template <typename Dtype>
void DenseCorrespondenceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> & bottom,
                                                 const vector<Blob<Dtype> *> & top) {


    if (impl_) {
        delete impl_;
    }

    std::cout << "creating implementation" << std::endl;
    impl_ = createImplementation<Dtype>(this->layer_param().dense_correspondence_param(),
                                        bottom[0]->width(),bottom[0]->height());

    impl_->LayerSetUp(bottom,top);

    std::cout << (int64_t)this << " --> " << (int64_t)impl_ << std::endl;

//    // -=-=-=-=- checks -=-=-=-=-

//    // make sure all blobs have the same num
//    const int nPairs = bottom[0]->num();
//    CHECK_EQ(nPairs,bottom[1]->num());
//    CHECK_EQ(nPairs,bottom[2]->num());
//    CHECK_EQ(nPairs,bottom[3]->num());
//    CHECK_EQ(nPairs,bottom[4]->num());

//    // make sure all blobs have the same shape
//    const int denseWidth = bottom[0]->width();
//    const int denseHeight = bottom[0]->height();
//    CHECK_EQ(denseWidth,bottom[1]->width());
//    CHECK_EQ(denseWidth,bottom[2]->width());
//    CHECK_EQ(denseWidth,bottom[3]->width());
//    CHECK_EQ(denseHeight,bottom[1]->height());
//    CHECK_EQ(denseHeight,bottom[2]->height());
//    CHECK_EQ(denseHeight,bottom[3]->height());

//    // make sure the dense representations have the same number of channels
//    const int denseChannels = bottom[0]->channels();
//    CHECK_EQ(denseChannels,bottom[1]->channels());

//    // make sure the vert maps have 3 channels
//    CHECK_EQ(3,bottom[2]->channels());
//    CHECK_EQ(3,bottom[3]->channels());

//    // make sure the transforms are 3x4
//    CHECK_EQ(3,bottom[4]->shape()[1]);
//    CHECK_EQ(4,bottom[4]->shape()[2]);

//    // -=-=-=-=- set params -=-=-=-=-
//    flX_ = this->layer_param().dense_correspondence_param().focal_length_x();
//    flY_ = this->layer_param().dense_correspondence_param().focal_length_y();
//    ppX_ = this->layer_param().dense_correspondence_param().principal_point_x();
//    ppY_ = this->layer_param().dense_correspondence_param().principal_point_y();

//    nPositiveSamplesPerFramePair_ = this->layer_param().dense_correspondence_param().positive_samples();
//    nNegativeSamplesPerFramePair_ = this->layer_param().dense_correspondence_param().negative_samples();

//    std::vector<int> sampleShape(3);
//    sampleShape[0] = nPairs;
//    sampleShape[1] = nPositiveSamplesPerFramePair_ + nNegativeSamplesPerFramePair_;
//    sampleShape[2] = 2;
//    samplesA_.Reshape(sampleShape);
//    samplesB_.Reshape(sampleShape);

//    std::vector<int> diffShape(3);
//    diffShape[0] = nPairs;
//    diffShape[1] = nPositiveSamplesPerFramePair_ + nNegativeSamplesPerFramePair_;
//    diffShape[2] = denseChannels;
//    diff_.Reshape(diffShape);

//    nSuccessfulPositiveSamples_.resize(nPairs);
}

template <typename Dtype>
void DenseCorrespondenceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> & bottom,
                                                  const vector<Blob<Dtype> *> & top) {

    impl_->Forward_cpu(bottom,top);

//    LOG(INFO) << "entering forward";

//    Dtype loss = 0;

//    const int numPairs = bottom[0]->num();

//    const int repWidth = bottom[0]->width();
//    const int repHeight = bottom[0]->height();
//    const int repChannels = bottom[0]->channels();

//    boost::uniform_int<> uDistribution(0,repWidth-1);
//    boost::uniform_int<> vDistribution(0,repHeight-1);

//    const Dtype margin = this->layer_param_.dense_correspondence_param().margin();

//    const bool doHardNegativeMining = this->layer_param().dense_correspondence_param().hard_negatives();
//    const bool doStochasticNegativeMining = this->layer_param().dense_correspondence_param().stochastic_mining();
//    const int nMiningSamples = this->layer_param().dense_correspondence_param().mining_samples();

////    static std::ofstream debugPositiveStream("/tmp/debugCasesPositive.txt");
////    static std::ofstream debugNegativeStream("/tmp/debugCasesNegative.txt");

//    // iterate over all frame pairs
//    for (int pair = 0; pair < numPairs; ++pair) {

//        //std::cout << "pair " << pair << std::endl;

//        const Dtype * repA = bottom[0]->cpu_data() + pair*bottom[0]->count(1);
//        const Dtype * repB = bottom[1]->cpu_data() + pair*bottom[1]->count(1);

//        const Dtype * vertsA = bottom[2]->cpu_data() + pair*bottom[2]->count(1);
//        //const Dtype * vertsB = bottom[3]->cpu_data() + pair*bottom[3]->count(1);

//        Dtype * sampleAData = samplesA_.mutable_cpu_data() + pair*samplesA_.count(1);
//        Dtype * sampleBData = samplesB_.mutable_cpu_data() + pair*samplesB_.count(1);

//        Dtype * diffData = diff_.mutable_cpu_data() + pair*diff_.count(1);

//        int sampleAttempts = 0;
//        bool breakingBad = false;

//        TransformationMatrix<Dtype> transformGlobalToB(bottom[4]->cpu_data() + pair*bottom[4]->count(1));

//        LOG(INFO) << transformGlobalToB;

//        //std::cout << transformGlobalToB << std::endl;

//        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- sample positives -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//        int i = 0;
//        for ( ; i < nPositiveSamplesPerFramePair_; ++i) {

////            std::cout << "positive " << i << std::endl;

//            Dtype ptA[3];
//            Dtype ptB[3];

//            Dtype uB;
//            Dtype vB;

//            int uA;
//            int vA;

//            do {

//                ++sampleAttempts;
//                do {

//                    ++sampleAttempts;

//                    // pick a random valid vertex from imageA
//                    do {

//                        ++sampleAttempts;
//                        static int maxSampleAttempts = 100*(nPositiveSamplesPerFramePair_);
//                        if (sampleAttempts >= maxSampleAttempts) {
//                            breakingBad = true;
//                            LOG(INFO) << "breaking";
//                            break;
//                        }

//                        uA = uDistribution(*caffe_rng());
//                        vA = vDistribution(*caffe_rng());

//                        ptA[0] = vertsA[uA + repWidth*(vA + repHeight*0)];
//                        ptA[1] = vertsA[uA + repWidth*(vA + repHeight*1)];
//                        ptA[2] = vertsA[uA + repWidth*(vA + repHeight*2)];


//                    } while (std::isnan(ptA[0]));

//                    if (breakingBad) {
//                        break;
//                    }

////                    std::cout << "trying A = (" << uA << ", " << vA << ")..." << std::endl;
////                    std::cout << "corresponding vertex = (" << ptA[0] << ", " << ptA[1] << ", " << ptA[2] << ")" << std::endl;

//                    // find corresponding pixel of imageB
//                    Point<Dtype,3> _ptA(ptA,1);
//                    Dtype ptBLocal[3];
//                    Point<Dtype,3> _ptBLocal(ptBLocal,1);
//                    transformGlobalToB.apply(_ptA,_ptBLocal);

//                    uB = ptBLocal[0]*flX_/ptBLocal[2] + ppX_;
//                    vB = ptBLocal[1]*flY_/ptBLocal[2] + ppY_;

//                } while(uB < 0 || uB >= repWidth - 2 ||
//                        vB < 0 || vB >= repHeight - 2);

//                if (breakingBad) {
//                    break;
//                }

////                std::cout << "getting vertex for B = (" << uB << ", " << vB << ")..." << std::endl;

//                interpolateVertex(bottom[3],uB,vB,ptB);

//            } while (std::isnan(ptB[0]) || ((ptA[0]-ptB[0])*(ptA[0]-ptB[0]) +
//                                            (ptA[1]-ptB[1])*(ptA[1]-ptB[1]) +
//                                            (ptA[2]-ptB[2])*(ptA[2]-ptB[2])) > 0.005);

//            if (breakingBad) {
//                break;
//            }

////            debugPositiveStream << globalPair << "A " << uA << " " << vA << " ";
////            debugPositiveStream << globalPair << "B " << uB << " " << vB << std::endl;

//            // save sample points for gradient computation
//            sampleAData[0 + 2*i] = uA;
//            sampleAData[1 + 2*i] = vA;
//            sampleBData[0 + 2*i] = uB;
//            sampleBData[1 + 2*i] = vB;

//            std::vector<Dtype> representationB(repChannels);
//            interpolateRepresentation(bottom[1],uB,vB,representationB.data());

//            std::vector<Dtype> representationA(repChannels);
//            for (int c=0; c<repChannels; ++c) {
//                representationA[c] = repA[uA + repWidth*(vA + repHeight*c)];
//            }

//            Dtype * thisDiff = diffData + i*repChannels;
//            caffe_sub(repChannels,representationA.data(),representationB.data(),thisDiff);

//            const Dtype distSquared = caffe_cpu_dot(repChannels,thisDiff,thisDiff);

//            loss += distSquared;

//        }

//        nSuccessfulPositiveSamples_[pair] = i;
//        LOG(INFO) << i << " successful positive samples";

//        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- sample negatives -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//        for (int i=0; i < nNegativeSamplesPerFramePair_; ++i) {

//            const Dtype * rep = i > ( nNegativeSamplesPerFramePair_ / 2 ) ? repB : repA;

//            // sample random point in A
//            int uA = uDistribution(*caffe_rng());
//            int vA = vDistribution(*caffe_rng());

//            std::vector<Dtype> representationA(repChannels);
//            for (int c=0; c<repChannels; ++c) {
//                representationA[c] = rep[uA + repWidth*(vA + repHeight*c)];
//            }

//            int uB;
//            int vB;

//            if (doHardNegativeMining) {

//                if (doStochasticNegativeMining) {

//                    int minUB = -1;
//                    int minVB = -1;
//                    Dtype minDistSquared = std::numeric_limits<Dtype>::infinity();

//                    for (int j = 0; j < nMiningSamples; ++j) {

//                        // sample point in B at least 2 pixels away
//                        do {
//                            uB = uDistribution(*caffe_rng());
//                            vB = vDistribution(*caffe_rng());
//                        } while ( ((uA-uB)*(uA-uB) + (vA-vB)*(vA-vB)) < 4 );

//                        Dtype distSquared = Dtype(0);
//                        for (int c=0; c<repChannels; ++c) {
//                            const Dtype diff = representationA[c] - rep[uB + repWidth*(vB + repHeight*c)];
//                            distSquared += diff*diff;
//                        }

//                        if (distSquared < minDistSquared) {
//                            minDistSquared = distSquared;
//                            minUB = uB;
//                            minVB = vB;
//                        }

//                    }

//                    uB = minUB;
//                    vB = minVB;

//                } else {

//                    LOG(ERROR) << "non-stochastic hard negative mining not implemented yet";
//                    std::abort();

//                }

//            } else {

//                // sample point in B at least 2 pixels away
//                do {
//                    uB = uDistribution(*caffe_rng());
//                    vB = vDistribution(*caffe_rng());
//                } while ( ((uA-uB)*(uA-uB) + (vA-vB)*(vA-vB)) < 4 );

//            }

//            // save sample points for gradient computation
//            sampleAData[0 + 2*(nPositiveSamplesPerFramePair_ + i)] = uA;
//            sampleAData[1 + 2*(nPositiveSamplesPerFramePair_ + i)] = vA;
//            sampleBData[0 + 2*(nPositiveSamplesPerFramePair_ + i)] = uB;
//            sampleBData[1 + 2*(nPositiveSamplesPerFramePair_ + i)] = vB;

//            // TODO: just use caffe_strided_dot

//            std::vector<Dtype> representationB(repChannels);
//            for (int c=0; c<repChannels; ++c) {
//                representationB[c] = rep[uB + repWidth*(vB + repHeight*c)];
//            }

//            Dtype * thisDiff = diffData + (nPositiveSamplesPerFramePair_ + i)*repChannels;
//            caffe_sub(repChannels,representationA.data(),representationB.data(),thisDiff);

//            const Dtype distSquared = caffe_cpu_dot(repChannels,thisDiff,thisDiff);
//            const Dtype dist = std::max<Dtype>(margin - std::sqrt(distSquared),Dtype(0));

//            loss += dist*dist;

//        }

//    }

//    top[0]->mutable_cpu_data()[0] = loss /
//            (nNegativeSamplesPerFramePair_*numPairs + std::accumulate(nSuccessfulPositiveSamples_.begin(),nSuccessfulPositiveSamples_.end(),0));

//    LOG(INFO) << "exiting forward";

}

template <typename Dtype>
void DenseCorrespondenceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> & top,
                                                   const vector<bool> & propagate_down,
                                                   const vector<Blob<Dtype>*> & bottom) {

    impl_->Backward_cpu(top,propagate_down,bottom);

//    LOG(INFO) << "entering backward";

//    const int numPairs = bottom[0]->num();

//    const int repWidth = bottom[0]->width();
//    const int repHeight = bottom[0]->height();
//    const int repChannels = bottom[0]->channels();

//    const Dtype alpha = top[0]->cpu_diff()[0] /
//            (nNegativeSamplesPerFramePair_*numPairs + std::accumulate(nSuccessfulPositiveSamples_.begin(),nSuccessfulPositiveSamples_.end(),0));

//    caffe_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_cpu_diff());
//    caffe_set(bottom[1]->count(),Dtype(0),bottom[1]->mutable_cpu_diff());

//    const Dtype margin = this->layer_param_.dense_correspondence_param().margin();

//    for (int pair = 0; pair < numPairs; ++pair) {

//        const Dtype * sampleAData = samplesA_.cpu_data() + pair*samplesA_.count(1);
//        const Dtype * sampleBData = samplesB_.cpu_data() + pair*samplesB_.count(1);

//        const Dtype * diffData = diff_.cpu_data() + pair*diff_.count(1);

//        Dtype * diffA = bottom[0]->mutable_cpu_diff() + pair*bottom[0]->count(1);
//        Dtype * diffB = bottom[1]->mutable_cpu_diff() + pair*bottom[1]->count(1);

//        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- gradients for positives -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//        for (int i = 0; i < nSuccessfulPositiveSamples_[pair]; ++i) {

//            const int uA = std::floor(sampleAData[0 + 2*i] + 0.5);
//            const int vA = std::floor(sampleAData[1 + 2*i] + 0.5);
//            const Dtype uB = sampleBData[0 + 2*i];
//            const Dtype vB = sampleBData[1 + 2*i];

//            const Dtype * thisDiff = diffData + i*repChannels;
//            for (int c=0; c<repChannels; ++c) {
//                diffA[uA + repWidth*(vA + repHeight*c)] += alpha*thisDiff[c];
//            }
//            deInterpolateGradient(bottom[1],uB,vB,thisDiff,-alpha);

//        }

//        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- gradients for negatives -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//        for (int i = 0; i < nNegativeSamplesPerFramePair_; ++i) {

//            Dtype * diff = i > (nNegativeSamplesPerFramePair_ / 2) ? diffB : diffA;

//            const int uA = std::floor(sampleAData[0 + 2*(nPositiveSamplesPerFramePair_ + i)] + 0.5);
//            const int vA = std::floor(sampleAData[1 + 2*(nPositiveSamplesPerFramePair_ + i)] + 0.5);
//            const int uB = std::floor(sampleBData[0 + 2*(nPositiveSamplesPerFramePair_ + i)] + 0.5);
//            const int vB = std::floor(sampleBData[1 + 2*(nPositiveSamplesPerFramePair_ + i)] + 0.5);

//            const Dtype * thisDiff = diffData + (nPositiveSamplesPerFramePair_ + i)*repChannels;

//            const Dtype distSquared = caffe_cpu_dot(repChannels,thisDiff,thisDiff);
//            const Dtype dist = std::sqrt(distSquared);
//            const Dtype mdist = margin - dist;
//            const Dtype beta = -alpha * mdist / (dist + Dtype(1e-4));

//            if (mdist > Dtype(0)) {

//                for (int c=0; c<repChannels; ++c) {
//                    diff[uA + repWidth*(vA + repHeight*c)] += beta*thisDiff[c];
//                    diff[uB + repWidth*(vB + repHeight*c)] -= beta*thisDiff[c];
//                }

//            }

//        }

//    }

//    LOG(INFO) << "exiting backward";

}

//template <typename Dtype>
//void DenseCorrespondenceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*> & top,
//                                                   const vector<bool> & propagate_down,
//                                                   const vector<Blob<Dtype>*> & bottom) {

//    impl_->Backward_gpu(top,propagate_down,bottom);

//}

#ifdef CPU_ONLY
STUB_GPU(DenseCorrespondenceLayer);
#endif

INSTANTIATE_CLASS(DenseCorrespondenceLayer);
REGISTER_LAYER_CLASS(DenseCorrespondence);

} // namespace caffe

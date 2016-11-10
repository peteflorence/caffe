#ifndef CAFFE_DENSE_CORRESPONDENCE_LAYER_HPP_
#define CAFFE_DENSE_CORRESPONDENCE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/dense_correspondence_layer_impl.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
class DenseCorrespondenceLayer : public LossLayer<Dtype> {
public:

    explicit DenseCorrespondenceLayer(const LayerParameter & param)
        : LossLayer<Dtype>(param), impl_(0) {

        std::cout << "constructing " << (int64_t)this << std::endl;

    }

    ~DenseCorrespondenceLayer() {
        std::cout << "destructing " << (int64_t)this << std::endl;
        delete impl_;
    }

    virtual void LayerSetUp(const vector<Blob<Dtype> *> & bottom,
                            const vector<Blob<Dtype> *> & top);

//    virtual void Reshape(const vector<Blob<Dtype> *> & bottom,
//                         const vector<Blob<Dtype> *> & top);

    virtual inline const char * type() const { return "DenseCorrespondenceLayer"; }

    // 0 - representation A (C x W x H)
    // 1 - representation B (C x W x H)
    // 2 - vertices A       (3 x W x H)
    // 3 - vertices B       (3 x W x H)
    // 4 - transforms       (3 x 4)
    virtual inline int ExactNumBottomBlobs() const { return 5; }

    virtual inline int ExactNumTopBlobs() const { return 1; }

    inline const Blob<Dtype> & samplesA() const { return impl_->samplesA(); }

    inline const Blob<Dtype> & samplesB() const { return impl_->samplesB(); }

    inline const int numPositivesPossible() const {
        std::cout << "(" << (int64_t)this << ")" << std::endl;
        std::cout << "implementation: " << (int64_t)impl_ << std::endl;
        return impl_->numPositivesPossible();
    }

    inline const int numNegativesPossible() const { return impl_->numNegativesPossible(); }

protected:

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
//    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        impl_->Backward_gpu(top,propagate_down,bottom);
    }

//    int nPositiveSamplesPerFramePair_;
//    int nNegativeSamplesPerFramePair_;
//    vector<int> nSuccessfulPositiveSamples_;

//    Dtype flX_, flY_, ppX_, ppY_;

//    Blob<Dtype> samplesA_, samplesB_;
//    Blob<Dtype> diff_;

    DenseCorrespondenceLayerImplBase<Dtype> * impl_;

};



//enum DenseCorrespondenceParameter_PositiveLoss {
//    DenseCorrespondenceParameter_PositiveLoss_L2 = 0,
//    DenseCorrespondenceParameter_PositiveLoss_HUBER = 1
//};

//enum DenseCorrespondenceParameter_NegativeLoss {
//    DenseCorrespondenceParameter_NegativeLoss_HINGE = 0,
//    DenseCorrespondenceParameter_NegativeLoss_HUBER_HINGE = 1
//};

} // namespace caffe

#endif // CAFFE_DENSE_CORRESPONDENCE_LAYER_HPP_

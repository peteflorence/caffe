#ifndef CAFFE_POINT_H_
#define CAFFE_POINT_H_

#include <cuda_runtime.h>

namespace caffe {

template <typename Dtype, int D>
class Point {
public:

    __host__ __device__
    explicit Point(const Dtype * data, const int stride=1)
      : data_(data), stride_(stride) { }

    __host__ __device__
    Point & operator=(const Point & rhs) {
        for (int d=0; d<D; ++d) {
            (*this)[d] = rhs[d];
        }
        return *this;
    }

    __host__ __device__
    const Dtype & operator[](std::size_t index) const { return data()[stride_*index]; }

    __host__ __device__
    Dtype & operator[](std::size_t index) { return const_cast<Dtype &>(static_cast<const Point &>(*this)[index]); }

    __host__ __device__
    inline const Dtype * data() const { return data_; }

    __host__ __device__
    inline Dtype * data() { return const_cast<Dtype *>(static_cast<const Point &>(*this).data()); }

    template<int D2>
    __host__ __device__
    Point<Dtype,D2> sub(const int start) const { return Point<Dtype,D2>(data_ + stride_*start,stride_); }

private:
    const Dtype * data_;
    int stride_;
};

template <typename Dtype, int D>
__host__ __device__
inline static Dtype dot(const Point<Dtype,D> a,
                 const Point<Dtype,D> b) {
    Dtype sum = 0;
    for (int d=0; d<D; ++d) {
      sum += a[d]*b[d];
    }
    return sum;
}

} // namespace caffe

#endif // CAFFE_POINT_H_

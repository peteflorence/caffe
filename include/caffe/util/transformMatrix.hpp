#ifndef CAFFE_TRANSFORM_MATRIX_H_
#define CAFFE_TRANSFORM_MATRIX_H_

#include <cmath>
#include <cuda_runtime.h>
#include "caffe/util/point.hpp"
#include <ostream>

namespace caffe {

template <typename Dtype>
class TransformationMatrix {
public:

    __host__ __device__
    explicit TransformationMatrix(const Dtype * data)
        : data_(data) { }

    explicit TransformationMatrix(const Dtype * data, const Point<Dtype,6> & tangent)
        : data_(data) {

        //std::cout << "tangent: " << tangent[0] << " " << tangent[1] << " " << tangent[2] << std::endl;
        const Dtype wx = tangent[3]; const Dtype wy = tangent[4]; const Dtype wz = tangent[5];
        const Dtype theta = std::sqrt(wx*wx + wy*wy + wz*wz);
        if (theta == 0) {
            init(1, 0, 0, tangent[0],
                 0, 1, 0, tangent[1],
                 0, 0, 1, tangent[2]);
        } else {
            const Dtype Va = (1-std::cos(theta)) / (theta*theta);
            const Dtype Vb = (theta - std::sin(theta)) / (theta*theta*theta);

            Dtype V[9] = { 1 +          Vb*(- wz*wz - wy*wy),     Va*-wz + Vb*(wx*wy)          ,     Va* wy + Vb*(wx*wz)          ,
                               Va* wz + Vb*(wx*wy)          ,1 +           Vb*(- wz*wz - wx*wx),     Va*-wx + Vb*(wy*wz)          ,
                               Va*-wy + Vb*(wx*wz)          ,     Va* wx + Vb*(wy*wz)          ,1 +           Vb*(- wy*wy - wx*wx) };

            Dtype t[3];
            for (int r=0; r<3; ++r) {
                Point<Dtype,3> tt = tangent.template sub<3>(0);
                t[r] = dot(Point<Dtype,3>(V + 3*r,1),tt);
            }

            const Dtype a = std::sin(theta) / theta;
            const Dtype b = (1-std::cos(theta)) / (theta*theta);

            init(1 +         b*(- wz*wz - wy*wy),     a*-wz + b*(wx*wy)          ,     a* wy + b*(wx*wz)          , t[0],
                     a* wz + b*(wx*wy)          , 1 +         b*(- wz*wz - wx*wx),     a*-wx + b*(wy*wz)          , t[1],
                     a*-wy + b*(wx*wz)          ,     a* wx + b*(wy*wz)          , 1 +       + b*(- wy*wy - wx*wx), t[2]);
        }
    }

    TransformationMatrix & operator=(const TransformationMatrix & copy) {
        init(copy[0][0], copy[0][1], copy[0][2], copy[0][3],
             copy[1][0], copy[1][1], copy[1][2], copy[1][3],
             copy[2][0], copy[2][1], copy[2][2], copy[2][3]);
        return *this;
    }

    inline void setIdentity() {
        init( 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0);
    }

    __host__ __device__
    const Point<Dtype,4> operator[](std::size_t index) const { return Point<Dtype,4>(data_ + 4*index,1); }

    __host__ __device__
    Point<Dtype,4> operator[](std::size_t index) { return Point<Dtype,4>(data_ + 4*index,1); }

    __host__ __device__
    inline const Dtype * data() const { return data_; }

    __host__ __device__
    inline Dtype * data() { return const_cast<Dtype *>(static_cast<const TransformationMatrix &>(*this).data()); }

    __host__ __device__
    const Point<Dtype,4> row(std::size_t rowIndex) const { return (*this)[rowIndex]; }

    __host__ __device__
    const Point<Dtype,3> rotationRow(std::size_t rowIndex) const { return Point<Dtype,3>(data_ + 4*rowIndex,1); }

    __host__ __device__
    const Point<Dtype,3> col(std::size_t colIndex) const { return Point<Dtype,3>(data_ + colIndex, 4); }

    __host__ __device__
    inline void apply(const Point<Dtype,3> input, Point<Dtype,3> transformed) {
        for (int d = 0; d<3; ++d) {
            transformed[d] = (*this)[d][0]*input[0] + (*this)[d][1]*input[1] + (*this)[d][2]*input[2] + (*this)[d][3];
        }
    }

private:
    inline void init(Dtype d00, Dtype d01, Dtype d02, Dtype d03,
                     Dtype d10, Dtype d11, Dtype d12, Dtype d13,
                     Dtype d20, Dtype d21, Dtype d22, Dtype d23) {
        (*this)[0][0] = d00; (*this)[0][1] = d01; (*this)[0][2] = d02; (*this)[0][3] = d03;
        (*this)[1][0] = d10; (*this)[1][1] = d11; (*this)[1][2] = d12; (*this)[1][3] = d13;
        (*this)[2][0] = d20; (*this)[2][1] = d21; (*this)[2][2] = d22; (*this)[2][3] = d23;
    }

    const Dtype * data_;
};


// -=-=-=- non-member methods -=-=-=-
template <typename Dtype>
inline static void multiply(const TransformationMatrix<Dtype> & lhs, const TransformationMatrix<Dtype> & rhs, TransformationMatrix<Dtype> & res) {
  for (int r=0; r<3; ++r) {
    for (int c=0; c<4; ++c) {
      res[r][c] = dot(lhs.rotationRow(r),rhs.col(c));
    }
    res[r][3] += lhs.row(r)[3];
  }
}

template <typename Dtype>
inline static void leftCompose(const TransformationMatrix<Dtype> & lhs, TransformationMatrix<Dtype> & rhs) {
  Dtype tmp[12];
  TransformationMatrix<Dtype> result(tmp);
  multiply(lhs,rhs,result);
  caffe_copy(12,result.data(),rhs.data());
}

template <typename Dtype>
inline static void rightCompose(TransformationMatrix<Dtype> & lhs, const TransformationMatrix<Dtype> & rhs) {
  Dtype tmp[12];
  TransformationMatrix<Dtype> result(tmp);
  multiply(lhs,rhs,result);
  caffe_copy(12,result.data(),lhs.data());
}

template <typename Dtype>
inline std::ostream & operator<<(std::ostream & os, TransformationMatrix<Dtype> & mx) {
    os << mx[0][0] << " " << mx[0][1] << " " << mx[0][2] << " " << mx[0][3] << std::endl;
    os << mx[1][0] << " " << mx[1][1] << " " << mx[1][2] << " " << mx[1][3] << std::endl;
    os << mx[2][0] << " " << mx[2][1] << " " << mx[2][2] << " " << mx[2][3] << std::endl;
    return os;
}

} // namespace caffe

#endif // CAFFE_TRANSFORM_MATRIX_H_

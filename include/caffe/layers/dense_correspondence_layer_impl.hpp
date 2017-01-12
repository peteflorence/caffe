#include <cmath>
#include <numeric>
#include <vector>

#include "caffe/blob.hpp"

#include "caffe/util/rng.hpp"
#include "caffe/util/transformMatrix.hpp"
#include "caffe/util/math_functions.hpp"

#include "nanoflann.hpp"

using namespace std;

namespace caffe {

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                addition models
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype>
class SingleThreadedAddition {
public:
    inline static void add(Dtype & dst, const Dtype addition) {
        dst += addition;
    }
};

#ifdef __CUDACC__
template <typename Dtype>
class CudaAtomicAddition {
public:
    __device__
    inline static void add(Dtype & dst, const Dtype addition);
};

template <>
class CudaAtomicAddition<float> {
public:
    __device__
    inline static void add(float & dst, const float addition) {
        atomicAdd(&dst,addition);
    }
};

template <>
class CudaAtomicAddition<double> {
public:
    __device__
    inline static void add(double & dst, const double addition) {

    }
};
#endif // __CUDACC__

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                vertex interpolation
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype>
inline void interpolateVertex(const Dtype * vertData, const int width, const int height,
                              const Dtype u, const Dtype v, Dtype * vert) {

    const int baseU = u;
    const int baseV = v;
    const int nextU = 1 + baseU;
    const int nextV = 1 + baseV;
    const Dtype offU = u - baseU;
    const Dtype offV = v - baseV;

    vert[0] = vert[1] = vert[2] = Dtype(0);
    Dtype weight = 0;
    if (!std::isnan(vertData[baseU + width*baseV])) {
        const Dtype w = (1-offU)*(1-offV);
        weight += w;
        for (int c=0; c<3; ++c) {
            vert[c] += w*vertData[baseU + width*(baseV + height*c)];
        }
    }
    if (!std::isnan(vertData[baseU + width*nextV])) {
        const Dtype w = (1-offU)*offV;
        weight += w;
        for (int c=0; c<3; ++c) {
            vert[c] += w*vertData[baseU + width*(nextV + height*c)];
        }
    }
    if (!std::isnan(vertData[nextU + width*baseV])) {
        const Dtype w = offU*(1-offV);
        weight += w;
        for (int c=0; c<3; ++c) {
            vert[c] += w*vertData[nextU + width*(baseV + height*c)];
        }
    }
    if (!std::isnan(vertData[nextU + width*nextV])) {
        const Dtype w = offU*offV;
        weight += w;
        for (int c=0; c<3; ++c) {
            vert[c] += w*vertData[nextU + width*(nextV + height*c)];
        }
    }

    const Dtype oneOverWeight = 1./weight;
    for (int c=0; c<3; ++c) {
        vert[c] *= oneOverWeight;
    }

}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                representation interpolation
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype>
inline void interpolateRepresentation(const Dtype * denseData, const int width, const int height, const int channels,
                                      const Dtype u, const Dtype v, Dtype * sampledRepresentation) {

    const int baseU = u;
    const int baseV = v;
    const int nextU = 1 + baseU;
    const int nextV = 1 + baseV;
    const Dtype offU = u - baseU;
    const Dtype offV = v - baseV;

    const Dtype wbb = (1-offU)*(1-offV);
    const Dtype wbn = (1-offU)*( offV );
    const Dtype wnb = ( offU )*(1-offV);
    const Dtype wnn = ( offU )*( offV );

    for (int c=0; c<channels; ++c) {
        sampledRepresentation[c] =
                wbb*denseData[baseU + width*(baseV + height*c)] +
                wbn*denseData[baseU + width*(nextV + height*c)] +
                wnb*denseData[nextU + width*(baseV + height*c)] +
                wnn*denseData[nextU + width*(nextV + height*c)];
    }

}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                 gradient de-interpolation
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype,
          template <typename> class AdditionModel>
#ifdef __CUDACC__
__device__
#endif // __CUDACC__
void deInterpolateGradient(Dtype * denseDiff, const int width, const int height, const int channels,
                           const Dtype u, const Dtype v, const Dtype * interpolatedGradient, const Dtype alpha=Dtype(1)) {

    const int baseU = u;
    const int baseV = v;
    const int nextU = 1 + baseU;
    const int nextV = 1 + baseV;
    const Dtype offU = u - baseU;
    const Dtype offV = v - baseV;

    const Dtype wbb = (1-offU)*(1-offV);
    const Dtype wbn = (1-offU)*( offV );
    const Dtype wnb = ( offU )*(1-offV);
    const Dtype wnn = ( offU )*( offV );

    for (int c=0; c<channels; ++c) {
//        denseDiff[baseU + width*(baseV + height*c)] += alpha*wbb*interpolatedGradient[c];
//        denseDiff[baseU + width*(nextV + height*c)] += alpha*wbn*interpolatedGradient[c];
//        denseDiff[nextU + width*(baseV + height*c)] += alpha*wnb*interpolatedGradient[c];
//        denseDiff[nextU + width*(nextV + height*c)] += alpha*wnn*interpolatedGradient[c];
        AdditionModel<Dtype>::add(denseDiff[baseU + width*(baseV + height*c)], alpha*wbb*interpolatedGradient[c]);
        AdditionModel<Dtype>::add(denseDiff[baseU + width*(nextV + height*c)], alpha*wbn*interpolatedGradient[c]);
        AdditionModel<Dtype>::add(denseDiff[nextU + width*(baseV + height*c)], alpha*wnb*interpolatedGradient[c]);
        AdditionModel<Dtype>::add(denseDiff[nextU + width*(nextV + height*c)], alpha*wnn*interpolatedGradient[c]);
    }

}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                    match refinement
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype>
void getOptimalInterpolation1D(const Dtype target[3],
                               const Dtype a[3],
                               const Dtype b[3],
                               Dtype & t) {

    const Dtype residual[3] = { a[0] - target[0], a[1] - target[1], a[2] - target[2] };
    const Dtype derivative[3] = { b[0] - a[0], b[1] - a[1], b[2] - a[2] };

    const Dtype dDotD = derivative[0]*derivative[0] + derivative[1]*derivative[1] + derivative[2]*derivative[2];
    const Dtype dDotR = derivative[0]*residual[0]   + derivative[1]*residual[1]   + derivative[2]*residual[2];

    if (dDotD != Dtype(0)) {
        const Dtype solution = -dDotR/dDotD;

        t = std::max(Dtype(0),std::min(solution,Dtype(1-1e-4)));
    } else {
        t = 0;
    }
}

template <typename Dtype>
void solve2x2(const Dtype A[4],
              const Dtype b[2],
              Dtype x[2]) {

    const Dtype detA = A[0 + 2*0]*A[1 + 2*1] - A[1 + 2*0]*A[0 + 2*1];

    const Dtype det0 =    b[0]   *A[1 + 2*1] - A[1 + 2*0]*   b[1]   ;

    const Dtype det1 = A[0 + 2*0]*   b[1]    -    b[0]   *A[0 + 2*1];

    if (detA == Dtype(0)) {
        x[0] = x[1] = Dtype(0);
    } else {
        x[0] = det0 / detA;
        x[1] = det1 / detA;
    }
}

template <typename Dtype>
void getOptimalInterpolation2D(const Dtype target[3],
                               const Dtype a[3],
                               const Dtype b[3],
                               const Dtype c[3],
                               const Dtype d[3],
                               Dtype & t1,
                               Dtype & t2) {

    const Dtype wa = (1-t1)*(1-t2);
    const Dtype wb = ( t1 )*(1-t2);
    const Dtype wc = (1-t1)*( t2 );
    const Dtype wd = ( t1 )*( t2 );

    Dtype pInterp[3];
    Dtype residual[3];
    Dtype d_pInterp_d_t1[3];
    Dtype d_pInterp_d_t2[3];
    Dtype sysA[4] = { 0,0,  0,0 };
    Dtype sysb[2] = { 0,0 };

    for (int i=0; i<3; ++i) {
        pInterp[i] = wa*a[i] + wb*b[i] + wc*c[i] + wd*d[i];
        residual[i] = pInterp[i] - target[i];
        d_pInterp_d_t1[i] = (1-t2)*(b[i] - a[i]) + t2*(d[i] - c[i]);
        d_pInterp_d_t2[i] = (1-t1)*(c[i] - a[i]) + t1*(d[i] - b[i]);

        sysA[0 + 2*0] += d_pInterp_d_t1[i]*d_pInterp_d_t1[i];
        sysA[1 + 2*1] += d_pInterp_d_t2[i]*d_pInterp_d_t2[i];
        sysA[0 + 2*1] += d_pInterp_d_t1[i]*d_pInterp_d_t2[i];

        sysb[0] += d_pInterp_d_t1[i]*residual[i];
        sysb[1] += d_pInterp_d_t2[i]*residual[i];
    }
    sysA[1 + 2*0] = sysA[0 + 2*1];

    Dtype solution[2];
    solve2x2(sysA, sysb, solution);
    t1 = std::max(Dtype(0),std::min(-solution[0],Dtype(1-1e-4)));
    t2 = std::max(Dtype(0),std::min(-solution[1],Dtype(1-1e-4)));

}

template <typename Dtype>
void doLocalMatchRefinement(const Dtype vertA[3],
                            const Dtype * vertsB,
                            const int width,
                            const int height,
                            const int2 initB,
                            Dtype & uB,
                            Dtype & vB,
                            const float connectionThreshDistSquared=0.01*0.01) {

    const Dtype vertB[3] = {
        vertsB[initB.x + width*(initB.y + height*0)],
        vertsB[initB.x + width*(initB.y + height*1)],
        vertsB[initB.x + width*(initB.y + height*2)]
    };
    if (std::isnan(vertB[0])) {
        uB = vB = -1;
    }

    unsigned char connections = 0x00;

    const int2 offsets[8] = { make_int2(-1,-1), make_int2(-1,0), make_int2(-1,1),make_int2(0,1),
                              make_int2(1,1), make_int2(1,0), make_int2(1,-1), make_int2(0,-1) };

    for (int i=0; i<8; ++i) {
        const int2 pt = make_int2(initB.x + offsets[i].x,initB.y + offsets[i].y);
        if (pt.x < 0 || pt.x >= width - 1 || pt.y < 0 || pt.y >= height - 1) {
            continue;
        } else {
            if (std::isnan(vertsB[pt.x + width*(pt.y + height*0)])) {
                continue;
            }
            Dtype dist = 0;
            for (int c=0; c<3; ++c) {
                const Dtype diff = vertA[c] - vertsB[pt.x + width*(pt.y + height*c)];
                dist += diff*diff;
            }
            if (dist < connectionThreshDistSquared) {
                connections |= (1<<i);
            }
        }
    }

    const int maskDownLeft = 1 << 0;
    const int maskLeft = 1 << 1;
    const int maskUpLeft = 1 << 2;
    const int maskUp = 1 << 3;
    const int maskUpRight = 1 << 4;
    const int maskRight = 1 << 5;
    const int maskDownRight = 1 << 6;
    const int maskDown = 1 << 7;

    Dtype gradLeft = 0, gradRight = 0, gradUp = 0, gradDown = 0;

    const Dtype vertBMinusVertA[3] = {
        vertB[0] - vertA[0],
        vertB[1] - vertA[1],
        vertB[2] - vertA[2]
    };

    Dtype vertLeft[3];
    Dtype vertRight[3];
    Dtype vertUp[3];
    Dtype vertDown[3];

    if (connections & maskLeft) {
        for (int c=0; c<3; ++c) {
            vertLeft[c] = vertsB[initB.x-1 + width*(initB.y + height*c)];
            gradLeft += vertBMinusVertA[c]*(vertLeft[c] - vertB[c]);
        }
    }
    if (connections & maskRight) {
        for (int c=0; c<3; ++c) {
            vertRight[c] = vertsB[initB.x+1 + width*(initB.y + height*c)];
            gradRight += vertBMinusVertA[c]*(vertRight[c] - vertB[c]);
        }
    }
    if (connections & maskUp) {
        for (int c=0; c<3; ++c) {
            vertUp[c] = vertsB[initB.x + width*(initB.y+1 + height*c)];
            gradUp += vertBMinusVertA[c]*(vertUp[c] - vertB[c]);
        }
    }
    if (connections & maskDown) {
        for (int c=0; c<3; ++c) {
            vertDown[c] = vertsB[initB.x + width*(initB.y-1 + height*c)];
            gradDown += vertBMinusVertA[c]*(vertDown[c] - vertB[c]);
        }
    }

    int direction = 0;
    if (gradLeft < 0) {
        if (gradUp < 0) {
            if (connections & maskUpLeft) {
                direction = maskUpLeft;
            } else if (gradLeft < gradUp) {
                direction = maskLeft;
            } else {
                direction = maskUp;
            }
        } else if (gradDown < 0) {
            if (connections & maskDownLeft) {
                direction = maskDownLeft;
            } else if (gradLeft < gradDown) {
                direction = maskLeft;
            } else {
                direction = maskDown;
            }
        } else {
            direction = maskLeft;
        }
    } else if (gradRight < 0) {
        if (gradUp < 0) {
            if (connections & maskUpRight) {
                direction = maskUpRight;
            } else if (gradRight < gradUp) {
                direction = maskRight;
            } else {
                direction = maskUp;
            }
        } else if (gradDown < 0) {
            if (connections & maskDownRight) {
                direction = maskDownRight;
            } else if (gradRight < gradDown) {
                direction = maskRight;
            } else {
                direction = maskDown;
            }
        } else {
            direction = maskRight;
        }
    } else if (gradUp < 0) {
        direction = maskUp;
    } else if (gradDown < 0) {
        direction = maskDown;
    } else {
        // go nowhere
    }

    switch (direction) {
    case maskLeft:
    {
        Dtype t;
        getOptimalInterpolation1D(vertA,vertB,vertLeft,t);
        uB = initB.x - t;
        vB = initB.y;
    }
       break;
    case maskRight:
    {
        Dtype t;
        getOptimalInterpolation1D(vertA,vertB,vertRight,t);
        uB = initB.x + t;
        vB = initB.y;
    }
        break;
    case maskUp:
    {
        Dtype t;
        getOptimalInterpolation1D(vertA,vertB,vertUp,t);
        uB = initB.x;
        vB = initB.y + t;
    }
        break;
    case maskDown:
    {
        Dtype t;
        getOptimalInterpolation1D(vertA,vertB,vertDown,t);
        uB = initB.x;
        vB = initB.y - t;
    }
        break;
    case maskUpLeft:
    {
        Dtype vertUpLeft[3];
        for (int c=0; c<3; ++c) {
            vertUpLeft[c] = vertsB[initB.x-1 + width*(initB.y+1 + height*c)];
        }
        Dtype t[2];
        getOptimalInterpolation2D(vertA,vertB,vertLeft,vertUp,vertUpLeft,t[0],t[1]);
        uB = initB.x - t[0];
        vB = initB.y + t[1];
    }
        break;
    case maskDownLeft:
    {
        Dtype vertDownLeft[3];
        for (int c=0; c<3; ++c) {
            vertDownLeft[c] = vertsB[initB.x-1 + width*(initB.y-1 + height*c)];
        }
        Dtype t[2];
        getOptimalInterpolation2D(vertA,vertB,vertLeft,vertDown,vertDownLeft,t[0],t[1]);
        uB = initB.x - t[0];
        vB = initB.y - t[1];
    }
        break;
    case maskDownRight:
    {
        Dtype vertDownRight[3];
        for (int c=0; c<3; ++c) {
            vertDownRight[c] = vertsB[initB.x+1 + width*(initB.y-1 + height*c)];
        }
        Dtype t[2];
        getOptimalInterpolation2D(vertA,vertB,vertRight,vertDown,vertDownRight,t[0],t[1]);
        uB = initB.x + t[0];
        vB = initB.y - t[1];
    }
        break;
    case maskUpRight:
    {
        Dtype vertUpRight[3];
        for (int c=0; c<3; ++c) {
            vertUpRight[c] = vertsB[initB.x+1 + width*(initB.y+1 + height*c)];
        }
        Dtype t[2];
        getOptimalInterpolation2D(vertA,vertB,vertRight,vertUp,vertUpRight,t[0],t[1]);
        uB = initB.x + t[0];
        vB = initB.y + t[1];
    }
        break;
    default:
        uB = initB.x;
        vB = initB.y;
        break;
    }

}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                      match finding
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype>
class RigidMatchFinder {
public:

    static const bool NeedsLocalRefinement = false;
    static const bool EnablesMatchless = true;

    explicit RigidMatchFinder(const TransformationMatrix<Dtype> & transformationGlobalToB,
                              const Dtype focalLengthX,
                              const Dtype focalLengthY,
                              const Dtype principalPointX,
                              const Dtype principalPointY)
        : transformationGlobalToB_(transformationGlobalToB),
          focalLengthX_(focalLengthX), focalLengthY_(focalLengthY),
          principalPointX_(principalPointX), principalPointY_(principalPointY) { }

    inline bool findMatch(const Dtype ptA[3],
                          Dtype & uB,
                          Dtype & vB) {

        Point<Dtype,3> _ptA(ptA,1);
        Dtype ptBLocal[3];
        Point<Dtype,3> _ptBLocal(ptBLocal,1);
        const_cast<TransformationMatrix<Dtype> &>(transformationGlobalToB_).apply(_ptA,_ptBLocal);

        if (ptBLocal[2] < 0) {
            return false;
        }

        uB = ptBLocal[0]*focalLengthX_/ptBLocal[2] + principalPointX_;
        vB = ptBLocal[1]*focalLengthY_/ptBLocal[2] + principalPointY_;

        return true;
    }

private:

    const TransformationMatrix<Dtype> & transformationGlobalToB_;
    const Dtype focalLengthX_;
    const Dtype focalLengthY_;
    const Dtype principalPointX_;
    const Dtype principalPointY_;

};

template <typename Dtype>
class FLANNMatchFinder {
public:

    static const bool NeedsLocalRefinement = true;
    static const bool EnablesMatchless = false;

    explicit FLANNMatchFinder(const Dtype * vertMap, const int width, const int height)
        : cloud_(vertMap,width,height), tree_(3, cloud_, nanoflann::KDTreeSingleIndexAdaptorParams(20 /* max leaf */)) {
        tree_.buildIndex();
    }

    inline bool findMatch(const Dtype ptA[3],
                          Dtype & uB,
                          Dtype & vB) {

        if (cloud_.kdtree_get_point_count() == 0) {
            return false;
        }

        Dtype distance;
        int index;
        tree_.knnSearch(ptA,1,&index,&distance);
        if (index >= 0 && index < cloud_.kdtree_get_point_count()) {
            uB = cloud_.pixIndex(index).x;
            vB = cloud_.pixIndex(index).y;
            return distance < 0.01*0.01;
        } else {
            return false;
        }
    }

private:

    struct KDPointCloud {
    public:

        explicit KDPointCloud(const Dtype * vertMap, const int width, const int height)
            : vertMap_(vertMap), width_(width), height_(height) {
            pixIndices_.reserve(0.5*width*height);
            for (int y=0; y<height; ++y) {
                for (int x=0; x<width; ++x) {
                    if (!std::isnan(vertMap[x + width*(y + height*0)])) {
                        pixIndices_.push_back(make_int2(x,y));
                    }
                }
            }
        }

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const {
            return pixIndices_.size();
        }

        // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
        inline Dtype kdtree_distance(const Dtype * p1, const size_t idx_p2,size_t /*size*/) const {
            const int i = pixIndices_[idx_p2].x + width_*pixIndices_[idx_p2].y;
            const Dtype d0 = p1[0] - vertMap_[i + 0*width_*height_];
            const Dtype d1 = p1[1] - vertMap_[i + 1*width_*height_];
            const Dtype d2 = p1[2] - vertMap_[i + 2*width_*height_];
            return d0*d0 + d1*d1 + d2*d2;
        }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline Dtype kdtree_get_pt(const size_t idx, int dim) const {
            const int i = pixIndices_[idx].x + width_*(pixIndices_[idx].y + height_*dim);
            return vertMap_[i];
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }

        inline int2 pixIndex(const int index) const { return pixIndices_[index]; }

    private:

        const Dtype * vertMap_;
        const int width_;
        const int height_;

        std::vector<int2> pixIndices_;

    };

    typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<Dtype, KDPointCloud>, KDPointCloud, 3, int> KDTree;

    KDPointCloud cloud_;
    KDTree tree_;

};

template <typename Dtype>
class RandomMatchFinder {
public:

    static const bool NeedsLocalRefinement = false;
    static const bool EnablesMatchless = false;

    explicit RandomMatchFinder(const Dtype * vertMap, const int width, const int height)
        : vertMap_(vertMap), width_(width), height_(height) { }

    inline bool findMatch(const Dtype ptA[3],
                          Dtype & uB,
                          Dtype & vB) {

        boost::uniform_int<> uDistribution(0,width_-1);
        boost::uniform_int<> vDistribution(0,height_-1);

        Dtype ptB[3];

        for (int attempt = 0; attempt < maxAttempts; ++attempt) {

            int uBi = uDistribution(*caffe_rng());
            int vBi = vDistribution(*caffe_rng());

            ptB[0] = vertMap_[uBi + width_*vBi];
            ptB[1] = vertMap_[uBi + width_*vBi + width_*height_];
            ptB[2] = vertMap_[uBi + width_*vBi + 2*width_*height_];

            const Dtype distSquared = ((ptA[0]-ptB[0]) * (ptA[0]-ptB[0])) +
                                      ((ptA[1]-ptB[1]) * (ptA[1]-ptB[1])) +
                                      ((ptA[2]-ptB[2]) * (ptA[2]-ptB[2]));

            if (distSquared < matchThresholdSquared) {
                uB = uBi;
                vB = vBi;
                return true;
            }

        }

        return false;

    }

private:

    const Dtype * vertMap_;
    const int width_, height_;

    static const int maxAttempts = 100000;
    static const Dtype matchThreshold = 0.005;
    static const Dtype matchThresholdSquared = 0.000025;

};

template <typename Dtype,
          template <typename> class MatchFinderT>
inline bool checkForMatch(const int uA, const int vA,
                          const Dtype * vertsA,
                          const Dtype * vertsB,
                          const int width, const int height,
                          MatchFinderT<Dtype> & matchFinder,
                          Dtype ptA[3],
                          Dtype & uB,
                          Dtype & vB,
                          bool & vertAValid) {

    for (int c=0; c<3; ++c) {
        ptA[c] = vertsA[uA + width*(vA + height*c)];
    }

    // make sure A is valid
    if (std::isnan(ptA[0])) {
        //std::cout << "invalid A" << std::endl;
        return false;
    }

    vertAValid = true;

    // transform the global point at (uA,vA) into B's camera
    if (!matchFinder.findMatch(ptA,uB,vB)) {
        return false;
    }

    // check if point is within interpolation bounds
    if (uB < 0 || uB >= width-2 || vB < 0 || vB >= height-2) {
        //std::cout << "out of frame" << std::endl;
        return false;
    }

    if (MatchFinderT<Dtype>::NeedsLocalRefinement) {

        doLocalMatchRefinement(ptA,vertsB,width,height,
                               make_int2(uB,vB),uB,vB);

    }

    assert(uB >= 0);
    assert(uB < width-2);
    assert(vB >= 0);
    assert(vB < height-2);
    if (uB < 0 || uB >= width-2 || vB < 0 || vB >= height-2) {
        std::cout << "out of frame" << std::endl;
        return false;
    }

    // get global vertex a (uB,vB)
    Dtype ptB[3];
    interpolateVertex(vertsB,width,height,uB,vB,ptB);

    // make sure B is valid
    if (std::isnan(ptB[0])) {
        //std::cout << "invalid B" << std::endl;
        return false;
    }

    const Dtype distAB = (ptA[0]-ptB[0])*(ptA[0]-ptB[0]) +
                         (ptA[1]-ptB[1])*(ptA[1]-ptB[1]) +
                         (ptA[2]-ptB[2])*(ptA[2]-ptB[2]);

    // make sure they're the same point
    if ( distAB > 0.005*0.005) {
        //std::cout << "too far" << std::endl;
        return false;
    }

//    std::cout << "accepting " << distAB << std::endl;
//    std::cout << ptA[0] << ", " << ptA[1] << ", " << ptA[2] << std::endl;
//    std::cout << ptB[0] << ", " << ptB[1] << ", " << ptB[2] << std::endl;

    return true;

}




//template <typename Dtype>
//inline bool checkForMatch(const int uA, const int vA,
//                          const Dtype * vertsA,
//                          const Dtype * vertsB,
//                          const int width, const int height,
//                          const TransformationMatrix<Dtype> & transformationGlobalToB,
//                          const Dtype focalLengthX, const Dtype focalLengthY,
//                          const Dtype principalPointX, const Dtype principalPointY,
//                          Dtype ptA[3],
//                          Dtype & uB,
//                          Dtype & vB,
//                          bool & vertAValid) {

//    for (int c=0; c<3; ++c) {
//        ptA[c] = vertsA[uA + width*(vA + height*c)];
//    }

//    // make sure A is valid
//    if (std::isnan(ptA[0])) {
//        //std::cout << "invalid A" << std::endl;
//        return false;
//    }

//    vertAValid = true;

//    // transform the global point at (uA,vA) into B's camera
//    Point<Dtype,3> _ptA(ptA,1);
//    Dtype ptBLocal[3];
//    Point<Dtype,3> _ptBLocal(ptBLocal,1);
//    const_cast<TransformationMatrix<Dtype> &>(transformationGlobalToB).apply(_ptA,_ptBLocal);

//    // ensure point is in front of camera
//    if (ptBLocal[2] < 0) {
//        //std::cout << "negative z" << std::endl;
//        return false;
//    }

//    // project point
//    uB = ptBLocal[0]*focalLengthX/ptBLocal[2] + principalPointX;
//    vB = ptBLocal[1]*focalLengthY/ptBLocal[2] + principalPointY;

//    // check if point is within interpolation bounds
//    if (uB < 0 || uB >= width-2 || vB < 0 || vB >= height-2) {
//        //std::cout << "out of frame" << std::endl;
//        return false;
//    }

//    // get global vertex a (uB,vB)
//    Dtype ptB[3];
//    interpolateVertex(vertsB,width,height,uB,vB,ptB);

//    // make sure B is valid
//    if (std::isnan(ptB[0])) {
//        //std::cout << "invalid B" << std::endl;
//        return false;
//    }

//    const Dtype distAB = (ptA[0]-ptB[0])*(ptA[0]-ptB[0]) +
//                         (ptA[1]-ptB[1])*(ptA[1]-ptB[1]) +
//                         (ptA[2]-ptB[2])*(ptA[2]-ptB[2]);

//    // make sure they're the same point
//    if ( distAB > 0.005) {
//        //std::cout << "too far" << std::endl;
//        return false;
//    }

//    return true;

//}

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                 positive match selection
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype>
class AllPositiveMatchesSelector {
public:

    explicit AllPositiveMatchesSelector(const int width, const int height)
        : width_(width), height_(height) { }

    inline void init() {
        x_ = 0;
        y_ = 0;
    }

    inline int totalPossibleMatches() const {
        return width_*height_;
    }


    template <template <typename> class MatchFinderT>
    bool getNextMatch(int & uA, int & vA,
                      Dtype & uB, Dtype & vB,
                      Dtype ptA[3],
                      const Dtype * vertsA,
                      const Dtype * vertsB,
                      const int width,
                      const int height,
                      MatchFinderT<Dtype> & matchFinder,
                      const bool allowMatchless,
                      bool & vertAValid) {

        // -=-=-=-=-=-=-=-=-=-=- check every vertex for match -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        bool foundMatch = false;
        do {

            uA = x_;
            vA = y_;

            vertAValid = false;
            //std::cout << "checking " << uA << ", " << vA << "...";
            foundMatch = checkForMatch(uA,vA,
                                       vertsA, vertsB,
                                       width, height,
                                       matchFinder,
                                       ptA,uB,vB,
                                       vertAValid);
            //std::cout << foundMatch << std::endl;

            if (allowMatchless) {
                if (!foundMatch && vertAValid) {
                    uB = vB = -1;
                    foundMatch = true;
                }
            }

            ++x_;
            if (x_ == width) {
                x_ = 0;
                ++y_;
                if (y_ == height) {
                    break;
                }
            }

        } while (!foundMatch);

        return foundMatch;

    }

private:

    int x_;
    int y_;

    const int width_;
    const int height_;

};

template <typename Dtype>
class RandomPositiveMatchesSelector {
public:

    explicit RandomPositiveMatchesSelector(const int K) : K_(K) { }

    inline void init() {
        sampleAttempts_ = 0;
        k_ = 0;
    }

    inline int totalPossibleMatches() const {
        return K_;
    }

    template <template <typename> class MatchFinderT>
    bool getNextMatch(int & uA, int & vA,
                      Dtype & uB, Dtype & vB,
                      Dtype ptA[3],
                      const Dtype * vertsA,
                      const Dtype * vertsB,
                      const int width,
                      const int height,
                      MatchFinderT<Dtype> & matchFinder,
                      const bool allowMatchless,
                      bool & vertAValid) {

        // -=-=-=-=-=-=-=-=-=-=- check random vertices for matches -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        boost::uniform_int<> uDistribution(0,width-1);
        boost::uniform_int<> vDistribution(0,height-1);

//        std::cout << k_ << "..." << std::endl;

        bool foundMatch = false;
        do {

            if (k_ >= K_) {
                std::cout << "reached the end" << std::endl;
                return false;
            }

            ++sampleAttempts_;
            if (sampleAttempts_ == width*height) {
                std::cout << "tried " << width*height << " times, i'm out" << std::endl;
                break;
            }

            uA = uDistribution(*caffe_rng());
            vA = vDistribution(*caffe_rng());

//            std::cout << "trying " << uA << ", " << vA << std::endl;

            vertAValid = false;
//            std::cout << "checking " << uA << ", " << vA << "...";
            foundMatch = checkForMatch(uA,vA,
                                       vertsA, vertsB,
                                       width, height,
                                       matchFinder,
                                       ptA,uB,vB,
                                       vertAValid);
//            std::cout << foundMatch << std::endl;

            if (allowMatchless) {
                if (!foundMatch && vertAValid) {
                    uB = vB = -1;
                    foundMatch = true;
                }
            }

        } while (!foundMatch);

        ++k_;

        return foundMatch;

    }

private:

    int k_;
    int sampleAttempts_;

    const int K_;

};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                 negative match selection
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype>
class AllNegativesSelector {
public:

    explicit AllNegativesSelector(const int width, const int height) :
        width_(width), height_(height) { }

    inline void initFrame(const Dtype * repB,
                          const int width,
                          const int height,
                          const int repChannels) { }

    inline void initPoint(const Dtype * representationA) {
        x_ = 0;
        y_ = 0;
    }

    inline int totalPossibleMatches() const {
        return width_*height_;
    }

    bool getNextMatch(int & uBNeg, int & vBNeg,
                      const int uBPos, const int vBPos,
                      const Dtype * repB,
                      const int width, const int height,
                      const int repChannels,
                      const Dtype ignoreMarginSquared,
                      const Dtype * representationA,
                      std::vector<Dtype> & representationB,
                      Dtype * diffAB) {

        // -=-=-=-=-=-=-=-=-=-=- check every vertex for match -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        bool foundMatch = false;
        do {

            uBNeg = x_;
            vBNeg = y_;

            if ( (uBNeg-uBPos)*(uBNeg-uBPos) + (vBNeg-vBPos)*(vBNeg-vBPos) > 2*2) {

                for (int c=0; c<repChannels; ++c) {
                    representationB[c] = repB[uBNeg + width*(vBNeg + height*c)];
                }
                caffe_sub(repChannels,representationA,representationB.data(),diffAB);

                if (std::isfinite(ignoreMarginSquared)) {

                    const Dtype distSquared = caffe_cpu_dot(repChannels,diffAB,diffAB);
                    if (distSquared < ignoreMarginSquared) {
                        foundMatch = true;
                    }

                } else {
                    foundMatch = true;
                }
            }

            ++x_;
            if (x_ == width) {
                x_ = 0;
                ++y_;
                if (y_ == height) {
                    break;
                }
            }

        } while (!foundMatch);

        return foundMatch;

    }

private:

    int x_;
    int y_;

    const int width_;
    const int height_;

};

template <typename Dtype>
class RandomNegativesSelector {
public:

    explicit RandomNegativesSelector(const int K) : K_(K) { }


    inline void initFrame(const Dtype * repB,
                          const int width,
                          const int height,
                          const int repChannels) { }

    inline void initPoint(const Dtype * representationA) {

        sampleAttempts_ = 0;
        k_ = 0;
    }

    inline int totalPossibleMatches() const {
        return K_;
    }

    bool getNextMatch(int & uBNeg, int & vBNeg,
                      const int uBPos, const int vBPos,
                      const Dtype * repB,
                      const int width, const int height,
                      const int repChannels,
                      const Dtype ignoreMarginSquared,
                      const Dtype * representationA,
                      std::vector<Dtype> & representationB,
                      Dtype * diffAB) {

        boost::uniform_int<> uDistribution(0,width-1);
        boost::uniform_int<> vDistribution(0,height-1);

        bool foundMatch = false;
        do {

            if (k_ == K_) {
                break;
            }

            ++sampleAttempts_;
            if (sampleAttempts_ == width*height) {
                break;
            }

            uBNeg = uDistribution(*caffe_rng());
            vBNeg = vDistribution(*caffe_rng());

            if ( (uBNeg-uBPos)*(uBNeg-uBPos) + (vBNeg-vBPos)*(vBNeg-vBPos) > 2*2) {

                for (int c=0; c<repChannels; ++c) {
                    representationB[c] = repB[uBNeg + width*(vBNeg + height*c)];
                }
                caffe_sub(repChannels,representationA,representationB.data(),diffAB);

                if (std::isfinite(ignoreMarginSquared)) {

                    const Dtype distSquared = caffe_cpu_dot(repChannels,diffAB,diffAB);
                    if (distSquared < ignoreMarginSquared) {
                        foundMatch = true;
                    } else {
                        // by advancing k, the learning will behave as if this patch
                        // was picked but had no effect
                        ++k_;
                    }

                } else {
                    foundMatch = true;
                }
            }

        } while (!foundMatch);

        ++k_;

        return foundMatch;

    }

private:

    int sampleAttempts_;
    int k_;

    const int K_;

};

//template <typename Dtype>
//class HardNegativesSelector {
//public:

//    // K -- how many to select
//    // M -- how many negatives to mine before picking the top K
//    explicit HardNegativesSelector(const int K,
//                                   const int M) : K_(K), M_(M) { }

//    inline void init(const int uA, const int vA,
//                     const Dtype * repB,
//                     const int width,
//                     const int height,
//                     const int repChannels,
//                     const Dtype ignoreMarginSquared,
//                     const std::vector<Dtype> & representationA) {

//        minedNegatives_.clear();

//        boost::uniform_int<> uDistribution(0,width-1);
//        boost::uniform_int<> vDistribution(0,height-1);

//        std::vector<Dtype> representationB(repChannels);
//        std::vector<Dtype> diffAB(repChannels);

//        for (int m=0; m<M_; ++m) {

//            const int uB = uDistribution(*caffe_rng());
//            const int vB = vDistribution(*caffe_rng());

//            if ( (uB-uA)*(uB-uA) + (vB-vA)*(vB-vA) > 2*2) {

//                for (int c=0; c<repChannels; ++c) {
//                    representationB[c] = repB[uB + width*(vB + height*c)];
//                }
//                caffe_sub(repChannels,representationA.data(),representationB.data(),diffAB.data());

//                const Dtype distSquared = caffe_cpu_dot(repChannels,diffAB.data(),diffAB.data());
//                if (distSquared < ignoreMarginSquared) {

//                    minedNegatives_.insert(typename NegMap::value_type(distSquared,std::pair<int,int>(uB,vB)));

//                }

//            }

//        }

//        k_ = 0;
//        minedIterator_ = minedNegatives_.begin();

//        std::cout << "mined " << minedNegatives_.size() << " (" << K_ << " / " << M_ << " )" << std::endl;

//    }

//    inline int totalPossibleMatches() const {
//        return K_;
//    }

//    bool getNextMatch(int & uB, int & vB,
//                      const int uA, const int vA,
//                      const Dtype * repB,
//                      const int width, const int height,
//                      const int repChannels,
//                      const Dtype ignoreMarginSquared,
//                      const std::vector<Dtype> & representationA,
//                      std::vector<Dtype> & representationB,
//                      Dtype * diffAB) {

//        // return if we've exhausted our store of mined negatives
//        if (minedIterator_ == minedNegatives_.end()) {
//            return false;
//        }

//        // return if we've returned the max number of samples
//        if (k_ == K_) {
//            return false;
//        }

//        uB = minedIterator_->second.first;
//        vB = minedIterator_->second.second;

//        for (int c=0; c<repChannels; ++c) {
//            representationB[c] = repB[uB + width*(vB + height*c)];
//        }
//        caffe_sub(repChannels,representationA.data(),representationB.data(),diffAB);

//        ++minedIterator_;
//        ++k_;

//        return true;

//    }

//private:

//    typedef std::multimap<Dtype,std::pair<int,int> > NegMap;

//    NegMap minedNegatives_;
//    typename NegMap::iterator minedIterator_;

//    int k_;

//    const int K_;
//    const int M_;

//};

template <typename Dtype>
class HardNegativesSelector {
public:

    // K -- how many to select
    // M -- how many negatives to mine before picking the top K
    explicit HardNegativesSelector(const int K) : K_(K), matchIndices_(K), matchDistances_(K) { }

    inline void initFrame(const Dtype * repB,
                          const int width,
                          const int height,
                          const int repChannels) {

        std::cout << "building tree" << std::endl;
        embeddingCloud_.reset(new KDEmbeddingCloud(repB,width,height,repChannels));
        embeddingTree_.reset(new KDTree(repChannels, *embeddingCloud_, nanoflann::KDTreeSingleIndexAdaptorParams(20 /* max leaf */)));
        embeddingTree_->buildIndex();
        std::cout << "built" << std::endl;

    }

    inline void initPoint(const Dtype * representationA) {

        embeddingTree_->knnSearch(representationA,
                                  K_,
                                  matchIndices_.data(),
                                  matchDistances_.data());
        k_ = 0;

    }

    inline int totalPossibleMatches() const {
        return K_;
    }

    bool getNextMatch(int & uBNeg, int & vBNeg,
                      const int uBPos, const int vBPos,
                      const Dtype * repB,
                      const int width, const int height,
                      const int repChannels,
                      const Dtype ignoreMarginSquared,
                      const Dtype * representationA,
                      std::vector<Dtype> & representationB,
                      Dtype * diffAB) {

        bool matchFound = false;

        while (!matchFound) {

            // return if we've returned the max number of samples
            if (k_ == K_) {
                break;
            }

            // return if we've passed the ignore margin
            if (matchDistances_[k_] > ignoreMarginSquared) {
                break;
            }

            const int index = matchIndices_[k_];
            uBNeg = index % width;
            vBNeg = index / width;

            if ( (uBNeg-uBPos)*(uBNeg-uBPos) + (vBNeg-vBPos)*(vBNeg-vBPos) > 2*2) {

                matchFound = true;

                for (int c=0; c<repChannels; ++c) {
                    representationB[c] = repB[uBNeg + width*(vBNeg + height*c)];
                }
                caffe_sub(repChannels,representationA,representationB.data(),diffAB);

            }

            ++k_;
        }

        return matchFound;

    }

private:

    struct KDEmbeddingCloud {
    public:

        explicit KDEmbeddingCloud(const Dtype * embeddingMap, const int width, const int height, const int channels)
            : embedding_(embeddingMap), width_(width), height_(height), channels_(channels) { }

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const {
            return width_*height_;
        }

        // Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
        inline Dtype kdtree_distance(const Dtype * p1, const size_t idx_p2,size_t /*size*/) const {
            Dtype dist(0);
            for (int c=0; c<channels_; ++c) {
                const Dtype d = p1[c] - embedding_[idx_p2 + c*width_*height_];
                dist += d*d;
            }
            return dist;
        }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline Dtype kdtree_get_pt(const size_t idx, int dim) const {
            return embedding_[idx + dim*width_*height_];
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX & /*bb*/) const { return false; }

    private:

        const Dtype * embedding_;
        const int width_;
        const int height_;
        const int channels_;

    };

    typedef nanoflann::KDTreeSingleIndexAdaptor< nanoflann::L2_Simple_Adaptor<Dtype, KDEmbeddingCloud>, KDEmbeddingCloud, -1, int> KDTree;

    boost::shared_ptr<KDEmbeddingCloud> embeddingCloud_;
    boost::shared_ptr<KDTree> embeddingTree_;

    std::vector<int> matchIndices_;
    std::vector<Dtype> matchDistances_;

    int k_;

    const int K_;

};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                        loss functors
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype>
class SquaredLossFunctor {
public:

#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline Dtype loss(const Dtype * diff, const int channels) const {
#ifdef __CUDACC__
        Dtype loss(0);
        for (int c = 0; c < channels; ++c) {
            loss += diff[c]*diff[c];
        }
        return 0.5 * loss;
#else
        return 0.5*caffe_cpu_dot(channels,diff,diff);
#endif // __CUDACC__
    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(const Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const int u, const int v,
                                  const Dtype alpha,
                                  Dtype * grad) const {
        for (int c=0; c<channels; ++c) {
            AdditionModel<Dtype>::add(grad[u + width*(v + height*c)],alpha*diff[c]);
        }
    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(const Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const Dtype u, const Dtype v,
                                  const Dtype alpha,
                                  Dtype * grad) const {
        deInterpolateGradient<Dtype,AdditionModel>(grad,width,height,channels,
                                                   u,v,diff,alpha);
    }

    inline bool hasMargin() const { return false; }

};

template <typename Dtype>
class HuberLossFunctor {
public:

    explicit HuberLossFunctor(const Dtype delta) : delta_(delta) { }

    inline Dtype loss(const Dtype * diff, const int channels) const {
//        const Dtype diffL1 = caffe_cpu_asum(channels,diff);
//        if (diffL1 < delta_) {
//            return 0.5*caffe_cpu_dot(channels,diff,diff);
//        } else {
//            return delta_*(diffL1 - 0.5*delta_);
//        }
        const Dtype squaredLoss = caffe_cpu_dot(channels,diff,diff);
        if (squaredLoss < delta_*delta_) {
            return 0.5*squaredLoss;
        }
        return delta_*(sqrtf(squaredLoss) - 0.5*delta_);
    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(const Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const int u, const int v,
                                  const Dtype alpha,
                                  Dtype * grad) const {

#ifdef __CUDACC__
        Dtype squaredLoss = 0;
        for (int c=0; c<channels; ++c) {
            squaredLoss += diff[c]*diff[c];
        }
#else
        const Dtype squaredLoss = caffe_cpu_dot(channels,diff,diff);
#endif // __CUDACC__

        if (squaredLoss < delta_*delta_) {
            for (int c=0; c<channels; ++c) {
                AdditionModel<Dtype>::add(grad[u + width*(v + height*c)],alpha*diff[c]);
            }
        } else {
            const Dtype oneOverLoss = Dtype(1)/sqrtf(squaredLoss);
            for (int c=0; c<channels; ++c) {
                AdditionModel<Dtype>::add(grad[u + width*(v + height*c)],alpha*delta_*diff[c]*oneOverLoss);
            }
        }

//#ifdef __CUDACC__
//        Dtype diffL1 = 0;
//        for (int c=0; c<channels; ++c) {
//            diffL1 += fabsf(diff[c]);
//        }
//#else
//        const Dtype diffL1 = caffe_cpu_asum(channels,diff);
//#endif // __CUDACC__
//        if (diffL1 < delta_) {
//            for (int c=0; c<channels; ++c) {
//                AdditionModel<Dtype>::add(grad[u + width*(v + height*c)],alpha*diff[c]);
//            }
//        } else {
//            for (int c=0; c<channels; ++c) {
//                AdditionModel<Dtype>::add(grad[u + width*(v + height*c)], alpha*(diff[c] > 0 ? delta_ : -delta_));
//            }
//        }
    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const Dtype u,
                                  const Dtype v,
                                  const Dtype alpha,
                                  Dtype * grad) const {

#ifdef __CUDACC__
        Dtype squaredLoss = 0;
        for (int c=0; c<channels; ++c) {
            squaredLoss += diff[c]*diff[c];
        }
#else
        const Dtype squaredLoss = caffe_cpu_dot(channels,diff,diff);
#endif // __CUDACC__

        if (squaredLoss <= delta_*delta_) {
            const Dtype oneOverLoss = Dtype(1)/sqrtf(squaredLoss);
            for (int c=0; c<channels; ++c) {
                diff[c] = delta_*diff[c]*oneOverLoss;
            }
        }
        deInterpolateGradient<Dtype,AdditionModel>(grad,width,height,channels,
                                                   u,v,diff,alpha);

//        const Dtype diffL1 = caffe_cpu_asum(channels,diff);
//        if (diffL1 < delta_) {
//            deInterpolateGradient<Dtype,AdditionModel>(grad,width,height,channels,
//                                                       u,v,diff,alpha);
//        } else {
//            //std::vector<Dtype> toDeinterpolate(channels);
//            for (int c=0; c<channels; ++c) {
//                //toDeinterpolate[c] = (diff[c] > 0 ? delta_ : -delta_); //std::copysign(delta_,diff[c]);
//                diff[c] = (diff[c] > 0 ? delta_ : -delta_);
//            }
//            deInterpolateGradient<Dtype,AdditionModel>(grad,width,height,channels,
//                                                       u,v,diff,alpha);
//        }

    }

    inline bool hasMargin() const { return false; }

private:

    const Dtype delta_;

};

template <typename Dtype>
class TukeyLossFunctor {
public:

    explicit TukeyLossFunctor(const Dtype c) : cSquared_(c*c) { }

    inline Dtype loss(const Dtype * diff, const int channels) const {

        const Dtype squaredLoss = caffe_cpu_dot(channels,diff,diff);
        if (squaredLoss < cSquared_) {
            return 0.5*squaredLoss;
        }
        return cSquared_ / 6;

    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(const Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const int u, const int v,
                                  const Dtype alpha,
                                  Dtype * grad) const {

#ifdef __CUDACC__
        Dtype squaredLoss = 0;
        for (int c=0; c<channels; ++c) {
            squaredLoss += diff[c]*diff[c];
        }
#else
        const Dtype squaredLoss = caffe_cpu_dot(channels,diff,diff);
#endif // __CUDACC__

        if (squaredLoss < cSquared_) {
            const Dtype sqrtOfGradientMultiplier = (1 - (squaredLoss / cSquared_));
            const Dtype gradientMultiplier = sqrtOfGradientMultiplier*sqrtOfGradientMultiplier;
            for (int c = 0; c < channels; ++c) {
                AdditionModel<Dtype>::add(grad[u + width*(v + height*c)],alpha*gradientMultiplier*diff[c]);
            }
        }
        // gradient is 0 otherwise

    }


    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const Dtype u,
                                  const Dtype v,
                                  const Dtype alpha,
                                  Dtype * grad) const {

#ifdef __CUDACC__
        Dtype squaredLoss = 0;
        for (int c=0; c<channels; ++c) {
            squaredLoss += diff[c]*diff[c];
        }
#else
        const Dtype squaredLoss = caffe_cpu_dot(channels,diff,diff);
#endif // __CUDACC__

        if (squaredLoss < cSquared_) {
            const Dtype sqrtOfGradientMultiplier = (1 - (squaredLoss / cSquared_));
            const Dtype gradientMultiplier = sqrtOfGradientMultiplier*sqrtOfGradientMultiplier;
            for (int c = 0; c < channels; ++c) {
                diff[c] *= gradientMultiplier;
            }
            deInterpolateGradient<Dtype,AdditionModel>(grad,width,height,channels,
                                                       u,v,diff,alpha);
        }
        // gradient is zero otherwise

    }

private:

    Dtype cSquared_;

};

template <typename Dtype>
class HingeLossFunctor {
public:

    explicit HingeLossFunctor(const Dtype margin) : margin_(margin) { }

    inline Dtype loss(const Dtype * diff, const int channels) {
        const Dtype distSquared = caffe_cpu_dot(channels,diff,diff);
        const Dtype mdist = std::max<Dtype>(margin_ - std::sqrt(distSquared),Dtype(0));
        return 0.5*mdist*mdist;
    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(const Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const int u, const int v,
                                  const Dtype alpha,
                                  Dtype * grad) const {
#ifdef __CUDACC__
        Dtype distSquared = 0;
        for (int c=0; c<channels; ++c) {
            distSquared += diff[c]*diff[c];
        }
#else
        const Dtype distSquared = caffe_cpu_dot(channels,diff,diff);
#endif // __CUDACC__
        const Dtype dist = std::sqrt(distSquared);
        const Dtype mdist = margin_ - dist;
        const Dtype beta = -alpha * mdist / (dist + Dtype(1e-4));

        if (mdist > Dtype(0)) {
            for (int c=0; c<channels; ++c) {
                AdditionModel<Dtype>::add(grad[u + width*(v + height*c)], beta*diff[c]);
            }
        }
    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(const Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const Dtype u, const Dtype v,
                                  const Dtype alpha,
                                  Dtype * grad) const {
#ifdef __CUDACC__
        Dtype distSquared = 0;
        for (int c=0; c<channels; ++c) {
            distSquared += diff[c]*diff[c];
        }
#else
        const Dtype distSquared = caffe_cpu_dot(channels,diff,diff);
#endif // __CUDACC__
        const Dtype dist = std::sqrt(distSquared);
        const Dtype mdist = margin_ - dist;
        const Dtype beta = -alpha * mdist / (dist + Dtype(1e-4));

        if (mdist > Dtype(0)) {
            deInterpolateGradient<Dtype,AdditionModel>(grad,width,height,channels,
                                                       u,v,diff,beta);
        }
    }

    inline bool hasMargin() const { return true; }

    inline Dtype margin() const { return margin_; }

private:

    const Dtype margin_;

};

template <typename Dtype>
class HuberHingeLossFunctor {
public:

    explicit HuberHingeLossFunctor(const Dtype margin, const Dtype delta) : margin_(margin), delta_(delta) { }

    inline Dtype loss(const Dtype * diff, const int channels) {
        const Dtype distSquared = caffe_cpu_dot(channels,diff,diff);
        const Dtype mdist = std::max<Dtype>(margin_ - std::sqrt(distSquared),Dtype(0));
        if (mdist < delta_) {
            return 0.5*mdist*mdist;
        } else {
            return delta_*(mdist - 0.5*delta_);
        }
    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(const Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const int u, const int v,
                                  const Dtype alpha,
                                  Dtype * grad) const {
#ifdef __CUDACC__
        Dtype distSquared = 0;
        for (int c=0; c<channels; ++c) {
            distSquared += diff[c]*diff[c];
        }
#else
        const Dtype distSquared = caffe_cpu_dot(channels,diff,diff);
#endif // __CUDACC__
        const Dtype dist = std::sqrt(distSquared);
        const Dtype mdist = margin_ - dist;

        if (mdist > Dtype(0)) {

            Dtype beta;
            if (mdist < delta_) {
                beta = -alpha * mdist / (dist + Dtype(1e-5));
            } else {
                beta = -alpha * delta_ / (dist + Dtype(1e-5));
            }

            for (int c=0; c<channels; ++c) {
                AdditionModel<Dtype>::add(grad[u + width*(v + height*c)], beta*diff[c]);
            }

        }
    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const Dtype u, const Dtype v,
                                  const Dtype alpha,
                                  Dtype * grad) const {
#ifdef __CUDACC__
        Dtype distSquared = 0;
        for (int c=0; c<channels; ++c) {
            distSquared += diff[c]*diff[c];
        }
#else
        const Dtype distSquared = caffe_cpu_dot(channels,diff,diff);
#endif // __CUDACC__
        const Dtype dist = std::sqrt(distSquared);
        const Dtype mdist = margin_ - dist;

        if (mdist > Dtype(0)) {

            Dtype beta;
            if (mdist < delta_) {
                beta = -alpha * mdist / (dist + Dtype(1e-5));
            } else {
                beta = -alpha * delta_ / (dist + Dtype(1e-5));
            }

            deInterpolateGradient<Dtype,AdditionModel>(grad,width,height,channels,
                                                       u,v,diff,beta);
        }
    }

    inline bool hasMargin() const { return true; }

    inline Dtype margin() const { return margin_; }

private:

    const Dtype margin_;
    const Dtype delta_;

};

template <typename Dtype>
class NegativeExponentialLossFunctor {
public:

    explicit NegativeExponentialLossFunctor(const Dtype sigma) : sigma_(sigma) { }

    inline Dtype loss(const Dtype * diff, const int channels) {
        const Dtype distSquared = caffe_cpu_dot(channels,diff,diff);
        return std::exp(-sigma_*0.5*distSquared);
    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(const Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const int u, const int v,
                                  const Dtype alpha,
                                  Dtype * grad) const {
#ifdef __CUDACC__
        Dtype distSquared = 0;
        for (int c=0; c<channels; ++c) {
            distSquared += diff[c]*diff[c];
        }
#else
        const Dtype distSquared = caffe_cpu_dot(channels,diff,diff);
#endif // __CUDACC__
        const Dtype beta = -alpha*(sigma_*std::exp(-sigma_*0.5*distSquared));

        for (int c=0; c<channels; ++c) {
            AdditionModel<Dtype>::add(grad[u + width*(v + height*c)], beta*diff[c]);
        }
    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void differentiateLoss(const Dtype * diff,
                                  const int width,
                                  const int height,
                                  const int channels,
                                  const Dtype u, const Dtype v,
                                  const Dtype alpha,
                                  Dtype * grad) const {
#ifdef __CUDACC__
        Dtype distSquared = 0;
        for (int c=0; c<channels; ++c) {
            distSquared += diff[c]*diff[c];
        }
#else
        const Dtype distSquared = caffe_cpu_dot(channels,diff,diff);
#endif // __CUDACC__
        const Dtype beta = -alpha*(sigma_*std::exp(-sigma_*0.5*distSquared));

        deInterpolateGradient<Dtype,AdditionModel>(grad,width,height,channels,
                                                   u,v,diff,beta);
    }

    inline bool hasMargin() const { return false; }

    inline Dtype margin() const { return std::numeric_limits<Dtype>::infinity(); }

private:

    const Dtype sigma_;

};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                    loss balancing functors
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype>
class LossBalancingFunctor {
public:

    inline Dtype posAlpha() const { return posAlpha_; }

    inline Dtype negAlpha() const { return negAlpha_; }

protected:

    inline Dtype superBalance(const Dtype posLoss, const Dtype negLoss) {
        return posLoss*posAlpha_ + negLoss*negAlpha_;
    }

    Dtype posAlpha_;
    Dtype negAlpha_;
};

template <typename Dtype>
class NormalizeByNumPositivesFunctor : public LossBalancingFunctor<Dtype> {
public:
    inline Dtype balance(const Dtype posLoss, const int nPositives,
                         const Dtype negLoss, const int nNegatives) {
        this->posAlpha_ = this->negAlpha_ = (nPositives == 0 ? 0 : Dtype(1)/nPositives);
        return this->superBalance(posLoss,negLoss);
    }
};

template <typename Dtype>
class NormalizeTotalFunctor : public LossBalancingFunctor<Dtype> {
public:
    inline Dtype balance(const Dtype posLoss, const int nPositives,
                         const Dtype negLoss, const int nNegatives) {
        const int total = nPositives+nNegatives;
        this->posAlpha_ = this->negAlpha_ = (total == 0 ? 0 : Dtype(1)/(nPositives+nNegatives));
        return this->superBalance(posLoss,negLoss);
    }
};

template <typename Dtype>
class NormalizeUniformFunctor : public LossBalancingFunctor<Dtype> {
public:
    inline Dtype balance(const Dtype posLoss, const int nPositives,
                         const Dtype negLoss, const int nNegatives) {
        this->posAlpha_ = (nPositives == 0 ? 0 : Dtype(1) / nPositives);
        this->negAlpha_ = (nNegatives == 0 ? 0 : Dtype(1) / nNegatives);
        return this->superBalance(posLoss,negLoss);
    }
};

template <typename Dtype>
class ReweightedNormalizeUniformFunctor : public LossBalancingFunctor<Dtype> {
public:

    explicit ReweightedNormalizeUniformFunctor(const Dtype posWeight,
                                               const Dtype negWeight)
        : posWeight_(posWeight), negWeight_(negWeight) { }

    inline Dtype balance(const Dtype posLoss, const int nPositives,
                         const Dtype negLoss, const int nNegatives) {
        this->posAlpha_ = (nPositives == 0 ? 0 : posWeight_ / nPositives);
        this->negAlpha_ = (nNegatives == 0 ? 0 : negWeight_ / nNegatives);
        return this->superBalance(posLoss,negLoss);
    }

private:

    const Dtype posWeight_;
    const Dtype negWeight_;

};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                      pixelwise weighting
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype>
class NoWeighting {
public:

    inline NoWeighting(const std::vector<Blob<Dtype> *> & /*bottom*/,
                       const int /*pair*/, bool /*gpu*/ = false,
                       bool /*doDiff*/ = false) { }

#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline Dtype weightA(const int /*x*/, const int /*y*/) const {
        return Dtype(1);
    }

#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline Dtype weightB(const Dtype /*x*/, const Dtype /*y*/) const {
        return Dtype(1);
    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void backpropWeightA(const int /*x*/, const int /*y*/, const Dtype /*topDiff*/) { }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void backpropWeightB(const Dtype /*x*/, const Dtype /*y*/, const Dtype /*topDiff*/) { }

};

template <typename Dtype>
class InputWeighting {
public:

    inline InputWeighting(const std::vector<Blob<Dtype> *> & bottom,
                          const int pair, bool doDiff = false,
                          bool gpu = false)
        : weightsA_((gpu ? bottom[5]->gpu_data() : bottom[5]->cpu_data()) + pair*bottom[5]->count(1)),
          weightsB_((gpu ? bottom[6]->gpu_data() : bottom[6]->cpu_data()) + pair*bottom[6]->count(1)),
          diffWeightsA_(doDiff ? (gpu ? bottom[5]->mutable_gpu_data() : bottom[5]->mutable_cpu_data() ) : 0),
          diffWeightsB_(doDiff ? (gpu ? bottom[6]->mutable_gpu_data() : bottom[6]->mutable_cpu_data() ) : 0),
          width_(bottom[5]->width()) { }

#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline Dtype weightA(const int x, const int y) const {

        return weightsA_[x + width_*y];

    }

#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline Dtype weightB(const Dtype x, const Dtype y) const {

        const int baseX = x;
        const int baseY = y;
        const Dtype offX = x - baseX;
        const Dtype offY = y - baseY;

        return (1-offX)*(1-offY)*weightsB_[( baseX ) + width_*( baseY )] +
               (1-offX)*( offY )*weightsB_[( baseX ) + width_*(baseY+1)] +
               ( offX )*(1-offY)*weightsB_[(baseX+1) + width_*( baseY )] +
               ( offX )*( offY )*weightsB_[(baseX+1) + width_*(baseY+1)];

    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void backpropWeightA(const int x, const int y, const Dtype topDiff) {

        AdditionModel<Dtype>::add(diffWeightsA_[x + width_*y], topDiff);

    }

    template <template <typename> class AdditionModel>
#ifdef __CUDACC__
    __device__
#endif // __CUDACC__
    inline void backpropWeightB(const Dtype x, const Dtype y, const Dtype topDiff) {

        const int baseX = x;
        const int baseY = y;
        const Dtype offX = x - baseX;
        const Dtype offY = y - baseY;

        AdditionModel<Dtype>::add(diffWeightsB_[( baseX ) + width_*( baseY )], (1-offX)*(1-offY)*topDiff);
        AdditionModel<Dtype>::add(diffWeightsB_[( baseX ) + width_*(baseY+1)], (1-offX)*( offY )*topDiff);
        AdditionModel<Dtype>::add(diffWeightsB_[(baseX+1) + width_*( baseY )], ( offX )*(1-offY)*topDiff);
        AdditionModel<Dtype>::add(diffWeightsB_[(baseX+1) + width_*(baseY+1)], ( offX )*( offY )*topDiff);

    }

private:

    const Dtype * weightsA_;
    const Dtype * weightsB_;

    Dtype * diffWeightsA_;
    Dtype * diffWeightsB_;

    const int width_;

};

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                        utility wrappers
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename T>
struct Type2Type {
    typedef T OriginalType;
};

template <typename Dtype>
inline RigidMatchFinder<Dtype> * makeMatchFinder(const Dtype * /*vertsB*/,
                                                 const int /*width*/,
                                                 const int /*height*/,
                                                 const TransformationMatrix<Dtype> & transformationGlobalToB,
                                                 const Dtype focalLengthX,
                                                 const Dtype focalLengthY,
                                                 const Dtype principalPointX,
                                                 const Dtype principalPointY,
                                                 Type2Type<RigidMatchFinder<Dtype> >/*tag*/) {

    return new RigidMatchFinder<Dtype>(transformationGlobalToB,focalLengthX,focalLengthY,
                                       principalPointX,principalPointY);

}

template <typename Dtype>
inline FLANNMatchFinder<Dtype> * makeMatchFinder(const Dtype * vertsB,
                                                 const int width,
                                                 const int height,
                                                 const TransformationMatrix<Dtype> & /*transformationGlobalToB*/,
                                                 const Dtype /*focalLengthX*/,
                                                 const Dtype /*focalLengthY*/,
                                                 const Dtype /*principalPointX*/,
                                                 const Dtype /*principalPointY*/,
                                                 Type2Type<FLANNMatchFinder<Dtype> > /*tag*/) {

    return new FLANNMatchFinder<Dtype>(vertsB,width,height);

}

template <typename Dtype>
inline RandomMatchFinder<Dtype> * makeMatchFinder(const Dtype * vertsB,
                                                  const int width,
                                                  const int height,
                                                  const TransformationMatrix<Dtype> & /*transformationGlobalToB*/,
                                                  const Dtype /*focalLengthX*/,
                                                  const Dtype /*focalLengthY*/,
                                                  const Dtype /*principalPointX*/,
                                                  const Dtype /*principalPointY*/,
                                                  Type2Type<RandomMatchFinder<Dtype> >/*tag*/) {

    return new RandomMatchFinder<Dtype>(vertsB,width,height);

}

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
                             PixelwiseWeightingT<Dtype> weighting);


template <typename Dtype,
          template <typename> class LossFunctorT>
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
                             Dtype * gradB);

// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
//
//                         implementation
//
// -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
template <typename Dtype>
class DenseCorrespondenceLayerImplBase {
public:

    virtual ~DenseCorrespondenceLayerImplBase() { }

    virtual void LayerSetUp(const vector<Blob<Dtype> *> & bottom,
                            const vector<Blob<Dtype> *> & top) = 0;

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top) = 0;

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom) = 0;

    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom) = 0;

    virtual const Blob<Dtype> & samplesA() const = 0;

    virtual const Blob<Dtype> & samplesB() const = 0;

    virtual int numPositivesPossible() const = 0;

    virtual int numNegativesPossible() const = 0;

};

template <typename Dtype,
          template <typename> class MatchFinderT,
          template <typename> class PositiveMatchSelectorT,
          template <typename> class PositiveLossFunctorT,
          template <typename> class NegativeMatchSelectorT,
          template <typename> class NegativeLossFunctorT,
          template <typename> class LossBalancingFunctorT,
          template <typename> class PixelwiseWeightingT = NoWeighting>
class DenseCorrespondenceLayerImpl : public DenseCorrespondenceLayerImplBase<Dtype> {
public:

    explicit DenseCorrespondenceLayerImpl(PositiveMatchSelectorT<Dtype> & positiveMatchSelector,
                                          PositiveLossFunctorT<Dtype> & positiveLossFunctor,
                                          NegativeMatchSelectorT<Dtype> & negativeMatchSelector,
                                          NegativeLossFunctorT<Dtype> & negativeLossFunctor,
                                          LossBalancingFunctorT<Dtype> & balancingFunctor,
                                          const Dtype flX, const Dtype flY,
                                          const Dtype ppX, const Dtype ppY,
                                          const bool allowMatchless)
        : positiveSelector_(positiveMatchSelector),
          posLossFunctor_(positiveLossFunctor),
          negativeSelector_(negativeMatchSelector),
          negLossFunctor_(negativeLossFunctor),
          balancingFunctor_(balancingFunctor),
          flX_(flX), flY_(flY), ppX_(ppX), ppY_(ppY),
          allowMatchless_(allowMatchless && MatchFinderT<Dtype>::EnablesMatchless) { }

    virtual ~DenseCorrespondenceLayerImpl() { }

    virtual const Blob<Dtype> & samplesA() const { return samplesA_; }

    virtual const Blob<Dtype> & samplesB() const { return samplesB_; }

    virtual inline int numPositivesPossible() const { return positiveSelector_.totalPossibleMatches(); }

    virtual inline int numNegativesPossible() const { return negativeSelector_.totalPossibleMatches(); }


    void LayerSetUp(const vector<Blob<Dtype> *> & bottom,
                    const vector<Blob<Dtype> *> & top) {

        // -=-=-=-=- checks -=-=-=-=-

        // make sure all blobs have the same num
        const int nPairs = bottom[0]->num();
        CHECK_EQ(nPairs,bottom[1]->num());
        CHECK_EQ(nPairs,bottom[2]->num());
        CHECK_EQ(nPairs,bottom[3]->num());
        CHECK_EQ(nPairs,bottom[4]->num());

        // make sure all blobs have the same shape
        const int denseWidth = bottom[0]->width();
        const int denseHeight = bottom[0]->height();
        CHECK_EQ(denseWidth,bottom[1]->width());
        CHECK_EQ(denseWidth,bottom[2]->width());
        CHECK_EQ(denseWidth,bottom[3]->width());
        CHECK_EQ(denseHeight,bottom[1]->height());
        CHECK_EQ(denseHeight,bottom[2]->height());
        CHECK_EQ(denseHeight,bottom[3]->height());

        // make sure the dense representations have the same number of channels
        const int denseChannels = bottom[0]->channels();
        CHECK_EQ(denseChannels,bottom[1]->channels());

        // make sure the vert maps have 3 channels
        CHECK_EQ(3,bottom[2]->channels());
        CHECK_EQ(3,bottom[3]->channels());

        // make sure the transforms are 3x4
        CHECK_EQ(3,bottom[4]->shape()[1]);
        CHECK_EQ(4,bottom[4]->shape()[2]);

        // -=-=-=-=- set sizes -=-=-=-=-
        const int nPositives = positiveSelector_.totalPossibleMatches();
        const int nNegativesPerPositive = negativeSelector_.totalPossibleMatches();
        const int totalMatches = nPositives*(1 + nNegativesPerPositive);

        std::vector<int> sampleShape(3);
        sampleShape[0] = nPairs;
        sampleShape[1] = totalMatches;
        sampleShape[2] = 2;
        samplesA_.Reshape(sampleShape);
        samplesB_.Reshape(sampleShape);

        std::vector<int> diffShape(3);
        diffShape[0] = nPairs;
        diffShape[1] = totalMatches;
        diffShape[2] = denseChannels;
        diff_.Reshape(diffShape);

        std::vector<int> representationsAShape(3);
        representationsAShape[0] = nPairs;
        representationsAShape[1] = nPositives;
        representationsAShape[2] = denseChannels;
        representationsA_.Reshape(representationsAShape);

        representationsPosB_.ReshapeLike(representationsA_);

        nSuccessfulPositiveSamples_.resize(nPairs);
        nSuccessfulNegativeSamples_.resize(nPairs);

    }

    void Forward_cpu(const vector<Blob<Dtype> *> & bottom,
                     const vector<Blob<Dtype> *> & top) {

        Dtype posLoss = 0;
        Dtype negLoss = 0;

        const int numPairs = bottom[0]->num();

        const int repWidth = bottom[0]->width();
        const int repHeight = bottom[0]->height();
        const int repChannels = bottom[0]->channels();

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- process each pair -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        for (int pair = 0; pair < numPairs; ++pair) {

            positiveSelector_.init();

            const Dtype * repA = bottom[0]->cpu_data() + pair*bottom[0]->count(1);
            const Dtype * repB = bottom[1]->cpu_data() + pair*bottom[1]->count(1);

            const Dtype * vertsA = bottom[2]->cpu_data() + pair*bottom[2]->count(1);
            const Dtype * vertsB = bottom[3]->cpu_data() + pair*bottom[3]->count(1);

            PixelwiseWeightingT<Dtype> pixelwiseWeighting(bottom,pair);

            TransformationMatrix<Dtype> transformationGlobalToB(bottom[4]->cpu_data() + pair*bottom[4]->count(1));

            Dtype * posDiffData = diff_.mutable_cpu_data() + pair*diff_.count(1);
            Dtype * negDiffData = posDiffData + repChannels*positiveSelector_.totalPossibleMatches();

            Dtype * posSampleAData = samplesA_.mutable_cpu_data() + pair*samplesA_.count(1);
            Dtype * negSampleAData = posSampleAData + 2*positiveSelector_.totalPossibleMatches();

            Dtype * posSampleBData = samplesB_.mutable_cpu_data() + pair*samplesB_.count(1);
            Dtype * negSampleBData = posSampleBData + 2*positiveSelector_.totalPossibleMatches();

            Dtype * representationAData = representationsA_.mutable_cpu_data() + pair*representationsA_.count(1);
            Dtype * representationPosBData = representationsPosB_.mutable_cpu_data() + pair*representationsPosB_.count(1);

            int positiveIndex = 0;
            int negativeIndex = 0;

            int uA, vA;
            Dtype uBPos=-1, vBPos=-1;
            Dtype ptA[3];
            bool vertAValid;

            std::auto_ptr<MatchFinderT<Dtype> > matchFinder(makeMatchFinder(vertsB,repWidth,repHeight,
                                                                            transformationGlobalToB,
                                                                            flX_, flY_, ppX_, ppY_,
                                                                            Type2Type<MatchFinderT<Dtype> >()));

            // -=-=-=-=- positives -=-=-=-=-
            while (positiveSelector_.getNextMatch(uA,vA,uBPos,vBPos,ptA,
                                                  vertsA,vertsB,
                                                  repWidth,repHeight,
                                                  *matchFinder,
//                                                  transformationGlobalToB,
//                                                  flX_,flY_,ppX_,ppY_,
                                                  allowMatchless_,
                                                  vertAValid)) {

                // save sample points for gradient computation
                posSampleAData[0 + 2*positiveIndex] = uA;
                posSampleAData[1 + 2*positiveIndex] = vA;
                posSampleBData[0 + 2*positiveIndex] = uBPos;
                posSampleBData[1 + 2*positiveIndex] = vBPos;

                // extract representation of A
                Dtype * thisRepresentationA = representationAData + repChannels*positiveIndex;
                for (int c=0; c<repChannels; ++c) {
                    thisRepresentationA[c] = repA[uA + repWidth*(vA + repHeight*c)];
                }

                const bool negOnly = allowMatchless_ && (uBPos < 0 || uBPos >= repWidth-2 || vBPos < 0 || vBPos >= repHeight-2);
                if (!negOnly) {

                    Dtype * thisRepresentationPosB = representationPosBData + repChannels*positiveIndex;

                    interpolateRepresentation(repB,repWidth,repHeight,repChannels,
                                              uBPos,vBPos,thisRepresentationPosB);

                    ++positiveIndex;

                }
            }

            if (positiveIndex > 0) {
                caffe_sub(positiveIndex*repChannels,representationAData,representationPosBData,posDiffData);
                for (int i=0; i<positiveIndex; ++i) {
                    const Dtype weightA = pixelwiseWeighting.weightA(posSampleAData[0 + 2*positiveIndex],
                                                                     posSampleAData[1 + 2*positiveIndex]);
                    const Dtype weightB = pixelwiseWeighting.weightB(posSampleBData[0 + 2*positiveIndex],
                                                                     posSampleBData[1 + 2*positiveIndex]);
                    posLoss += weightA*weightB*posLossFunctor_.loss(posDiffData + i*repChannels,repChannels);
                }
            }

            // -=-=-=-=- negatives -=-=-=-=-
            negativeSelector_.initFrame(repB,repWidth,repHeight,repChannels);

            for (int i=0; i<positiveIndex; ++i) {

                const Dtype * thisRepresentationA = representationAData + repChannels*i;

                negativeSelector_.initPoint(thisRepresentationA);

                uA = posSampleAData[0 + 2*i];
                vA = posSampleAData[1 + 2*i];
                uBPos = posSampleBData[0 + 2*i];
                vBPos = posSampleBData[1 + 2*i];

                int uBNeg, vBNeg;

                std::vector<Dtype> representationB(repChannels);
                while (negativeSelector_.getNextMatch(uBNeg,vBNeg,
                                                      uBPos,vBPos,
                                                      repB,
                                                      repWidth,repHeight,
                                                      repChannels,
                                                      negLossFunctor_.hasMargin() ? negLossFunctor_.margin()*negLossFunctor_.margin() : std::numeric_limits<Dtype>::infinity(),
                                                      thisRepresentationA,
                                                      representationB,
                                                      negDiffData + negativeIndex*repChannels)) {

                    // save sample points for gradient computation
                    negSampleAData[0 + 2*negativeIndex] = uA;
                    negSampleAData[1 + 2*negativeIndex] = vA;
                    negSampleBData[0 + 2*negativeIndex] = uBNeg;
                    negSampleBData[1 + 2*negativeIndex] = vBNeg;

                    Dtype * thisDiff = negDiffData + negativeIndex*repChannels;

                    negLoss += negLossFunctor_.loss(thisDiff,repChannels);

                    ++negativeIndex;

                }

            }

            nSuccessfulPositiveSamples_[pair] = positiveIndex;
            nSuccessfulNegativeSamples_[pair] = negativeIndex;

        }


        const int nPositives = std::accumulate(nSuccessfulPositiveSamples_.begin(),nSuccessfulPositiveSamples_.end(),0);
        const int nNegatives = std::accumulate(nSuccessfulNegativeSamples_.begin(),nSuccessfulNegativeSamples_.end(),0);

        const Dtype loss = balancingFunctor_.balance(posLoss,nPositives,negLoss,nNegatives);

        std::cout << posLoss << ", " << nPositives << "; " << negLoss << ", " << nNegatives << "; " << loss << std::endl;

        top[0]->mutable_cpu_data()[0] = loss;

    }

    void Backward_cpu(const vector<Blob<Dtype>*> & top,
                      const vector<bool> & propagate_down,
                      const vector<Blob<Dtype>*> & bottom) {

        const int numPairs = bottom[0]->num();

        const int repWidth = bottom[0]->width();
        const int repHeight = bottom[0]->height();
        const int repChannels = bottom[0]->channels();

        const Dtype posAlpha = top[0]->cpu_diff()[0]*balancingFunctor_.posAlpha();
        const Dtype negAlpha = top[0]->cpu_diff()[0]*balancingFunctor_.negAlpha();

        caffe_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_cpu_diff());
        caffe_set(bottom[1]->count(),Dtype(0),bottom[1]->mutable_cpu_diff());

        for (int pair = 0; pair < numPairs; ++pair) {

            std::cout << "pair " << pair << std::endl;

            const Dtype * posDiffData = diff_.mutable_cpu_data() + pair*diff_.count(1);
            const Dtype * negDiffData = posDiffData + repChannels*positiveSelector_.totalPossibleMatches();

            const Dtype * posSampleAData = samplesA_.mutable_cpu_data() + pair*samplesA_.count(1);
            const Dtype * negSampleAData = posSampleAData + 2*positiveSelector_.totalPossibleMatches();

            const Dtype * posSampleBData = samplesB_.mutable_cpu_data() + pair*samplesB_.count(1);
            const Dtype * negSampleBData = posSampleBData + 2*positiveSelector_.totalPossibleMatches();

            Dtype * diffA = bottom[0]->mutable_cpu_diff() + pair*bottom[0]->count(1);
            Dtype * diffB = bottom[1]->mutable_cpu_diff() + pair*bottom[1]->count(1);

            PixelwiseWeightingT<Dtype> pixelwiseWeighting(bottom,pair,true);

            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- gradients for positives -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

            std::cout << "positives" << std::endl;
            for (int i = 0; i < nSuccessfulPositiveSamples_[pair]; ++i) {

                const int uA = std::floor(posSampleAData[0 + 2*i] + 0.5);
                const int vA = std::floor(posSampleAData[1 + 2*i] + 0.5);
                const Dtype uB = posSampleBData[0 + 2*i];
                const Dtype vB = posSampleBData[1 + 2*i];

                const Dtype * thisDiff = posDiffData + i*repChannels;

                const Dtype weightA = pixelwiseWeighting.weightA(uA,vA);
                const Dtype weightB = pixelwiseWeighting.weightB(uB,vB);
                const Dtype thisAlpha = weightA*weightB*posAlpha;

                posLossFunctor_.template differentiateLoss<SingleThreadedAddition>(thisDiff,repWidth,repHeight,repChannels,
                                                                                   uA,vA, thisAlpha,diffA);
                posLossFunctor_.template differentiateLoss<SingleThreadedAddition>(thisDiff,repWidth,repHeight,repChannels,
                                                                                   uB,vB,-thisAlpha,diffB);

                pixelwiseWeighting.template backpropWeightA<SingleThreadedAddition>(uA,vA,weightB*posLossFunctor_.loss(thisDiff,repChannels));
                pixelwiseWeighting.template backpropWeightB<SingleThreadedAddition>(uB,vB,weightA*posLossFunctor_.loss(thisDiff,repChannels));

            }

            std::cout << "negatives (" << nSuccessfulNegativeSamples_[pair] << ")" << std::endl;
            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- gradients for negatives -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            for (int i = 0; i < nSuccessfulNegativeSamples_[pair]; ++i) {

                const int uA = std::floor(negSampleAData[0 + 2*i] + 0.5);
                const int vA = std::floor(negSampleAData[1 + 2*i] + 0.5);
                const int uB = std::floor(negSampleBData[0 + 2*i] + 0.5);
                const int vB = std::floor(negSampleBData[1 + 2*i] + 0.5);

                const Dtype * thisDiff = negDiffData + i*repChannels;

                negLossFunctor_.template differentiateLoss<SingleThreadedAddition>(thisDiff,repWidth,repHeight,repChannels,
                                                                                   uA,vA,negAlpha,diffA);
                negLossFunctor_.template differentiateLoss<SingleThreadedAddition>(thisDiff,repWidth,repHeight,repChannels,
                                                                                   uB,vB,-negAlpha,diffB);

            }

        }

    }


    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down,
                              const vector<Blob<Dtype>*>& bottom) {

        caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_gpu_diff());
        caffe_gpu_set(bottom[1]->count(),Dtype(0),bottom[1]->mutable_gpu_diff());

        if (bottom.size() > 5) {
            caffe_gpu_set(bottom[5]->count(),Dtype(0),bottom[5]->mutable_gpu_diff());
            caffe_gpu_set(bottom[6]->count(),Dtype(0),bottom[6]->mutable_gpu_diff());
        }

        const int numPairs = bottom[0]->num();

        const int width = bottom[0]->width();
        const int height = bottom[0]->height();
        const int channels = bottom[0]->channels();

        const Dtype posAlpha = top[0]->cpu_diff()[0]*balancingFunctor_.posAlpha();
        const Dtype negAlpha = top[0]->cpu_diff()[0]*balancingFunctor_.negAlpha();

        for (int pair = 0; pair < numPairs; ++pair) {

            std::cout << "pair " << pair << std::endl;

            const Dtype * posDiffData = diff_.mutable_gpu_data() + pair*diff_.count(1);
            const Dtype * negDiffData = posDiffData + channels*positiveSelector_.totalPossibleMatches();

            const Dtype * posSampleAData = samplesA_.mutable_gpu_data() + pair*samplesA_.count(1);
            const Dtype * negSampleAData = posSampleAData + 2*positiveSelector_.totalPossibleMatches();

            const Dtype * posSampleBData = samplesB_.mutable_gpu_data() + pair*samplesB_.count(1);
            const Dtype * negSampleBData = posSampleBData + 2*positiveSelector_.totalPossibleMatches();

            Dtype * diffA = bottom[0]->mutable_gpu_diff() + pair*bottom[0]->count(1);
            Dtype * diffB = bottom[1]->mutable_gpu_diff() + pair*bottom[1]->count(1);

            PixelwiseWeightingT<Dtype> pixelwiseWeighting(bottom,pair,true,true);

            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- gradients for positives -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

            std::cout << "positives" << std::endl;
            const int posN = nSuccessfulPositiveSamples_[pair];

            backwardPositiveWrapper<Dtype,PositiveLossFunctorT,PixelwiseWeightingT>(
                        posN,width,height,channels,posDiffData,posSampleAData,posSampleBData,
                        posAlpha,posLossFunctor_,diffA,diffB,pixelwiseWeighting);


            std::cout << "negatives (" << nSuccessfulNegativeSamples_[pair] << ")" << std::endl;
            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- gradients for negatives -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            const int negN = nSuccessfulNegativeSamples_[pair];

            backwardNegativeWrapper<Dtype,NegativeLossFunctorT>(
                        negN,width,height,channels,negDiffData,negSampleAData,negSampleBData,
                        negAlpha,negLossFunctor_,diffA,diffB);

        }


    }

private:

    PositiveMatchSelectorT<Dtype> positiveSelector_;
    PositiveLossFunctorT<Dtype> posLossFunctor_;

    NegativeMatchSelectorT<Dtype> negativeSelector_;
    NegativeLossFunctorT<Dtype> negLossFunctor_;

    LossBalancingFunctorT<Dtype> balancingFunctor_;

    const Dtype flX_, flY_;
    const Dtype ppX_, ppY_;
    const bool allowMatchless_;

    vector<int> nSuccessfulPositiveSamples_;
    vector<int> nSuccessfulNegativeSamples_;

    Blob<Dtype> samplesA_, samplesB_;
    Blob<Dtype> representationsA_, representationsPosB_;
    Blob<Dtype> diff_;
};

} // namespace caffe

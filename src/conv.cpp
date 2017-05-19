#include <Rcpp.h>
#include "tensor.h"

using namespace Rcpp;

inline int cap(int x, int a, int b) {
    if (x < a) return a;
    if (x > b) return b;
    return x;
}

inline int cap_reflect(int x, int a, int b) {
    if (x < a) {
        int d = a - x;
        return b - d;
    }
    if (x > b) {
        int d = x - b;
        return a + d;
    }
    return x;
}

inline double _conv3_ij_f(DTensor3 image, DTensor3 kernel, int i, int j) {
    int H = (kernel.dim(0) - 1)/2;
    int W = (kernel.dim(1) - 1)/2;
    int I = image.dim(0);
    int J = image.dim(1);
    int K = image.dim(2);
    int i_h;
    int j_w;
    double ans = 0.0;

    for (int k = 0; k < K; k++) {
        for (int w = -W; w <= W; w++) {
            j_w = cap(j - w, 0, J-1);
            for (int h = -H; h <= H; h++) {
                i_h = cap(i - h, 0, I-1);
                ans += kernel(h+H, w+W, k) * image(i_h, j_w, k);
            }
        }
    }

    return ans;
}

inline double _conv3_ij_b(DTensor3 image, DTensor3 kernel, int i, int j) {
    int H = (kernel.dim(0) - 1)/2;
    int W = (kernel.dim(1) - 1)/2;
    int I = image.dim(0);
    int J = image.dim(1);
    int K = image.dim(2);
    int i_h;
    int j_w;
    double ans = 0.0;

    for (int k = 0; k < K; k++) {
        for (int w = -W; w <= W; w++) {
            int w1 = w;
            j_w = j - w;
            if (j_w < 0 || j_w >= J) {
                w1 = -w;
            }
            j_w = cap(j_w, 0, J-1);
            for (int h = -H; h <= H; h++) {
                i_h = i-h;
                int h1 = h;
                if (i_h < 0 || i_h >= I) {
                    h1 = -h;
                }
                i_h = cap(i_h, 0, I-1);
                ans += kernel(h1+H, w1+W, k) * image(i_h, j_w, k);
            }
        }
    }

    return ans;
}

void _conv3_f(DTensor3 image, DTensor4 kernels, DTensor3 outImage) {
    int I = outImage.dim(0);
    int J = outImage.dim(1);
    int K = outImage.dim(2);

    for (int k = 0; k < K; k++) {
        DTensor3 kernel = kernels(k);
        for (int j = 0; j < J; j++) {
            for (int i = 0; i < I; i++) {
                outImage(i,j,k) = _conv3_ij_f(image, kernel, i, j);
            }
        }
    }
}

void _conv3_b(DTensor3 image, DTensor4 kernels, DTensor3 outImage) {
    int I = outImage.dim(0);
    int J = outImage.dim(1);
    int K = outImage.dim(2);

    for (int k = 0; k < K; k++) {
        DTensor3 kernel = kernels(k);
        for (int j = 0; j < J; j++) {
            for (int i = 0; i < I; i++) {
                outImage(i,j,k) = _conv3_ij_b(image, kernel, i, j);
            }
        }
    }
}


// [[Rcpp::export]]
NumericVector conv3d(NumericVector images_4d, NumericVector kernels_4d, NumericVector outImages_4d, std::string flag) {
    IntegerVector inputDim = images_4d.attr("dim");
    IntegerVector outDim = outImages_4d.attr("dim");

    int width    = inputDim(0);
    int height   = inputDim(1);
    int depth    = inputDim(2);
    int nImages  = inputDim(3);
    int outDepth = outDim(2);

    int inSize  = width * height * depth;
    int outSize = width * height * outDepth;
    bool forward = flag == "forward";

    DTensor4 kernels(kernels_4d.begin(), kernels_4d.attr("dim"));

    if (forward) {
        for (int i = 0; i < nImages; i++) {
            DTensor3 image(&images_4d[0] + i*inSize, width, height, depth);
            DTensor3 outImage(&outImages_4d[0] + i*outSize, width, height, outDepth);
            _conv3_f(image, kernels, outImage);
        }
    } else {
        for (int i = 0; i < nImages; i++) {
            DTensor3 image(&images_4d[0] + i*inSize, width, height, depth);
            DTensor3 outImage(&outImages_4d[0] + i*outSize, width, height, outDepth);
            _conv3_b(image, kernels, outImage);
        }
    }
    return outImages_4d;
}

void _convFlip(DTensor4 source, DTensor4 target) {
    int I = source.dim(0);
    int J = source.dim(1);
    int K = source.dim(2);
    int L = source.dim(3);
    for (int l = 0; l < L; l++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++) {
                for (int i = 0; i < I; i++) {
                    target((I-i-1), (J-j-1), l, k) = source(i, j, k, l);
                }
            }
        }
    }
}

// [[Rcpp::export]]
NumericVector convFlip(NumericVector source, NumericVector target) {
    DTensor4 _source(source.begin(), source.attr("dim"));
    DTensor4 _target(target.begin(), target.attr("dim"));
    _convFlip(_source, _target);
    return target;
}

void convInv(DTensor3 u, DTensor3 v, DTensor4 out, int H, int W) {
    int I = u.dim(0);
    int J = u.dim(1);
    int D1 = u.dim(2);
    int D2 = v.dim(2);

    for (int h = -H; h <= H; h++) {
        for (int w = -W; w <= W; w++) {
            for (int d1 = 0; d1 < D1; d1++) {
                for (int d2 = 0; d2 < D2; d2++) {
                    double ans = 0.0;
                    for (int i = 0; i < I; i++) {
                        for (int j = 0; j < J; j++) {
                            int i_h = cap(i-h, 0, I-1);
                            int j_w = cap(j-w, 0, J-1);
                            ans += u(i,j,d1) * v(i_h, j_w, d2);
                        }
                    }
                    out(h+H, w+W, d2, d1) = ans;
                }
            }
        }
    }
}

// [[Rcpp::export]]
NumericVector sumConvInv(NumericVector delta, NumericVector x, int H, int W) {
    DTensor4 _delta(delta.begin(), delta.attr("dim"));
    DTensor4 _x(x.begin(), x.attr("dim"));
    int I = (2*H+1);
    int J = (2*W+1);
    int K = _x.dim(2);
    int L = _delta.dim(2);

    int size = I*J*K*L;
    NumericVector out(size);
    out.attr("dim") = IntegerVector::create(I, J, K, L);
    NumericVector tmp(size);
    tmp.attr("dim") = IntegerVector::create(I, J, K, L);
    DTensor4 _tmp(tmp.begin(), tmp.attr("dim"));
    int nExamples = _delta.dim(3);
    for (int i = 0; i < nExamples; i++) {
        convInv(_delta(i), _x(i), _tmp, H, W);
        out += tmp;
    }
    return out;
}

// layer$a = MaxPoolForwardC(x, H, W)
// [[Rcpp::export]]
NumericVector MaxPoolForwardC(NumericVector x, int H, int W) {
    IntegerVector dim = x.attr("dim");
    
    int I = dim[0], J = dim[1], K = dim[2], L = dim[3];
    int I1 = I / H, J1 = J / W;

    NumericVector out(I1*J1*K*L);
    out.attr("dim") = IntegerVector::create(I1, J1, K, L);

    DTensor4 _x(x.begin(), x.attr("dim"));
    DTensor4 _out(out.begin(), out.attr("dim"));

    for (int l = 0; l < L; l++) {
        for (int k = 0; k < K; k++) {
            for (int j1 = 0; j1 < J1; j1++) {
                for (int i1 = 0; i1 < I1; i1++) {
                    int i0 = i1 * H;
                    int j0 = j1 * W;
                    double ans = _x(i1*H, j1*W, k, l);
                    for (int j = 0; j < W; j++) {
                        for (int i = 0; i < H; i++) {
                            double tmp = _x(i0+i, j0+j, k, l);
                            ans = ans < tmp ? tmp : ans;
                        }
                    }
                    _out(i1,j1,k,l) = ans;
                }
            }
        }
    }
    return out;
}

//  layer$errorIn = MaxPoolBackwardC(layer$x, layer$a, errorOut)
// [[Rcpp::export]]
NumericVector MaxPoolBackwardC(NumericVector x, NumericVector a, NumericVector errorOut) {
    DTensor4 _x(x.begin(), x.attr("dim"));
    DTensor4 _a(a.begin(), a.attr("dim"));
    DTensor4 _errorOut(errorOut.begin(), errorOut.attr("dim"));
    NumericVector errorIn(x.size());
    errorIn.attr("dim") = x.attr("dim");
    DTensor4 _errorIn(errorIn.begin(), x.attr("dim"));

    int I = _x.dim(0), J = _x.dim(1), K = _x.dim(2), L = _x.dim(3);
    int I1 = _a.dim(0), J1 = _a.dim(1);
    int H = I/I1, W = J/J1;

    for (int l = 0; l < L; l++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < J; j++) {
                for (int i = 0; i < I; i++) {
                    if (fabs(_x(i,j,k,l)-_a(i/H,j/W,k,l)) < 1e-5) {
                        _errorIn(i,j,k,l) = _errorOut(i/H,j/W,k,l);
                    }
                }
            }
        }
    }
    return errorIn;
}


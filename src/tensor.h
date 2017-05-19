#ifndef Rcpp__Tensor__h
#define Rcpp__Tensor__h

#include <Rcpp.h>
using namespace Rcpp;

template<typename ValueType>
class Tensor2 {
    ValueType* data;
    int I, J;
public:
    Tensor2<ValueType> (ValueType* _data, int _I, int _J) {
        I = _I;
        J = _J;
        data = _data;
    }

    Tensor2<ValueType> (ValueType* _data, IntegerVector dim) {
        data = _data;
        I = dim[0];
        J = dim[1];
    }
    
    ValueType& operator() (int i, int j) {
        return data[j*I + i];
    }

    ValueType& operator[] (int i) {
        return data[i];
    }

    void fill(ValueType value) {
        int n = I * J;
        for (int i = 0; i < n; i++) {
            data[i] = value;
        }
    }

    int dim(int i) {
        switch (i)  {
            case 0: return I;
            case 1: return J;
            default: return 0;
        }
    }

};

template<typename ValueType>
class Tensor3 {
    ValueType* data;
    int I, J, K, IJ;
public:
    Tensor3<ValueType> (ValueType* _data, int _I, int _J, int _K) {
        I = _I;
        J = _J;
        K = _K;
        IJ = I * J;
        data = _data;
    }

    Tensor3<ValueType> (ValueType* _data, IntegerVector dim) {
        data = _data;
        I = dim[0];
        J = dim[1];
        K = dim[2];
        IJ = I * J;
    }
    
    ValueType& operator() (int i, int j, int k) {
        return data[k*IJ + j*I + i];
    }

    Tensor2<ValueType> operator() (int k) {
        return Tensor2<ValueType> (data + k*IJ, I, J);
    }

    ValueType& operator[] (int i) {
        return data[i];
    }

    void fill(ValueType value) {
        int n = IJ * K;
        for (int i = 0; i < n; i++) {
            data[i] = value;
        }
    }

    int dim(int i) {
        switch (i)  {
            case 0: return I;
            case 1: return J;
            case 2: return K;
            default: return 0;
        }
    }

};

template<typename ValueType>
class Tensor4 {
    ValueType* data;
    int I, J, K, L, IJ, IJK;
public:
    Tensor4<ValueType> (ValueType* _data, int _I, int _J, int _K, int _L) {
        I = _I;
        J = _J;
        K = _K;
        L = _L;
        IJ = I * J;
        IJK = I * J * K;
        data = _data;
    }

    Tensor4<ValueType> (ValueType* _data, IntegerVector dim) {
        data = _data;
        I = dim[0];
        J = dim[1];
        K = dim[2];
        L = dim[3];
        IJ = I * J;
        IJK = I * J * K;
    }
    
    ValueType& operator() (int i, int j, int k, int l) {
        return data[l*IJK + k*IJ + j*I + i];
    }

    Tensor3<ValueType> operator() (int l) {
        return Tensor3<ValueType> (data + l*IJK, I, J, K);
    }

    Tensor2<ValueType> operator() (int k, int l) {
        return Tensor2<ValueType> (data + l*IJK + k*IJ, I, J);
    }

    ValueType& operator[] (int i) {
        return data[i];
    }

    void fill(ValueType value) {
        int n = IJK * L;
        for (int i = 0; i < n; i++) {
            data[i] = value;
        }
    }

    int dim(int i) {
        switch (i)  {
            case 0: return I;
            case 1: return J;
            case 2: return K;
            case 3: return L;
            default: return 0;
        }
    }

};



typedef Tensor2<double> DTensor2;
typedef Tensor3<double> DTensor3;
typedef Tensor4<double> DTensor4;

#endif
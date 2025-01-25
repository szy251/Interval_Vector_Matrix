#include <immintrin.h>
#include <limits>
#include<iostream>
namespace opt{
#define i_i(l, b) ((l << 4) | b)

  enum IntervalClass {
    // sole interval
    E = 0b0000,  // Empty
    M  = 0b0110, // [-a, b]
    P = 0b0011,  // [a, b]
    P0 = 0b0111, // [0, b]
    N0 = 0b1110, // [-a, 0]
    N  = 0b1100, // [-a, -b]
    Z = 0b1111,  // [0, 0]
    // interval x interval
    E_E = i_i(E, E),
    E_Z = i_i(E, Z),
    E_P = i_i(E, P),
    E_P0 = i_i(E, P0),
    E_N = i_i(E, N),
    E_N0 = i_i(E, N0),
    E_M = i_i(E, M),
    
    Z_E = i_i(Z, E),
    Z_Z = i_i(Z, Z),
    Z_P = i_i(Z, P),
    Z_P0 = i_i(Z, P0),
    Z_N = i_i(Z, N),
    Z_N0 = i_i(Z, N0),
    Z_M = i_i(Z, M),
  
    P_E = i_i(P, E),
    P_Z = i_i(P, Z),
    P_P = i_i(P, P),
    P_P0 = i_i(P, P0),
    P_N = i_i(P, N),
    P_N0 = i_i(P, N0),
    P_M = i_i(P, M),

    P0_E = i_i(P0, E),
    P0_Z = i_i(P0, Z),
    P0_P = i_i(P0, P),
    P0_P0 = i_i(P0, P0),
    P0_N = i_i(P0, N),
    P0_N0 = i_i(P0, N0),
    P0_M = i_i(P0, M),
  
    N_E = i_i(N, E),
    N_Z = i_i(N, Z),
    N_P = i_i(N, P),
    N_P0 = i_i(N, P0),
    N_N = i_i(N, N),
    N_N0 = i_i(N, N0),
    N_M = i_i(N, M),

    N0_E = i_i(N0, E),
    N0_Z = i_i(N0, Z),
    N0_P = i_i(N0, P),
    N0_P0 = i_i(N0, P0),
    N0_N = i_i(N0, N),
    N0_N0 = i_i(N0, N0),
    N0_M = i_i(N0, M),
  
    _M_E = i_i(M, E),
    M_Z = i_i(M, Z),
    M_P = i_i(M, P),
    M_P0 = i_i(M, P0),
    M_N = i_i(M, N),
    M_N0 = i_i(M, N0),
    M_M = i_i(M, M),
  };


__m128d swap(__m128d vec)
{
    return (_mm_shuffle_pd(vec,vec,1));
}
 __m128d neg_0(__m128d i) {
  return _mm_xor_pd(i, __m128d{-0.,0.});
}
int intervalClass(__m128d i) {
   return ((_mm_movemask_pd(_mm_cmple_pd(i, __m128d{0., 0.})) << 2) | _mm_movemask_pd(_mm_cmpge_pd(i, __m128d{0., 0.})));
 }
int intervalClass(__m128d x, __m128d y) {
  return i_i(intervalClass(neg_0(x)), intervalClass(neg_0(y)));
}

__m128d mult(__m128d lhs, __m128d rhs) {
  switch (intervalClass(lhs, rhs))
  {
  case M_M: {
      // __m128d a = _mm_shuffle_pd(lhs, lhs, 0); // [-a, b] -> [-a, -a]
      // __m128d b = swap(rhs); // [-c, d] -> [d, -c]
      
      // __m128d c = _mm_shuffle_pd(lhs, lhs, 3); // [-a, b] -> [b, b]
      // __m128d d = rhs; // [-c, d] -> [-c, d]

      return _mm_max_pd(_mm_mul_pd(_mm_shuffle_pd(lhs, lhs, 0), swap(rhs)), _mm_mul_pd(_mm_shuffle_pd(lhs, lhs, 3), rhs)); // max([-a*d, -a*-c], [b*-c, b*d]), (min(i) = max(-i))
  }
  case M_N0:
  case M_N:
      return _mm_mul_pd(swap(lhs), _mm_shuffle_pd(rhs, rhs, 0)); // [-a, b] -> [b, -a], [-c, d] -> [-c, -c], [b * -c, a * c]
  case M_P0:
  case M_P: {
      return _mm_mul_pd(lhs, _mm_shuffle_pd(rhs, rhs, 3)); // [-a, b], [-c, d] -> [d, d], [-a * d, b * d]
  }
  case N0_M:
  case N_M: {
      return _mm_mul_pd(_mm_shuffle_pd(lhs, lhs, 0), swap(rhs)); // [-a, b] -> [-a, -a], [-c, d] -> [d, -c], [-a * d, a * c]
  }
  case N0_N0:
  case N0_N:
  case N_N0:
  case N_N: {
      return _mm_mul_pd(neg_0(swap(lhs)), swap(rhs)); // [-a, b] -> [-b, -a], [-c , d] -> [d, -c], [-b * d, a * c]
  }
  case N0_P0:
  case N0_P:
  case N_P0:
  case N_P: {
      return _mm_mul_pd(lhs, swap(neg_0(rhs))); // [-a, b], [-c , d] -> [d, c], [-a * d, b * c]
  }
  case P0_M:
  case P_M: {
      return _mm_mul_pd(_mm_shuffle_pd(lhs, lhs, 3), rhs); // [-a, b] -> [b, b], [-c, d], [b * -c, b * d]
  }
  case P0_N0:
  case P0_N:
  case P_N0:
  case P_N: {
      return _mm_mul_pd(swap(neg_0(lhs)), rhs); // [-a, b] -> [b, a], [-c, d], [b * -c, a * d]
  }
  case P0_P0:
  case P0_P:
  case P_P0:
  case P_P: {
      return _mm_mul_pd(neg_0(lhs), rhs); // [-a, b] -> [a, b], [-c, d], [a * -c, b * d]
  }
  case M_Z:
  case N0_Z:
  case N_Z:
  case P0_Z:
  case P_Z:
  case Z_M:
  case Z_N0:
  case Z_N:
  case Z_P0:
  case Z_P:
  case Z_Z:
      return __m128d{0., 0.}; // [0, 0]
  case E_E:
  case E_M:
  case E_N0:
  case E_N:
  case E_P0:
  case E_P:
  case E_Z:
  case _M_E:
  case N0_E:
  case N_E:
  case P0_E:
  case P_E:
  case Z_E: {
      return __m128d{std::numeric_limits<double>::signaling_NaN(), std::numeric_limits<double>::signaling_NaN()};// [nan, nan]
  }
  default:
    throw std::runtime_error("Not specified case in multiply operation");
  }
  return __m128d{0., 0.};
}

__m128d div(__m128d lhs, __m128d rhs) {
  switch (intervalClass(lhs, rhs)){
    case E_E:
    case E_M:
    case E_N0:
    case E_N:
    case E_P0:
    case E_P:
    case E_Z:
    case _M_E:
    case N0_E:
    case N_E:
    case P0_E:
    case P_E:
    case Z_E:
    case M_Z:
    case N0_Z:
    case N_Z:
    case P0_Z:
    case P_Z:
    case Z_Z: {
        return __m128d{std::numeric_limits<double>::signaling_NaN(), std::numeric_limits<double>::signaling_NaN()};// [nan, nan]
    }
    case Z_M:
    case Z_N0:
    case Z_N:
    case Z_P0:
    case Z_P: {
        return __m128d{0., 0.}; // [0, 0]
    }
    case M_M: 
    case M_N0:
    case M_P0:
    case N0_M:
    case N_M:
    case P0_M:
    case P_M: {
        return __m128d{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()}; // [-∞, ∞]
    }
    case M_N: {
      rhs = neg_0(swap(rhs)); // [-c, d] -> [-d, -c]
      return _mm_div_pd(swap(lhs), _mm_shuffle_pd(rhs, rhs, 0)); // [-a, b] -> [b, -a], [-d, -c] -> [-d, -d], [b/-d, a/d]
    }
    case M_P: {
      rhs = neg_0(rhs); // [-c, d] -> [c, d]
      return _mm_div_pd(lhs, _mm_shuffle_pd(rhs, rhs, 0)); // [-a, b], [c, d] -> [c, c], [-a/c, b/c] 
    }
    case N0_N0:
    case N_N0: {
      __m128d d = _mm_div_pd(swap(lhs), _mm_shuffle_pd(rhs, rhs, 0)); // [-a, b] -> [b, -a], [-c, d], [b/-c, -a/d]
      __m128d inf = __m128d{std::numeric_limits<double>::infinity(), 0.}; // [∞, 0]
      return _mm_shuffle_pd(d, inf, 0); // [b/-c, -a/d], [∞, 0], [b/-c, ∞]
    }
    case N0_N:
    case N_N: {
      lhs = neg_0(lhs); // [-a, b] -> [a, b]
      return _mm_div_pd(swap(lhs), rhs); // [a, b] -> [b, a], [-c, d], [b/-c, a/d] 
    }
    case N0_P0:
    case N_P0: {
      __m128d d = _mm_div_pd(lhs, rhs); // [-a, b], [-c, d], [-a/-c, b/d]
      __m128d inf = __m128d{std::numeric_limits<double>::infinity(), 0.}; // [∞, 0]
      return _mm_shuffle_pd(inf, d, 2); // [∞, 0], [a/c, b/d], [∞, b/d]
    }
    case N0_P:
    case N_P: {
        return _mm_div_pd(lhs, neg_0(rhs)); //[-a, b], [-c, d] -> [c, d], [-a/c, b/d] 
    }

    case P0_N0:
    case P_N0: {
      __m128d d = _mm_div_pd(lhs, rhs); // [-a, b], [-c, d], [-a/-c, b/d]
      __m128d inf = __m128d{std::numeric_limits<double>::infinity(), 0.}; // [∞, 0]
      return _mm_shuffle_pd(inf, d, 0); // [∞, 0], [a/c, b/d], [∞, a/c]
    }
    case P0_N:
    case P_N: {
        return _mm_div_pd(swap(lhs), neg_0(swap(rhs))); //[-a, b] -> [b, -a], [-c, d] -> [-d, -c], [b/-d, -a/-c] 
    }
    case P0_P0:
    case P_P0: {
      __m128d d = _mm_div_pd(lhs, swap(rhs)); // [-a, b], [-c, d] -> [d, -c], [-a/d, b/-c]
      __m128d inf = __m128d{std::numeric_limits<double>::infinity(), 0.}; // [∞, 0]
      return _mm_shuffle_pd(d, inf, 0); // [-a/d, b/-c], [∞, 0], [-a/d, ∞]
    }
    case P0_P:
    case P_P: {
        return _mm_div_pd(lhs, swap(neg_0(rhs))); // [-a, b], [-c, d] -> [d, c], [-a/d, b/c]
    }
    default:
      throw std::runtime_error("Not specified case in divide operation");
    }
  return __m128d{0., 0.};
}

__m128d add(__m128d a, __m128d b){
    return _mm_add_pd(a,b);
}
__m128d sub(__m128d a, __m128d b){
    return _mm_add_pd(a,swap(b));
}

bool equal(__m128d a, __m128d b){
    return _mm_movemask_pd(_mm_cmpeq_pd(a,b))==3;
}

double _leftBound(__m128d a)
{
    return -_mm_cvtsd_f64(a);
}
double _rightBound(__m128d a){
    return _mm_cvtsd_f64(_mm_unpackhi_pd(a,a));
}
}
#include "tfmath.h"

/** 极速数学运算库
 * terminal fast math library
 * all functions with 'tf' prefix
 */

#define TF_HIGH_PRECISION 1
#define USE_SQRT_INTRINS 0
#define TF_INLINE inline

#if USE_SQRT_INTRINS &&                                                        \
    (defined(__x86_64__) || defined(_M_X64) ||                                 \
     ((defined(i386) || defined(__i386__) || defined(__i386) ||                \
       defined(_M_IX86)) &&                                                    \
      (defined(__SSE2__) || (defined(_M_IX86_FP) && _M_IX86_FP == 2))))
#include <emmintrin.h>
#include <xmmintrin.h>
#if defined(__cpp_lambdas)
#define SQRT(x)                                                                \
  (([&] {                                                                      \
    double a;                                                                  \
    __m128d ret = _mm_set_sd(x);                                               \
    ret = _mm_sqrt_sd(ret, ret);                                               \
    _mm_store_sd(&a, ret);                                                     \
    return a;                                                                  \
  })())
#define SQRTF(x)                                                               \
  (([&] {                                                                      \
    float a;                                                                   \
    __m128 ret;                                                                \
    ret = _mm_sqrt_ss(_mm_set_ss(x));                                          \
    _mm_store_ss(&a, ret);                                                     \
    return a;                                                                  \
  })())
#else
#define SQRT(x) _mm_cvtsd_f64(_mm_sqrt_sd(_mm_set_sd(x), _mm_set_sd(x)))
#define SQRTF(x) _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x)))
#endif
#else
//#pragma warning("the compiler can not USE_SQRT_INTRINS!")
#ifdef USE_SQRT_INTRINS
#undef USE_SQRT_INTRINS
#endif
#define USE_SQRT_INTRINS 0
#endif // USE_SQRT_INTRINS
#if !USE_SQRT_INTRINS
#define SQRT(x) tfsqrt(x)
#define SQRTF(x) tfsqrtf(x)
#endif

#define shr_adc(a, c) (((uint32_t)(a) >> ((c)-1)) - ((uint32_t)(a) >> (c)))
#define sar_adc(a, c) (((int32_t)(a) >> ((c)-1)) - ((int32_t)(a) >> (c)))
#if defined(__GNUC__) &&                                                       \
    (defined(__x86_64__) || defined(_M_X64) || defined(i386) ||                \
     defined(__i386__) || defined(__i386) || defined(_M_IX86))
#define shrd_adc(a, b, c) /* (((u64)b<<32)+a)>>sr, and return retsult + CF */  \
  ({                                                                           \
    uint32_t ta = a, tb = b;                                                   \
    asm("shll %%edx,%3 \n\t"                                                   \
        "shrl %%eax,%2 \n\t"                                                   \
        "adcl %%eax,%%edx \n\t"                                                \
        : "+a"(ta), "+d"(tb)                                                   \
        : "i"(c), "i"(32 - (c)));                                              \
    ta;                                                                        \
  })
#else
#define shrd_adc(a, b, c) /* (((u64)b<<32)+a)>>sr, and return retsult + CF */  \
  (shr_adc((a), (c)) + ((uint32_t)(b) << (32 - (c))))
#endif
#define sard_adc(a, b, c) (sar_adc((a), (c)) + ((int32_t)(b) << (32 - (c))))

/**
 * 无符号a与b相乘的结果向右移动sr位(或向左移-sr位)，返回低32位
 */
TF_INLINE uint32_t mullo_shr(uint32_t a, uint32_t b, const int sr) {
  assert(sr > -32 && sr < 32);
  uint64_t r = (uint64_t)a * b;
  if (sr < 0)
    return (uint32_t)(r << (-sr));
  return (uint32_t)(r >> sr);
}

/**
 * 有符号a与b相乘的结果向右移动sr位(或向左移-sr位)，返回低32位
 */
TF_INLINE int32_t imullo_sar(int32_t a, int32_t b, const int sr) {
  assert(sr > -32 && sr < 32);
  int64_t r = (int64_t)a * b;
  if (sr < 0)
    return (int32_t)(r << (-sr));
  return (int32_t)(r >> sr);
}

/**
 * 无符号a与b相乘的结果向右移动sr位(或向左移-sr位)，返回高32位
 */
TF_INLINE uint32_t mulhi_shr(uint32_t a, uint32_t b, const int sr) {
  assert(sr > -32 && sr < 32);
  uint64_t r = (uint64_t)a * b;
  return (uint32_t)(r >> (32 + sr));
}

/**
 * 有符号a与b相乘的结果向右移动sr位(或向左移-sr位)，返回高32位
 */
TF_INLINE int32_t imulhi_sar(int32_t a, int32_t b, const int sr) {
  assert(sr > -32 && sr < 32);
  int64_t r = (int64_t)a * b;
  return (int32_t)(r >> (32 + sr));
}

/**
 * 无符号a与b相乘的结果向右移动sr位并处理舍入，返回低32位
 */
TF_INLINE uint32_t mul_shrd_adc(uint32_t a, uint32_t b, int sr) {
  assert(sr > 0 && sr < 32);
  uint64_t r = (uint64_t)a * b;
  return shrd_adc(r, r >> 32, sr);
}

/**
 * 有符号a与b相乘的结果向右移动sr位并处理舍入，返回低32位
 */
TF_INLINE int32_t imul_sard_adc(int32_t a, int32_t b, int sr) {
  assert(sr > 0 && sr < 32);
  int64_t r = (int64_t)a * b;
  return sard_adc(r, r >> 32, sr);
}

/* 公式数据使用matlab计算，对于double和int之类的按内容互转，使用自定义matlab函数完成
 */

/**
 * 浮点绝对值
 */
double tfabs(double x) {
  int64_t ix = *(int64_t *)&x;
  ix &= 0x7fffffffffffffffLL;
  return *(double *)&ix;
}

/**
 * 浮点绝对值
 */
float tfabsf(float x) {
  int32_t ix = *(int32_t *)&x;
  ix &= 0x7fffffff;
  return *(float *)&ix;
}

/**
 * 将x拆分为x=a+n*log(2),n为整数,并返回exp(a)-1,a in [-log(2)/2,log(2)/2]
 * 仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE double _expm1_lim(double x, int *n) {
  static const double log2e = 1.4426950408889634074;  // 1/log(2)
  static const double log2s = 0.69314718055989033019; // 0x3fe62e42fefa3800
  /* 如何确保log2s+log2t==log(2), matlab计算:(必须用str2sym计算)
  log2t=vpa(str2sym('log(2)-int2double(0x3fe62e42fefa3800)')) */
  static const double log2t = 5.4979230187083711747e-14; // log(2)-log2s
  double px, qx, xx;
  /* 计算x与log(2)的商（整数）和余数（浮点） */
  xx = x * log2e;
  int nn;
  ROUND(nn, xx); // n为商
  *n = nn;
  xx = nn;
  /* 这种计算方式确保结果精确 */
  x -= xx * log2s;
  x -= xx * log2t; // x为余数
  static const double PQ[] = {
      /* P的系数部分 */
      42.03125295928045445,  // 42
      10093.135126288721961, // 10080
      333112.91791346222314, // 332640
      /* Q的系数部分 */
      //                       1
      840.93803276294094188, // 840
      75705.089904821137737, // 75600
      666225.83582692444637  // 665280
  };                         // P和Q的系数列，应该微调
  /* 运用Padé Approximant多项式逼近 */
  xx = x * x;
  px = x * (PQ[2] + xx * (PQ[1] + xx * PQ[0]));
  qx = PQ[5] + xx * (PQ[4] + xx * (PQ[3] + xx));
  x = (px + px) / (qx - px);
  return x;
}

/**
 * 将x拆分为x=a+n*log(2),n为整数,并返回exp(a)-1,a in [-log(2)/2,log(2)/2]
 * 仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE float _expm1f_lim(float x, int *n) {
  static const float log2e = 1.442695041f;
  static const float log2s = 0.6931457520f;   // 0x3f317200
  static const float log2t = 1.428606820e-6f; // log(2)-log2s
  float xx;
  /* 计算x与log(2)的商（整数）和余数（浮点） */
  xx = x * log2e;
  int nn;
  ROUNDF(nn, xx); // n为商
  *n = nn;
  xx = (float)nn;
  /* 这种计算方式确保结果精确 */
  x -= xx * log2s;
  x -= xx * log2t; // x为余数
#if TF_DIV_SLOW
  static const float A[] = {
      0.00137529782f,  // 1/720
      0.008366136305f, // 1/120
      0.04166920787f,  // 1/24
      0.1666654263f,   // 1/6
      0.4999999105f    // 1/2
  };                   // 已理论最优调参
  x += x * x * (A[4] + x * (A[3] + x * (A[2] + x * (A[1] + x * A[0]))));
#else
  /* 运用Padé Approximant多项式逼近 */
  static const float PQ[] = {
      /* P为x */
      /* Q的系数部分 */
      -0.001383849335f, // -1/720
      0.08333316536f,   // 1/12
      -0.5000000000f    // -1/2
  }; // P和Q的系数列，已理论最优调参，还需手动优化
  x = x / (1.0f + x * (PQ[2] + x * (PQ[1] + x * x * PQ[0])));
#endif
  return x;
}

/**
 * 计算自然指数函数
 * exp(x)=exp(a+n*log(2))=exp(a)+pow(2,n)
 * exp(a)=1+2*x*P(x*x)/(Q(x*x)-x*P(x*x)), PQ coef is calc by:
 * pade((exp(x)-1),'Order',[5,6])
 */
double tfexp(double x) {
  /* 如果结果是非规格化数，则直接返回0而不进行计算 */
  static const double underflow =
      -708.39641853226407875; // -1022*log(2),允许非规格化数则值应为-1023*log(2)
  static const double overflow = 709.78271289338399684; // 1024*log(2)
  int64_t ix;
  int n;
  // x = x < overflow ? x : overflow;
  // x = x > underflow ? x : underflow;
  if (!(x < overflow)) {
    if (x != x)
      return TF_NAN;
    return TF_INFINITY;
  }
  if (x <= underflow)
    return 0.0;
  x = 1.0 + _expm1_lim(x, &n);
  /* 上阶码 */
  ix = *(int64_t *)&x;
  ix += ((uint64_t)n << 52);
  return *(double *)&ix;
}

/**
 * 计算自然指数函数
 * exp(x)=exp(a+n*log(2))=exp(a)+pow(2,n)
 * exp(a)=1+2*x*P(x*x)/(Q(x*x)-x*P(x*x)), PQ coef is calc by:
 * pade((exp(x)-1),'Order',[3,3]) exp(a)=1+x+x*x*A(x), taylor expand, and
 * optimize by vpasolve.
 */
float tfexpf(float x) {
  /* 如果结果是非规格化数，则直接返回0而不进行计算 */
  static const float underflow =
      -87.33654475f; // -126*log(2),允许非规格化数则值应为-127*log(2)
  static const float overflow = 88.72283911f; // 128*log(2)
  int32_t ix, n;
  // x = x < overflow ? x : overflow;
  // x = x > underflow ? x : underflow;
  if (!(x < overflow)) {
    if (x != x)
      return (float)TF_NAN;
    return (float)TF_INFINITY;
  }
  if (x <= underflow)
    return 0.0f;
  x = 1.0f + _expm1f_lim(x, &n);
  /* 上阶码 */
  ix = *(int32_t *)&x;
  ix += (n << 23);
  return *(float *)&ix;
}

/**
 * 计算exp(x)-1
 */
double tfexpm1(double x) {
  /* 如果结果是非规格化数，则直接返回0而不进行计算 */
  static const double underflow =
      -708.39641853226407875; // -1022*log(2),允许非规格化数则值应为-1023*log(2)
  static const double overflow = 709.78271289338399684; // 1024*log(2)
  double xx;
  int64_t ix, ii;
  int n;
  // x = x < overflow ? x : overflow;
  // x = x > underflow ? x : underflow;
  if (!(x < overflow)) {
    if (x != x)
      return TF_NAN;
    return TF_INFINITY;
  }
  if (x <= underflow)
    return -1.0;
  xx = _expm1_lim(x, &n);
  ix = ((uint64_t)n << 52);
  ii = 0x4000000000000000LL - ix;
  ix += 0x3ff0000000000000LL;
  x = (1.0 - *(double *)&ii * 0.5) + xx;
  /* 上阶码 */
  x *= *(double *)&ix;
  return x;
}

/**
 * 计算exp(x)-1
 */
float tfexpm1f(float x) {
  /* 如果结果是非规格化数，则直接返回0而不进行计算 */
  static const float underflow =
      -87.33654475f; // -126*log(2),允许非规格化数则值应为-127*log(2)
  static const float overflow = 88.72283911f; // 128*log(2)
  float xx;
  int32_t ix, n;
  // x = x < overflow ? x : overflow;
  // x = x > underflow ? x : underflow;
  if (!(x < overflow)) {
    if (x != x)
      return (float)TF_NAN;
    return (float)TF_INFINITY;
  }
  if (x <= underflow)
    return -1.0f;
  xx = _expm1f_lim(x, &n);
  n <<= 23;
  ix = 0x40000000 - n;
  n += 0x3f800000;
  x = (1.0f - *(float *)&ix * 0.5f) + xx;
  /* 上阶码 */
  x *= *(float *)&n;
  return x;
}

/**
 * 计算以2为底的指数
 */
double tfexp2(double x) {
  /* 如果结果是非规格化数，则直接返回0而不进行计算 */
  static const double underflow = -1022.0;
  static const double overflow = 1024.0;
  int64_t ix;
  int n;
  // x = x < overflow ? x : overflow;
  // x = x > underflow ? x : underflow;
  if (!(x < overflow)) {
    if (x != x)
      return TF_NAN;
    return TF_INFINITY;
  }
  if (x <= underflow)
    return 0.0;
  double px, qx, xx;
  ROUND(n, x);
  xx = n;
  x -= xx;
  static const double PQ[] = {
      /* P的系数部分 */
      //                     // 1
      874.80022068433019843, // 420/log(2)^2
      65555.96964316346295,  // 15120/log(2)^4
      /* Q的系数部分 */
      43.302513465887873929, // 30/log(2)
      10097.46246813799049,  // 3360/log(2)^3
      189154.54460971871716  // 30240/log(2)^5
  };                         // P和Q的系数列，应该微调
  /* 运用Padé Approximant多项式逼近 */
  xx = x * x;
  px = x * (PQ[1] + xx * (PQ[0] + xx));
  qx = PQ[4] + xx * (PQ[3] + xx * PQ[2]);
  x = 1.0 + 2.0 * px / (qx - px);
  /* 上阶码 */
  ix = *(int64_t *)&x;
  ix += ((uint64_t)n << 52);
  return *(double *)&ix;
}

/**
 * 计算以2为底的指数
 */
float tfexp2f(float x) {
  /* 如果结果是非规格化数，则直接返回0而不进行计算 */
  static const float underflow = -126;
  static const float overflow = 128;
  float xx;
  int32_t ix, n;
  // x = x < overflow ? x : overflow;
  // x = x > underflow ? x : underflow;
  if (!(x < overflow)) {
    if (x != x)
      return (float)TF_NAN;
    return (float)TF_INFINITY;
  }
  if (x <= underflow)
    return 0.0f;
  ROUNDF(n, x);
  xx = (float)n;
  x -= xx;
  static const float A[] = {
      0.0001536337163f, // 1/720*log(2)^6
      0.001339468916f,  // 1/120*log(2)^5
      0.009618365087f,  // 1/24*log(2)^4
      0.05550342317f,   // 1/6*log(2)^3
      0.2402264882f,    // 1/2*log(2)^2
      0.6931471992f     // log(2)
  };                    // 已理论最优调参
  x = x * (A[5] + x * (A[4] + x * (A[3] + x * (A[2] + x * (A[1] + x * A[0])))));
  x += 1.0f;
  /* 上阶码 */
  ix = *(int32_t *)&x;
  ix += (n << 23);
  return *(float *)&ix;
}

/**
 * 计算以10为底的指数
 */
double tfexp10(double x) {
  /* 如果结果是非规格化数，则直接返回0而不进行计算 */
  static const double underflow = -307.65265556858878151; // -1022*lg(2)
  static const double overflow = 308.2547155599167439;    // 1024*lg(2)
  int64_t ix;
  int n;
  // x = x < overflow ? x : overflow;
  // x = x > underflow ? x : underflow;
  if (!(x < overflow)) {
    if (x != x)
      return TF_NAN;
    return TF_INFINITY;
  }
  if (x <= underflow)
    return 0.0;
  static const double lg2v = 3.3219280948873623479;  // 1/lg(2)
  static const double lg2s = 0.30102999566395283182; // 0x3fd34413509f7800
  /* 如何确保lg2s+lg2t==lg(2), matlab计算:(必须用str2sym计算)
  lg2t=vpa(str2sym('log(2)/log(10)-int2double(0x3fd34413509f7800)')) */
  static const double lg2t = 2.8363394551044989398e-14; // lg(2)-lg2s
  double px, qx, xx;
  x *= lg2v;
  ROUND(n, x);
  xx = n;
  x -= xx * lg2s;
  x -= xx * lg2t;
  static const double PQ[] = {
      /* P的系数部分 */
      18.253941227695225153, // 42/log(10)
      826.75917173662033403, // 10080/log(10)^3
      5146.5124911389678449, // 332640/log(10)^5
      /* Q的系数部分 */
      //                     // 1
      158.61074944102648455, // 840/log(10)^2
      2693.1610494636069001, // 75600/log(10)^4
      4470.2039518956239061  // 665280/log(10)^6
  };                         // P和Q的系数列，应该微调
  /* 运用Padé Approximant多项式逼近 */
  xx = x * x;
  px = x * (PQ[2] + xx * (PQ[1] + xx * PQ[0]));
  qx = PQ[5] + xx * (PQ[4] + xx * (PQ[3] + xx));
  x = 1.0 + 2.0 * px / (qx - px);
  /* 上阶码 */
  ix = *(int64_t *)&x;
  ix += ((uint64_t)n << 52);
  return *(double *)&ix;
}

/**
 * 计算以10为底的指数
 */
float tfexp10f(float x) {
  /* 如果结果是非规格化数，则直接返回0而不进行计算 */
  static const float underflow = -37.92977945f; // -126*lg(2)
  static const float overflow = 38.53183944f;   // 128*lg(2)
  float xx;
  int32_t ix, n;
  // x = x < overflow ? x : overflow;
  // x = x > underflow ? x : underflow;
  if (!(x < overflow)) {
    if (x != x)
      return (float)TF_NAN;
    return (float)TF_INFINITY;
  }
  if (x <= underflow)
    return 0.0f;
  static const float lg2v = 3.321928095f;  // 1/lg(2)
  static const float lg2s = 0.3010253906f; // 0x3e9a2000
  /* 如何确保lg2s+lg2t==lg(2), matlab计算:(必须用str2sym计算)
  lg2t=vpa(str2sym('log(2)/log(10)-int2float(0x3e9a2000)')) */
  static const float lg2t = 4.605038981e-6f; // lg(2)-lg2s
  x *= lg2v;
  ROUNDF(n, x);
  xx = (float)n;
  x -= xx * lg2s;
  x -= xx * lg2t;
  static const float A[] = {
      0.2064561869f, // 1/720*log(10)^6
      0.5418558644f, // 1/120*log(10)^5
      1.171283885f,  // 1/24*log(10)^4
      2.034653463f,  // 1/6*log(10)^3
      2.650948848f,  // 1/2*log(10)^2
      2.302585155f   // log(10)
  };                 // 已理论最优调参
  x = x * (A[5] + x * (A[4] + x * (A[3] + x * (A[2] + x * (A[1] + x * A[0])))));
  x += 1.0f;
  /* 上阶码 */
  ix = *(int32_t *)&x;
  ix += (n << 23);
  return *(float *)&ix;
}

/**
 * 计算atanh(x)+n*log(2)/2, x in [-0.1716,0.1716]
 * 仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE double _atanh_pnlog(double x, int n) {
  /* half_log2t取负值可以正负补偿，精度更高 */
  static const double half_log2s = 0.34657359028278733604; // 0x3fd62e42fefb0000
  /* half_log2t=vpa(str2sym('log(2)/2-int2double(0x3fd62e42fefb0000)')) */
  static const double half_log2t =
      -2.8146813279468588876e-12; // log(2)/2-half_log2s
  double y, xx;
  xx = x * x;
  /* 建议修改计算方法,log(1+x)函数在|x|足够小时用多项分式逼近计算,
     可以避免部分除法误差,此外执行并行度更高 */
#if 1 /* 这样计算误差已经足够了 */
  static const double A[] = {
      0.074109598799505832971, // 1/15
      0.076554942176428945648, // 1/13
      0.09091853196950475735,  // 1/11
      0.11111097644860654761,  // 1/9
      0.142857143908097516,    // 1/7
      0.19999999999595248536,  // 1/5
      0.33333333333333901994   // 1/3
  };                           // 已理论最优调参
  y = A[5] + xx * (A[4] + xx * (A[3] + xx * (A[2] + xx * (A[1] + xx * A[0]))));
  y = x * xx * (A[6] + xx * y);
#else /* 这种方式计算精度要高一点，但不会有优势 */
  static const double PQ[] = {
      -0.83488032249937011842, // -82841/99225
      8.2362811791383219955,   // 18161/2205
      -19.520634920634920635,  // -6149/315
      12.862433862433862434,   // 2431/189
      // 1.0,
      -14.666666666666666667, // -44/3
      57.2,                   // 286/5
      -81.714285714285714286, // -572/7
      38.587301587301587302   // 2431/63
  };                          // P和Q的系数列，未优化调参
  double px, qx;
  px = PQ[3] + xx * (PQ[2] + xx * (PQ[1] + xx * PQ[0]));
  qx = PQ[7] + xx * (PQ[6] + xx * (PQ[5] + xx * (PQ[4] + xx)));
  y = x * (xx * px / qx);
#endif
  xx = n;
  y += xx * half_log2t;
  y += x;
  y += xx * half_log2s;
  return y;
}

/**
 * 计算atanh(x)+n*log(2)/2, x in [-0.1716,0.1716]
 * 仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE float _atanh_pnlogf(float x, int n) {
  /* half_log2t取负值可以正负补偿，精度更高 */
  static const float half_log2s = 0.3465805054f; // 0x3eb17300
  /* half_log2t=vpa(str2sym('log(2)/2-int2float(0x3eb17300)')) */
  static const float half_log2t = -6.915091121e-6f; // log(2)/2-half_log2s
  float y, xx;
  xx = x * x;
  /* 这样计算误差已经足够了 */
  static const float A[] = {
      0.1494219998f, // 1/7
      0.1998827605f, // 1/5
      0.3333339562f, // 1/3
  };                 // 已理论最优调参
  y = x * xx * (A[2] + xx * (A[1] + xx * A[0]));
  xx = (float)n;
  y += xx * half_log2t;
  y += x;
  y += xx * half_log2s;
  return y;
}

/**
 * 计算自然对数的1/2
 * log(x)/2=atanh((x-1)/(x+1))=atanh(z)=z+z^3/3+z^5/5+z^7/7+z^9/9...
 * 设x=a*2^n，那么log(x)/2=log(a)/2+n*log(2)/2;
 * 计算log(a)/2, a in [sqrt(2)/2,sqrt(2)], 对应atanh(z), z in [-0.1716,0.1716]
 */
TF_INLINE double half_log(double x) {
  int n;
  int64_t ii, ix = *(int64_t *)&x;
  if (ix < 0)
    return TF_NAN;
  if (ix < 0x10000000000000LL)
    return -TF_INFINITY;
  if (ix >= 0x7ff0000000000000LL)
    return x;
  ii = (ix & 0xfffffffffffffLL) | 0x3fe0000000000000LL;
  if (ii < 0x3fe6a09e667f3bcdLL)
    ii += 0x10000000000000LL;
  n = (ix - ii) >> 52;
  x = *(double *)&ii;
  x = (x - 1.0) / (x + 1.0);
  return _atanh_pnlog(x, n);
}

/**
 * 计算自然对数的1/2
 */
TF_INLINE float half_logf(float x) {
  int n;
  int32_t ii, ix = *(int32_t *)&x;
  if (ix < 0)
    return (float)TF_NAN;
  if (ix < 0x800000)
    return -TF_INFINITY;
  if (ix >= 0x7f800000)
    return x;
  ii = (ix & 0x7fffff) | 0x3f000000;
  if (ii < 0x3f3504f3)
    ii += 0x800000;
  n = (ix - ii) >> 23;
  x = *(float *)&ii;
  x = (x - 1.0f) / (x + 1.0f);
  return _atanh_pnlogf(x, n);
}

/**
 * 计算自然对数
 */
double tflog(double x) {
  x = half_log(x);
  return x + x;
}

/**
 * 计算自然对数
 */
float tflogf(float x) {
  x = half_logf(x);
  return x + x;
}

/**
 * 计算1+x的自然对数的1/2
 * 为了保证当x趋于0时的精度，不能直接用log计算，采用修正项补足误差
 * 反双曲函数需要使用此函数计算
 */
TF_INLINE double half_log1p(double x) {
  double y, xx;
  int64_t ii, ix, n;
  if (x < -1.0)
    return TF_NAN;
  if (x == -1.0)
    return -TF_INFINITY;
  if (!(x < TF_INFINITY))
    return x;
  xx = x + 1.0;               // 存在误差，不能忽略
  y = 0.5 * (x - (xx - 1.0)); // 先将误差修正项除以2
  ix = *(int64_t *)&xx;
  ii = (ix & 0xfffffffffffffLL) | 0x3fe0000000000000LL;
  if (ii < 0x3fe6a09e667f3bcdLL)
    ii += 0x10000000000000LL;
  n = (ix - ii) >> 52; // n in [-1022,1024]
  ix = 0x4000000000000000LL -
       (n << 52); // 计算2/2^n作为修正项的系数，为确保不溢出
  x = *(double *)&ii;
  /* log(org)=log(xx+y)=log(x+y*x/xx)+n*log(2), x/xx=1/2^n */
  y *= *(double *)&ix;                   // 乘以2/2^n作修正项结果
  x = ((x - 1.0) + y) / ((x + 1.0) + y); // 为了保证精度
  return _atanh_pnlog(x, (int)n);
}

/**
 * 计算1+x的自然对数的1/2
 */
TF_INLINE float half_log1pf(float x) {
  int n;
  float y, xx;
  int32_t ii, ix;
  if (x < -1.0f)
    return (float)TF_NAN;
  if (x == -1.0f)
    return -TF_INFINITY;
  if (!(x < TF_INFINITY))
    return x;
  xx = x + 1.0f;
  y = 0.5f * (x - (xx - 1.0f)); // 先将修正项除以2
  ix = *(int32_t *)&xx;
  ii = (ix & 0x7fffff) | 0x3f000000;
  if (ii < 0x3f3504f3)
    ii += 0x800000;
  n = (ix - ii) >> 23; // n in [-126,128]
  ix = 0x40000000 - (n << 23); // 计算2/2^n作为修正项的系数，为确保不溢出
  x = *(float *)&ii;
  y *= *(float *)&ix;                      // 乘以2/2^n作修正项结果
  x = ((x - 1.0f) + y) / ((x + 1.0f) + y); // 为了保证精度
  return _atanh_pnlogf(x, n);
}

/**
 * 计算1+x的自然对数
 */
double tflog1p(double x) {
  x = half_log1p(x);
  return x + x;
}

/**
 * 计算1+x的自然对数
 */
float tflog1pf(float x) {
  x = half_log1pf(x);
  return x + x;
}

/**
 * 计算以2为底的对数
 */
double tflog2(double x) {
  int n;
  int64_t ii, ix = *(int64_t *)&x;
  if (ix < 0)
    return TF_NAN;
  if (ix < 0x10000000000000LL)
    return -TF_INFINITY;
  if (ix >= 0x7ff0000000000000LL)
    return x;
  ii = (ix & 0xfffffffffffffLL) | 0x3fe0000000000000LL;
  if (ii < 0x3fe6a09e667f3bcdLL)
    ii += 0x10000000000000LL;
  n = (ix - ii) >> 52;
  x = *(double *)&ii;
  x = (x - 1.0) / (x + 1.0);
  double y, xx;
  xx = x * x;
  /* 这样计算误差已经足够了 */
  static const double A[] = {
      0.2143449998764819823,  // 2/15/log(2)
      0.22083325179073622464, // 2/13/log(2)
      0.26233806234723070175, // 2/11/log(2)
      0.32059844724820697341, // 2/9/log(2)
      0.41219858694883895896, // 2/7/log(2)
      0.57707801633834106228, // 2/5/log(2)
      0.96179669392601024936, // 2/3/log(2)
      2.8853900817779268911   // 2/log(2)
  };                          // 已理论最优调参
  y = A[5] + xx * (A[4] + xx * (A[3] + xx * (A[2] + xx * (A[1] + xx * A[0]))));
  y = x * (A[7] + xx * (A[6] + xx * y));
  xx = n;
  y += xx;
  return y;
}

/**
 * 计算以2为底的对数
 */
float tflog2f(float x) {
  int n;
  int32_t ii, ix = *(int32_t *)&x;
  if (ix < 0)
    return (float)TF_NAN;
  if (ix < 0x800000)
    return -TF_INFINITY;
  if (ix >= 0x7f800000)
    return x;
  ii = (ix & 0x7fffff) | 0x3f000000;
  if (ii < 0x3f3504f3)
    ii += 0x800000;
  n = (ix - ii) >> 23;
  x = *(float *)&ii;
  x = (x - 1.0f) / (x + 1.0f);
  float y, xx;
  xx = x * x;
  /* 这样计算误差已经足够了 */
  static const float A[] = {
      0.4329028021f, // 2/7/log(2)
      0.576646369f,  // 2/5/log(2)
      0.9617999711f, // 2/3/log(2)
      2.885390075f   // 2/log(2)
  };                 // 已理论最优调参
  y = x * (A[3] + xx * (A[2] + xx * (A[1] + xx * A[0])));
  xx = (float)n;
  y += xx;
  return y;
}

/**
 * 计算以10为底的对数
 */
double tflog10(double x) {
  int n;
  int64_t ii, ix = *(int64_t *)&x;
  if (ix < 0)
    return TF_NAN;
  if (ix < 0x10000000000000LL)
    return -TF_INFINITY;
  if (ix >= 0x7ff0000000000000LL)
    return x;
  ii = (ix & 0xfffffffffffffLL) | 0x3fe0000000000000LL;
  if (ii < 0x3fe6a09e667f3bcdLL)
    ii += 0x10000000000000LL;
  n = (ix - ii) >> 52;
  x = *(double *)&ii;
  x = (x - 1.0) / (x + 1.0);
  /* lg2t取负值可以正负补偿，精度更高 */
  static const double lg2s = 0.30102999566588550806; // 0x3fd3441350a00000
  /* lg2t=vpa(str2sym('log(2)/log(10)-int2double(0x3fd3441350a00000)')) */
  static const double lg2t = -1.9043128467164275162e-12; // log(2)/log(10)-lg2s
  double y, xx;
  xx = x * x;
  /* 这样计算误差已经足够了 */
  static const double A[] = {
      0.064524274383413412714, // 2/15/log(10)
      0.066477432829028184725, // 2/13/log(10)
      0.078971625770884076512, // 2/11/log(10)
      0.096509749185006836825, // 2/9/log(10)
      0.12408413884190815165,  // 2/7/log(10)
      0.17371779275610965712,  // 2/5/log(10)
      0.28952965460217827714,  // 2/3/log(10)
      0.86858896380650356735   // 2/log(10)
  };                           // 已理论最优调参
  y = A[5] + xx * (A[4] + xx * (A[3] + xx * (A[2] + xx * (A[1] + xx * A[0]))));
  y = x * xx * (A[6] + xx * y);
  xx = n;
  y += xx * lg2t;
  y += x * A[7];
  y += xx * lg2s;
  return y;
}

/**
 * 计算以10为底的对数
 */
float tflog10f(float x) {
  int n;
  int32_t ii, ix = *(int32_t *)&x;
  if (ix < 0)
    return (float)TF_NAN;
  if (ix < 0x800000)
    return -TF_INFINITY;
  if (ix >= 0x7f800000)
    return x;
  ii = (ix & 0x7fffff) | 0x3f000000;
  if (ii < 0x3f3504f3)
    ii += 0x800000;
  n = (ix - ii) >> 23;
  x = *(float *)&ii;
  x = (x - 1.0f) / (x + 1.0f);
  /* lg2t取负值可以正负补偿，精度更高 */
  static const float lg2s = 0.30103302f; // 0x3e9a2100
  /* lg2t=vpa(str2sym('log(2)/log(10)-int2float(0x3e9a2100)')) */
  static const float lg2t = -3.02435555e-6f; // log(2)/log(10)-lg2s
  float y, xx;
  xx = x * x;
  /* 这样计算误差已经足够了 */
  static const float A[] = {
      0.1303167286f, // 2/7/log(10)
      0.1735878539f, // 2/5/log(10)
      0.2895306411f, // 2/3/log(10)
      0.8685889618f  // 2/log(10)
  };                 // 已理论最优调参
  y = x * xx * (A[2] + xx * (A[1] + xx * A[0]));
  xx = (float)n;
  y += xx * lg2t;
  y += x * A[3];
  y += xx * lg2s;
  return y;
}

/**
 * 反双曲正切
 */
double tfatanh(double x) {
#if 0 /* 适合SIMD的算法 */
  /* 以下函数无法保证在-1附近的误差有足够精度，由于除法截断，故不适用 */
  // return half_log1p(2.0 * x / (1.0 - x));
  /* 和log1p一样，采用误差修正项 */
  double y, xx;
  int64_t n, ii, ix = *(int64_t *)&x;
  ii = ix & 0x7fffffffffffffffLL;
  if (ii == 0x3ff0000000000000LL) {
    ix |= 0x7ff0000000000000LL; // TF_INFINITY
    return *(double *)&ix;
  }
  if (ii > 0x3ff0000000000000LL) {
    ix |= 0x7ff8000000000000LL; // TF_NAN
    return *(double *)&ix;
  }
  int64_t sign = ix & 0x8000000000000000LL;
  x = *(double *)&ii;
  x = 2.0 * x / (1.0 - x);
  xx = x + 1.0;               // 存在误差，不能忽略
  y = 0.5 * (x - (xx - 1.0)); // 先将误差修正项除以2
  ix = *(int64_t *)&xx;
  ii = (ix & 0xfffffffffffffLL) | 0x3fe0000000000000LL;
  if (ii < 0x3fe6a09e667f3bcdLL)
    ii += 0x10000000000000LL;
  n = (ix - ii) >> 52; // n in [-1022,1024]
  ix = 0x4000000000000000LL -
       (n << 52); // 计算2/2^n作为修正项的系数，为确保不溢出
  x = *(double *)&ii;
  /* log(org)=log(xx+y)=log(x+y*x/xx)+n*log(2), x/xx=1/2^n */
  y *= *(double *)&ix;                   // 乘以2/2^n作修正项结果
  x = ((x - 1.0) + y) / ((x + 1.0) + y); // 为了保证精度
  y = _atanh_pnlog(x, (int)n);
  sign |= *(int64_t *)&y;
  return *(double *)&sign;
#else /* 适合单数据流算法 */
  int64_t ii, ix = *(int64_t *)&x;
  ii = ix & 0x7fffffffffffffffLL;
  if (ii == 0x3ff0000000000000LL) {
    ix |= 0x7ff0000000000000LL; // TF_INFINITY
    return *(double *)&ix;
  }
  if (ii > 0x3ff0000000000000LL) {
    ix |= 0x7ff8000000000000LL; // TF_NAN
    return *(double *)&ix;
  }
  if (ii > 0x3fe0000000000000LL) { // 绝对值大于0.5则间接计算
    int n;
    x = (1.0 + x) / (1.0 - x);
    ix = *(int64_t *)&x;
    ii = (ix & 0xfffffffffffffLL) | 0x3fe0000000000000LL;
    if (ii < 0x3fe6a09e667f3bcdLL)
      ii += 0x10000000000000LL;
    n = (ix - ii) >> 52;
    x = *(double *)&ii;
    x = (x - 1.0) / (x + 1.0);
    return _atanh_pnlog(x, n);
  }
  if (ii > 0x3fc72db93e0ce866LL) { // 绝对值大于p=0.18108287362867087921
    /* ry = int2double(0x3fd671541846d727), r0 = atanh(ry) */
    double r0 = 0.3662040962231781771, ry = 0.35066702242635233722;
    int64_t sign = ix & 0x8000000000000000LL;
    *(int64_t *)&r0 ^= sign;
    *(int64_t *)&ry ^= sign;
    x = (x - ry) / (1.0 - ry * x);
    return r0 + _atanh_pnlog(x, 0);
  }
  return _atanh_pnlog(x, 0);
#if 0
  /* 这里为了精度，使用多项分式逼近，因为幂级数展开的误差过大 */
  static const double PQ[] = {
      /* P的系数部分 */
      -0.854074331929669305196, // -414713/480249
      12.0426861384072379242,   // 198796/14553
      -46.1252884198732692637,  // -39338/693
      65.4566728676544377376,   // 890188/10395
      -30.9092539379866942570,  // -4199/99
      /* Q的系数部分 */
      // 1.00000000000000000000,// 1
      -19.5638849376911654834, // -65/3
      108.938092147140262656,  // 130
      -249.839401325893582852, // -2210/7
      252.006675691344555838,  // 20995/63
      -92.7277618139601130017  // -4199/33
  };                           // P和Q的系数列，未优化调参
  double px, qx, xx;
  xx = x * x;
  px = PQ[4] + xx * (PQ[3] + xx * (PQ[2] + xx * (PQ[1] + xx * PQ[0])));
  qx = PQ[9] + xx * (PQ[8] + xx * (PQ[7] + xx * (PQ[6] + xx * (PQ[5] + xx))));
  x += x * xx * px / qx;
  return x;
#endif
#endif
}

/**
 * 反双曲正切
 */
float tfatanhf(float x) {
#if 0 /* 适合SIMD的算法 */
  /* 以下函数无法保证在-1附近的误差有足够精度，由于除法截断，故不适用 */
  // return half_log1pf(2.0f * x / (1.0f - x));
  /* 和log1pf一样，采用误差修正项 */
  int n;
  float y, xx;
  int32_t ii, ix = *(int32_t *)&x;
  ii = ix & 0x7fffffff;
  if (ii == 0x3f800000) {
    ix |= 0x7f800000; // TF_INFINITY
    return *(float *)&ix;
  }
  if (ii > 0x3f800000) {
    ix |= 0x7fc00000; // TF_NAN
    return *(float *)&ix;
  }
  int32_t sign = ix & 0x80000000;
  x = *(float *)&ii;
  x = 2.0f * x / (1.0f - x);
  xx = x + 1.0f;               // 存在误差，不能忽略
  y = 0.5f * (x - (xx - 1.0f)); // 先将误差修正项除以2
  ix = *(int32_t *)&xx;
  ii = (ix & 0x7fffff) | 0x3f000000;
  if (ii < 0x3f3504f3)
    ii += 0x800000;
  n = (ix - ii) >> 23; // n in [-126,128]
  ix = 0x40000000 - (n << 23); //计算2/2^n作为修正项的系数，为确保不溢出
  x = *(float *)&ii;
  /* log(org)=log(xx+y)=log(x+y*x/xx)+n*log(2), x/xx=1/2^n */
  y *= *(float *)&ix;                   // 乘以2/2^n作修正项结果
  x = ((x - 1.0f) + y) / ((x + 1.0f) + y); // 为了保证精度
  y = _atanh_pnlogf(x, n);
  sign |= *(int32_t *)&y;
  return *(float *)&sign;
#else /* 适合单数据流算法 */
  int32_t ii, ix = *(int32_t *)&x;
  ii = ix & 0x7fffffff;
  if (ii == 0x3f800000) {
    ix |= 0x7f800000; // TF_INFINITY
    return *(float *)&ix;
  }
  if (ii > 0x3f800000) {
    ix |= 0x7fc00000; // TF_NAN
    return *(float *)&ix;
  }
  if (ii > 0x3e99999a) { // 绝对值大于0.3间接计算
    int n;
    x = (1.0f + x) / (1.0f - x);
    ix = *(int32_t *)&x;
    ii = (ix & 0x7fffff) | 0x3f000000;
    if (ii < 0x3f3504f3)
      ii += 0x800000;
    n = (ix - ii) >> 23; // n in [-126,128]
    x = *(float *)&ii;
    x = (x - 1.0f) / (x + 1.0f);
    return _atanh_pnlogf(x, n);
  }
  // 可以直接计算，并允许误差
  static const float A[] = {
      0.1644190194f, // 1/7
      0.1987951143f, // 1/5
      0.3333531336f, // 1/3
  };                 // 已理论最优调参
  float xx = x * x;
  x += x * xx * (A[2] + xx * (A[1] + xx * A[0]));
  return x;
#endif
}

/**
 * 快速开平方算法，仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE double quik_sqrt(double x) {
#if USE_SQRT_INTRINS
  return SQRT(x);
#else
  /* 最快的开方算法 */
  // 第一步,取方根倒数估值
  int64_t ix = *(int64_t *)&x;
#if TF_USE_SUBNORM
  int64_t n = 0;
  while (ix < 0x10000000000000LL) {
    ix <<= 1;
    n -= 0x10000000000000LL;
  };
  ix += n;
#endif
  ix = 0x5fe6ec85e7a5ce2aLL - (ix >> 1); // 误差3.4212812659e-02
  double zx, rx = *(double *)&ix;
  zx = rx * (rx * x); // 第二步,五阶快速开方根迭代,误差1.6288e-09
  static const double A[6] = {-6.1175639740611562052, 15.740800259772506609,
                              -22.041641246618643254, 18.363982156884348475,
                              -11.01126632556096694,  -0.24596076301606897641};
  rx = (A[4] + zx * (A[3] + zx * (A[2] + zx * (A[1] + zx * (A[0] + zx))))) *
       A[5] * rx;
#if TF_HIGH_PRECISION // 0.5ulps,不考虑运算精度则理论上正确舍入,但无法验证(数据量过大)
  // zx = rx * x; return 0.5 * (zx + x / zx);//除法舍入导致0.75ulps误差,不采用
  rx += -0.5 * rx * (-1.0 + rx * (rx * x));
  dualdouble rx2 = dmul(x, rx);
  double tx = -0.5 * rx2.hi;
  zx = fmuladd(rx2.hi, rx, -1.0);
  return rx2.hi + (zx * tx - rx2.lo * (-1.0 - rx * tx));
#else                 // 1.231ulps,0.8520ulps
  /* err is 1 ulps, more precision by use extention-float */
  zx = rx * x; // 第三步,方根迭代返回,误差-3.9794e-18
#if TF_USE_SUBNORM
  return -0.5 * zx * (-1.0 + zx * rx) + zx; // 1.231ulps
#else
  return -0.5 * rx * (zx * zx - x) + zx;  // 0.8520ulps
#endif
#endif
#endif
}

/**
 * 快速开平方算法，仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE float quik_sqrtf(float x) {
#if USE_SQRT_INTRINS
  return SQRTF(x);
#else
  /* 最快的开方算法 */
  // 第一步,取方根倒数估值
  int32_t ix = *(int32_t *)&x;
#if TF_USE_SUBNORM
  int32_t n = 0;
  while (ix < 0x800000) {
    ix <<= 1;
    n -= 0x800000;
  };
  ix += n;
#endif
  ix = 0x5f37642f - (ix >> 1); // 误差3.4212837634e-02
  float zx, rx = *(float *)&ix;
  zx = rx * (rx * x); // 第二步,二阶快速开方根迭代,误差4.0626e-05
  rx = 0.3748118208f * rx * (5.004350971f + zx * (-3.336343317f + zx));
#if TF_HIGH_PRECISION // 0.5ulps,通过所有测试是正确舍入的
  // zx = rx * x; return 0.5f * (zx + x / zx);//除法舍入导致0.75ulps误差,不采用
  rx += -0.5f * rx * (-1.0f + rx * (rx * x));
  dualfloat rx2 = dmulf(x, rx);
  float tx = -0.5f * rx2.hi;
  zx = fmuladdf(rx2.hi, rx, -1.0f);
  return rx2.hi + (zx * tx - rx2.lo * (-1.0f - rx * tx));
#else                 // 1.211ulps,0.8553ulps
  /* err is 1 ulps, more precision by use extention-float */
  zx = rx * x; // 第三步,方根迭代返回,误差-2.4757e-09
#if TF_USE_SUBNORM
  return -0.5f * zx * (-1.0f + zx * rx) + zx; // 1.211ulps
#else
  return -0.5f * rx * (zx * zx - x) + zx; // 0.8553ulps
#endif
#endif
#endif
}

/**
 * 反双曲正弦
 */
double tfasinh(double x) {
  double xx;
  int64_t ix, sign;
  ix = *(int64_t *)&x;
  sign = ix & 0x8000000000000000LL;
  ix &= 0x7fffffffffffffffLL;
  x = *(double *)&x;
  xx = x * x;
  x = tflog1p(x + xx / (1.0 + quik_sqrt(1.0 + xx)));
  sign |= *(int64_t *)&x;
  return *(double *)&sign;
}

/**
 * 反双曲正弦
 */
float tfasinhf(float x) {
  float xx;
  int32_t ix, sign;
  ix = *(int32_t *)&x;
  sign = ix & 0x80000000;
  ix &= 0x7fffffff;
  x = *(float *)&x;
  xx = x * x;
  x = tflog1pf(x + xx / (1.0f + quik_sqrtf(1.0f + xx)));
  sign |= *(int32_t *)&x;
  return *(float *)&sign;
}

/**
 * 反双曲余弦
 */
double tfacosh(double x) {
  double xm = x - 1.0;
  return tflog1p(xm + quik_sqrt(xm * (x + 1.0)));
}

/**
 * 反双曲余弦
 */
float tfacoshf(float x) {
  float xm = x - 1.0f;
  return tflog1pf(xm + quik_sqrtf(xm * (x + 1.0f)));
}

/**
 * 双曲正弦
 */
double tfsinh(double x) {
  double xx;
  int64_t ix, ii, sign;
  int n;
  ix = *(int64_t *)&x;
  sign = ix & 0x8000000000000000LL;
  ix &= 0x7fffffffffffffffLL;
  if (ix >= 0x7ff0000000000000LL)
    return x;
  if (ix >= 0x408633ce8fb9f87eLL) { // 1025*log(2)
    sign |= 0x7ff0000000000000LL;
    return *(double *)&sign;
  }
  xx = _expm1_lim(x, &n);
  ix = ((uint64_t)n << 52);
  ii = 0x4000000000000000LL - ix;
  ix += 0x3ff0000000000000LL;
  xx += (1.0 - *(double *)&ii * 0.5);
  /* 上阶码 */
  xx *= *(double *)&ix;
  x = _expm1_lim(-x, &n);
  ix = ((uint64_t)n << 52);
  ii = 0x4000000000000000LL - ix;
  ix += 0x3ff0000000000000LL;
  x += (1.0 - *(double *)&ii * 0.5);
  /* 上阶码 */
  x *= *(double *)&ix;
  x = 0.5 * (xx - x);
  sign |= *(int64_t *)&x;
  return *(double *)&sign;
}

/**
 * 双曲正弦
 */
float tfsinhf(float x) {
  float xx;
  int32_t ix, ii, sign;
  int n;
  ix = *(int32_t *)&x;
  sign = ix & 0x80000000;
  ix &= 0x7fffffff;
  if (ix >= 0x7f800000)
    return x;
  if (ix >= 0x42b2d4fc) { // 129*log(2)
    sign |= 0x7f800000;
    return *(float *)&sign;
  }
  xx = _expm1f_lim(x, &n);
  ix = ((int32_t)n << 23);
  ii = 0x40000000 - ix;
  ix += 0x3f800000;
  xx += (1.0f - *(float *)&ii * 0.5f);
  /* 上阶码 */
  xx *= *(float *)&ix;
  x = _expm1f_lim(-x, &n);
  ix = ((int32_t)n << 23);
  ii = 0x40000000 - ix;
  ix += 0x3f800000;
  x += (1.0f - *(float *)&ii * 0.5f);
  /* 上阶码 */
  x *= *(float *)&ix;
  x = 0.5f * (xx - x);
  sign |= *(int32_t *)&x;
  return *(float *)&sign;
}

/**
 * 双曲余弦
 */
double tfcosh(double x) {
  int64_t ix;
  double xx;
  int n;
  ix = *(int64_t *)&x;
  ix &= 0x7fffffffffffffffLL;
  if (ix > 0x7ff0000000000000LL)
    return x;
  if (ix >= 0x408633ce8fb9f87eLL) // 1025*log(2)
    return TF_INFINITY;
  xx = 1.0 + _expm1_lim(x, &n);
  ix = ((int32_t)n << 23);
  /* 上阶码 */
  ix += 0x3ff0000000000000LL;
  xx *= *(double *)&ix;
  x = 1.0 + _expm1_lim(-x, &n);
  ix = ((int32_t)n << 23);
  /* 上阶码 */
  ix += 0x3ff0000000000000LL;
  x *= *(double *)&ix;
  return 0.5 * (x + xx);
}

/**
 * 双曲余弦
 */
float tfcoshf(float x) {
  float xx;
  int32_t ix;
  int n;
  ix = *(int32_t *)&x;
  ix &= 0x7fffffff;
  if (ix > 0x7f800000)
    return x;
  if (ix >= 0x42b2d4fc) // 129*log(2)
    return TF_INFINITY;
  xx = 1.0f + _expm1f_lim(x, &n);
  ix = ((int32_t)n << 23);
  /* 上阶码 */
  ix += 0x3f800000;
  xx *= *(float *)&ix;
  x = 1.0f + _expm1f_lim(-x, &n);
  ix = ((int32_t)n << 23);
  /* 上阶码 */
  ix += 0x3f800000;
  x *= *(float *)&ix;
  return 0.5f * (x + xx);
}

/**
 * 双曲正切
 */
double tftanh(double x) {
  int64_t ix, sign;
  int n;
  ix = *(int64_t *)&x;
  sign = ix & 0x8000000000000000LL;
  ix |= 0x8000000000000000LL;
  if (ix > (int64_t)0xfff0000000000000LL)
    return x;
  if (ix >= (int64_t)0xc0330fc1931f09c9LL) { // 27.5*log(2)
    sign |= 0x3ff0000000000000LL;
    return *(double *)&sign;
  }
  x = *(double *)&ix * 2.0;
  x = _expm1_lim(x, &n);
  /* 上阶码 */
  ix = *(int64_t *)&x;
  ix += ((uint64_t)n << 52);
  x = *(double *)&ix;
  x = x / (-2.0 - x); // 这样计算精度高
  sign |= *(int64_t *)&x;
  return *(double *)&sign;
}

/**
 * 双曲正切
 */
float tftanhf(float x) {
  int32_t ix, sign;
  int n;
  ix = *(int32_t *)&x;
  sign = ix & 0x80000000;
  ix |= 0x80000000;
  if (ix > (int32_t)0xff800000)
    return x;
  if (ix >= (int32_t)0xc1102cb3) { // 13*log(2)
    sign |= 0x3f800000;
    return *(float *)&sign;
  }
  x = *(float *)&ix * 2.0f;
  x = _expm1f_lim(x, &n);
  /* 上阶码 */
  ix = *(int32_t *)&x;
  ix += ((int32_t)n << 23);
  x = *(float *)&ix;
  x = x / (-2.0f - x); // 这样计算精度高
  sign |= *(int32_t *)&x;
  return *(float *)&sign;
}

/**
 * 计算xv mod pi/2,商存储在n,余数存储在xv,返回是否精确计算,
 * x in (-3373259426.915902903,3373259425.3451065763), else false is return.
 * 仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE bool reduce_quarter_pi(double *xv, int32_t *n) {
  static const double r2pi = 0.63661977236758134308; // 2/pi
  /* pi_r2h = int2double(0x3ff921fb00000000) */
  static const double pi_r2h = 1.5707960128784179688; // pi/2 high
  /* pi_r2m = int2double(0x3e95110b00000000) */
  static const double pi_r2mh = 3.1391641641675960273e-7; // pi/2 mid-high
  /* pi_r2m = int2double(0x3d31846980000000) */
  static const double pi_r2ml = 6.2233719696699885127e-14; // pi/2 mid-low
  /* pi_r2l = pi/2 - pi_r2h - pi_r2mh - pi_r2ml */
  static const double pi_r2l = 2.022266160358745071e-21; // pi/2 low
  double x = *xv, qx = x * r2pi;
  /* x not in (-3373259426.915902903,3373259425.3451065763) and can't reduce */
  if (qx >= 2147483647.5 || qx <= -2147483648.5)
    return false;
  static const double cvt_cst = 6755399441055744.0;
  qx += cvt_cst;
  *n = (int32_t)(*(int64_t *)&qx);
  qx -= cvt_cst;
  x -= pi_r2h * qx;
  x -= pi_r2mh * qx;
  x -= pi_r2ml * qx;
  x -= pi_r2l * qx;
  *xv = x;
  return true;
}

/**
 * 计算sin(x), x in [-pi/4,pi/4]
 * 仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE double _sine_lim(double x) {
  static const double A[] = {
      1.5901932754310654885e-10, // 1/6227020800
      -2.5050844763839947568e-8, // -1/39916800
      2.755731423057249163e-6,   // 1/362880
      -1.9841269831309294808e-4, // -1/5040
      8.3333333333242086541e-3,  // 1/120
      -1.66666666666666388e-1    // -1/6
  }; // 已理论最优调参,还需微调,峰值误差1.1882e-16
  double xx = x * x;
  double y =
      A[5] + xx * (A[4] + xx * (A[3] + xx * (A[2] + xx * (A[1] + xx * A[0]))));
  return x + x * xx * y;
}

/**
 * 计算cos(x), x in [-pi/4,pi/4]
 * 仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE double _cose_lim(double x) {
#if 1
  double xx = x * x;
#if TF_HIGH_PRECISION // 略大于1ulps
  static const double A[] = {
      -1.1361664560892313469e-11, // -1/87178291200
      2.0875760348115274644e-9,   // 1/479001600
      -2.7557314609246580328e-7,  // -1/3628800
      2.480158729030729606e-5,    // 1/40320
      -1.388888888887532123e-3,   // -1/720
      4.1666666666666605489e-2    // 1/24
  }; // 已理论最优调参,还需微调,峰值误差1.5248e-16
  x = A[5] + xx * (A[4] + xx * (A[3] + xx * (A[2] + xx * (A[1] + xx * A[0]))));
#else // 2ulps
  static const double A[] = {
      2.064511271717373227e-9,   // 1/479001600
      -2.7555523002479591345e-7, // -1/3628800
      2.4801580706602825516e-5,  // 1/40320
      -1.3888888877609576487e-3, // -1/720
      4.1666666666596513003e-2   // 1/24
  }; // 已理论最优调参,还需微调,峰值误差2.3278e-16
  x = A[4] + xx * (A[3] + xx * (A[2] + xx * (A[1] + xx * A[0])));
#endif
  return 1.0 + xx * (-0.5 + xx * x);
#else
  x = _sine_lim(0.5 * x);
  x *= x;
  return 1.0 - (x + x); // 倍角公式,不会有任何优势,故不采用
#endif
}

/**
 * 计算tan(x)或cot(x), x in [-pi/4,pi/4]
 * 仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE double _tane_lim(double x, const int cot_flag) {
  /* SIMD计算tan也像cot那样用二级连分式展开 */
  if (cot_flag) {
    static const double PQ[] = {
        /* P的系数部分 */
        2.6861162066274158786e1,  // 27
        -2.7458810142707822421e3, // -2772
        4.456665377452732447e4,   // 45045
        /* Q的系数部分 */
        // 1.000000000000000000E0,// 1
        -3.7509309625856462538e2, // -378
        1.715097379773679289e4,   // 17325
        -1.3369996132358256652e5  // -135135
    }; // P和Q的系数列,已理论最优调参,还需微调
    /* SIMD的计算方式
     * tan(x)=x/(1+xx*px/qx),峰值误差2.0619e-16
     * cot(x)=(1+xx*px/qx)/x,峰值误差2.3557e-16
     */
    double px, qx, xx = x * x;
    px = PQ[2] + xx * (PQ[1] + xx * PQ[0]);
    qx = PQ[5] + xx * (PQ[4] + xx * (PQ[3] + xx));
#if TF_HIGH_PRECISION
    return (1.0 + xx * px / qx) / x; // 峰值误差2.3557e-16
#else
    return 1.0 / x + x * px / qx; // 峰值误差2.8002e-16
#endif
  }
  static const double PQ[] = {
      /* P的系数部分 */
      -9.6410931574765064422e-1, // -27/28
      9.8559749768298313678e1,   // 99
      -1.5996777165302667243e3,  // -6435/4
      /* Q的系数部分 */
      // 1.00000000000000000000E0,// 1
      -1.1202301215312576469e2, // -225/2
      2.2152925091418541089e3,  // 4455/2
      -4.7990331495908196991e3  // -19305/4
  }; // P和Q的系数列,已理论最优调参,还需微调,峰值误差1.4679e-16
  double px, qx, xx = x * x;
  px = x * xx * (PQ[2] + xx * (PQ[1] + xx * PQ[0]));
  qx = PQ[5] + xx * (PQ[4] + xx * (PQ[3] + xx));
  return x + px / qx; // 以其倒数计算cot(x)精度不可行
}

/**
 * 正弦函数
 */
double tfsin(double x) {
  /* SIMD可以将x缩减至pi/2,然後直接幂级数展开8或9项计算sin值 */
  int32_t n;
  if (x != x)
    return x; // TF_NAN
  if (!reduce_quarter_pi(&x, &n))
    return 0.0; // 精度范围之外
  x = (n & 1) ? _cose_lim(x) : _sine_lim(x);
  if (n & 2)
    x = -x;
  return x;
}

float tfsinf(float x) { return 0.0f; }

/**
 * 余弦函数
 */
double tfcos(double x) {
  /* SIMD可以将x缩减至pi/2,然後化为-sin(x-pi/2)计算
    不能用cos在x=0上展开式计算cos(pi/2)因为相对误差会很大,所以必须转换为sin计算
    q=ROUND(x*r1pi-0.5);
    q=q+q+1;
    x-=0.5*pi_quad*q;
    return -sin(x);
  */
  int32_t n;
  if (x != x)
    return x; // TF_NAN
  if (!reduce_quarter_pi(&x, &n))
    return 0.0; // 精度范围之外
  x = (n & 1) ? _sine_lim(-x) : _cose_lim(x);
  if (n & 2)
    x = -x;
  return x;
}

float tfcosf(float x) { return 0.0f; }

/**
 * 正切函数
 */
double tftan(double x) {
  /* SIMD也是将x缩减至pi/2,然後用多项分式逼近计算 */
  int32_t n;
  if (x != x)
    return x; // TF_NAN
  if (!reduce_quarter_pi(&x, &n))
    return TF_NAN; // 精度范围之外
  x = (n & 1) ? _tane_lim(x, 0) : -_tane_lim(x, 1);
  return x;
}

float tftanf(float x) { return 0.0f; }

/**
 * 计算asin(x), x in [-1/2,1/2]
 * 仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE double _asin_lim(double x) {
  static const double PQ[] = {
      /* P的系数部分 */
      5.8300132101419984374e-3, // 916998508421098379761/184739451946344830130750
      -5.6503215328820585636e-1, // -12016684869700688476/20733945224056658825
      3.5283954725207835148e0,   // 246420234961696736408/62201835672169976475
      -6.5169634619793077041e0,  // -7334967905788244384/940868102604251745
      3.6551495197543571342e0,   // 95865708467514752/20908180057872261
      /* Q的系数部分 */
      // 1.0,                   // 1
      -1.1070640069868623706e1, // -41987135784489168/3497039167491425
      3.7332833404856169392e1,  // 210623679560209968/4895854834487995
      -4.8970684475181636166e1, // -153326421524882944/2591923147670115
      2.1930897118525943432e1   // 191731416935029504/6969393352624087
  }; // P和Q的系数列,已理论最优调参,还需微调,峰值误差1.4643e-16
  double px, qx, xx = x * x;
  px = PQ[4] + xx * (PQ[3] + xx * (PQ[2] + xx * (PQ[1] + xx * PQ[0])));
  qx = PQ[8] + xx * (PQ[7] + xx * (PQ[6] + xx * (PQ[5] + xx)));
  return x + x * xx * (px / qx);
}

/**
 * 计算acos(x), x in [1/2,1]
 * 仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE double _acos_lim(double x) {
  x = 1.0 - x;
#if TF_HIGH_PRECISION // 略大于1ulps
  static const double PQ[] = {
      /* P的系数部分 */
      4.5766650466758442472e-3, // 85212205139/23099021719125
      -1.6880042957458655806e0, // -239564359892/139994071025
      2.1284420983077711485e1,  // 39573414918583/1679928852300
      -7.8467331684105896875e1, // -7763536427647/83996442615
      8.7743557047255221807e1,  // 2426474832509/22399051364
      /* Q的系数部分 */
      // 1.0,                   // 1
      -1.9634432598289792256e1, // -118381097380/5599762841
      1.2278312483905933023e2,  // 783213748590/5599762841
      -3.0559684068988057568e2, // -10190011260156/27998814205
      2.6323067114176210211e2   // 7279424497527/22399051364
  }; // P和Q的系数列,已理论最优调参,还需微调,峰值误差1.6837e-16
  double px, qx;
  px = PQ[4] + x * (PQ[3] + x * (PQ[2] + x * (PQ[1] + x * PQ[0])));
  qx = PQ[8] + x * (PQ[7] + x * (PQ[6] + x * (PQ[5] + x)));
  /* 两步计算误差,应该是最优计算路线,只有计算根式的底和商两步引入了误差 */
  return quik_sqrt((x + x) + x * x * (px / qx));
#else // 2ulps
  /* SIMD用此方法计算,公式:
  acos(x)=2*asin(sqrt((1-x)/2))=ra+ra*a*P(a)/Q(a), a=(1-x)/2,ra=2*sqrt(a). */
  static const double PQ[] = {
      /* P的系数部分 */
      2.9251951556974158601e-3, // 916998508421098379761/369478903892689660261500
      -5.647200256360796288e-1, // -12016684869700688476/20733945224056658825
      7.0396306249262941843e0,  // 492840469923393472816/62201835672169976475
      -2.5970498014980620062e1, // -29339871623152977536/940868102604251745
      2.910353101553034653e1,   // 766925667740118016/20908180057872261
      /* Q的系数部分 */
      // 1.0,                   // 1
      -2.2103611203863140816e1, // -83974271568978336/3497039167491425
      1.4888954122819891028e2,  // 842494718240839872/4895854834487995
      -3.9022550992134194084e2, // -1226611372199063552/2591923147670115
      3.492423721863588717e2    // 3067702670960472064/6969393352624087
  }; // P和Q的系数列,已理论最优调参,还需微调,峰值误差2.1477e-16
  double px, qx, rx = quik_sqrt(x + x);
  px = PQ[4] + x * (PQ[3] + x * (PQ[2] + x * (PQ[1] + x * PQ[0])));
  qx = PQ[8] + x * (PQ[7] + x * (PQ[6] + x * (PQ[5] + x)));
  return rx + rx * x * (px / qx); // 三步计算误差
#endif
}

/**
 * 计算atan(x), x in [-t,t],t=0.32491968837022148064
 * 仅用于其他函数调用，不应直接从外部调用
 */
TF_INLINE double _atan_lim(double x) {
  static const double PQ[] = {
      /* P的系数部分 */
      -7.2241622406049884756e-3, // -256/33075
      -7.2615039623189664227e-1, // -2651/3675
      -2.8242480242957806988e0,  // -286/105
      -2.3990702680319529749e0,  // -143/63
      /* Q的系数部分 */
      // 1.0,                  // 1
      6.7685746091582350939e0, // 33/5
      1.2791070555307181013e1, // 429/35
      7.1972108040960050784e0  // 143/21
  }; // P和Q的系数列,已理论最优调参,还需微调,峰值误差1.2609e-16
  double px, qx, xx = x * x;
  px = PQ[3] + xx * (PQ[2] + xx * (PQ[1] + xx * PQ[0]));
  qx = PQ[6] + xx * (PQ[5] + xx * (PQ[4] + xx));
  return x + x * xx * (px / qx);
}

/**
 * 反正弦
 */
double tfasin(double x) {
  int64_t ix, ii, sign;
  ix = *(int64_t *)&x;
  ii = ix & ~0x8000000000000000LL;
  if (ii > 0x3ff0000000000000LL) // 绝对值大于1
    return TF_NAN;
  if (ii <= 0x3fe0000000000000LL) // 绝对值不大于0.5
    return _asin_lim(x);
  static const double pi_r2h = 1.570796326794896558;      // high of pi/2
  static const double pi_r2l = 6.1232339957205113024e-17; // pi/2-pi_r2h
  sign = ix & 0x8000000000000000LL;
  x = *(double *)&ii;
  x = pi_r2h + (pi_r2l - _acos_lim(x));
  *(int64_t *)&x ^= sign;
  return x;
}

float tfasinf(float x) { return 0.0f; }

/**
 * 反余弦
 */
double tfacos(double x) {
  static const double pi_r2h = 1.570796326794896558;      // high of pi/2
  static const double pi_r2l = 6.1232339957205113024e-17; // pi/2-pi_r2h
  int64_t ix, ii, sign;
  ix = *(int64_t *)&x;
  ii = ix & ~0x8000000000000000LL;
  if (ii > 0x3ff0000000000000LL) // 绝对值大于1
    return TF_NAN;
  if (ii <= 0x3fe0000000000000LL) // 绝对值不大于0.5
    return pi_r2h + (pi_r2l - _asin_lim(x));
  if (ix > 0) // 大于0.5
    return _acos_lim(x);
  // 小于0.5
  static const double pi_h = 3.141592653589793116;      // high of pi
  static const double pi_l = 1.2246467991462850478e-16; // pi-pi_h
  return pi_h + (pi_l - _acos_lim(-x));
}

float tfacosf(float x) { return 0.0f; }

/**
 * 反正切
 */
double tfatan(double x) {
  /* SIMD也可以使用该算法 */
  /* 用公式计算：atan(x)=atan(p)+atan((x-p)/(1+p*x)),x>0 */
  double r0, ry;
  int64_t ix, ii, sign;
  ix = *(int64_t *)&x;
  ii = ix & ~0x8000000000000000LL;
  // 分割点t=0.32491968837022148064,s=1.3763819101787854806
  if (ii <= 0x3fd4cb7bf2d81a6d) // 绝对值小于等于t
    return _atan_lim(x);
  if (ii > 0x7ff0000000000000LL)
    return x; // 返回NAN
  sign = ix & 0x8000000000000000LL;
  if (ii >= 0x43349ff16b9c1e3e) { // 5805358775541310.084
    r0 = 1.5707963267948966192;   // pi/2
    *(int64_t *)&r0 ^= sign;
    return r0; // 提前返回,防止按公式计算导致溢出
  }
  if (ii <= 0x3ff605a909b061de) { // 绝对值小于s
    r0 = 0.62831853427389128265;  // 约等于pi/5
    ry = 0.72654253343834251933;  // 值p,使q=2*p/(1-p*p).
  } else {                        // 绝对值大于s
    r0 = 1.2566370685477825653;   // 约等于2*pi/5
    ry = 3.0776836116516750329;   // 值q约等于sqrt(5+2*sqrt(5))
  }
  *(int64_t *)&ry ^= sign;
  *(int64_t *)&r0 ^= sign;
  x = (x - ry) / (1.0 + ry * x);
  return r0 + _atan_lim(x);
}

float tfatanf(float x) { return 0.0f; }

double tfatan2(double x, double y) { return 0.0; }

float tfatan2f(float x, float y) { return 0.0f; }

/**
 * 开平方
 */
double tfsqrt(double x) {
#if USE_SQRT_INTRINS
  return SQRT(x);
#else
  if (x < 0)
    return TF_NAN;
  if (x > 0 && x < TF_INFINITY)
    return quik_sqrt(x);
  return x;
#endif
}

/**
 * 开平方
 */
float tfsqrtf(float x) {
#if USE_SQRT_INTRINS
  return SQRTF(x);
#else
  if (x < 0)
    return (float)TF_NAN;
  if (x > 0 && x < (float)TF_INFINITY)
    return quik_sqrtf(x);
  return x;
#endif
}

/**
 * 开立方
 */
double tfcbrt(double x) {
  /* 最快的开立方算法 */
  // 第一步,取方根倒数估值
  int64_t ix = *(int64_t *)&x, sign;
  sign = ix & 0x8000000000000000LL;
  ix &= ~0x8000000000000000LL;
  if ((uint64_t)(ix - 1) >= 0x7fefffffffffffffLL)
    return x;
#if TF_USE_SUBNORM
  int64_t n = 0;
  while (ix < 0x10000000000000LL) {
    ix <<= 1;
    n -= 0x10000000000000LL;
  };
  ix += n;
#endif
  // 0x553eee717bc1d4ceLL是不修改迭代系数的最优解
  ix = 0x553ef0ff687cf4ddLL - (ix >> 32) * 0x55555557; // 误差3.4240570776e-02
  ix |= sign;
  double zx, tx, rx = *(double *)&ix; // 为了误差尽可能小,修改部分迭代系数
  zx = rx * rx * (rx * x); // 一阶迭代得到较准确的方根倒数
  rx = 0.33373361630002350824 * rx * (4.0 - zx); // 误差1.2008e-03
  // rx = 0.33333333333333333333 * rx * (4.0 - rx * rx * (rx * x));
  zx = rx * rx * (rx * x); // 第二步,二阶快速开方根迭代,误差-3.2341e-09
  rx = 0.22222158131743543249 * rx *
       (7.0000233610511509819 + zx * (-3.5000103826749442382 + zx));
#if TF_HIGH_PRECISION // 0.5ulps,不考虑运算精度则理论上正确舍入,但无法验证(数据量过大)
  // I don't believe that four-order iteration can provides the correct rounding
  /*static const double A[] = {-5.2000186328804615262, 11.142920929756265591,
                             -13.000083507353885243, 13.000073850030566885,
                             0.14403218541839017194};
  zx = rx * rx * (rx * x); // 迭代误差-9.9939e-15
  rx = (A[3] + zx * (A[2] + zx * (A[1] + zx * (A[0] + zx)))) * A[4] * rx;
  如果使用四阶迭代则最後理论误差-4.9939e-28,不敢保证能正确舍入 */
  rx += -0.33333333333333333333 * rx * (-1.0 + rx * rx * (rx * x)); // 误差2ulps
  dualdouble rx2 = fdfmul(dsqr(rx), x);
  tx = rx2.hi * -0.66666666666666666667;
  zx = fmuladd(rx2.hi, rx, -1.0); // 理论最後误差-9.861e-31
  return rx2.hi + (zx * tx - rx2.lo * (-1.0 - rx * tx));
#else // 1.692ulps
  /* err is 1 ulps, more precision by use extention-float */
  tx = rx * x; // 第三步,方根迭代返回,误差-5.2297e-17
  zx = rx * tx;
  return -0.66666666666666666667 * rx * (zx * zx - tx) + zx;
#endif
}

/**
 * 开立方
 */
float tfcbrtf(float x) {
  /* 最快的开立方算法 */
  // 第一步,取方根倒数估值
  int32_t ix = *(int32_t *)&x, sign;
  sign = ix & 0x80000000;
  ix &= ~0x80000000; // 不进行迭代则最优估值常数是0x54a232a4
  if ((uint32_t)(ix - 1) >= 0x7f7fffff)
    return x;
#if TF_USE_SUBNORM
  int32_t n = 0;
  while (ix < 0x800000) {
    ix <<= 1;
    n -= 0x800000;
  };
  ix += n;
#endif
  // 0x54a21e33是不修改迭代系数的最优解
  ix = 0x54a232a4 - imulhi_sar(ix, 0x55555557, 0); // 误差3.4240517765e-02
  ix |= sign;
  float zx, tx, rx = *(float *)&ix; // 为了误差尽可能小,修改部分迭代系数
  // rx = 0.3333333333f * rx * (4.0f - rx * rx * (rx * x));
  zx = rx * rx * (rx * x); // 第二步,二阶快速开方根迭代,误差-7.5934e-05
  rx = 0.2217018279f * rx * (7.019015731f + zx * (-3.508441878f + zx));
#if TF_HIGH_PRECISION // 0.5ulps,通过所有测试是正确舍入的
  /* 为了正确舍入,必须额外进行一次迭代 */
  rx += -0.3333333333f * rx * (-1.0f + rx * rx * (rx * x)); // 误差2ulps
  dualfloat rx2 = fdfmulf(dsqrf(rx), x);
  tx = rx2.hi * -0.6666666667f;
  zx = fmuladdf(rx2.hi, rx, -1.0f); // 理论最後误差-7.1054e-14
  return rx2.hi + (zx * tx - rx2.lo * (-1.0f - rx * tx));
#else // 1.637ulps
  /* err is 1 ulps, more precision by use extention-float */
  tx = rx * x; // 第三步,方根迭代返回
  zx = rx * tx;
  return -0.6666666667f * rx * (zx * zx - tx) + zx; // 误差-2.8832e-08
#endif
}

/**
 * 整数幂
 */
double tfnpow(double x, const int n) {
  int N = n < 0 ? -n : n;
  double ans = (n & 1) ? x : 1.0;
  while ((N >>= 1) != 0) {
    x *= x;
    if (N & 1) {
      ans *= x;
    }
  }
  if (n < 0)
    ans = 1.0 / ans;
  return ans;
}

/**
 * 整数幂
 */
float tfnpowf(float x, const int n) {
  int N = n < 0 ? -n : n;
  float ans = (n & 1) ? x : 1.0f;
  while ((N >>= 1) != 0) {
    x *= x;
    if (N & 1) {
      ans *= x;
    }
  }
  if (n < 0)
    ans = 1.0f / ans;
  return ans;
}

/*
 * 方根倒数估值方法：
 * 首先假定x>0,n趋于无穷,求解最优估值方法: k - x / n.
 * 当x为float类型时, k = int32((1+1/n)*double(0x3f800000)+ret*2^23),
 * ret = a·ln(2e/(e+2a)), a=1/ln2. ret≈-0.0436774489.
 * 于是有如下函数：
 * 求方根倒数估值，x和n皆为正，n越大则理论误差越小且趋于0.02982120581.
TF_INLINE double Qrcp_nroot(const int n, double x) {
  int64_t ix = *(int64_t*)&x;
  ix = 0x3fef4d18e0162e5dLL - (ix - 0x3ff0000000000000LL) / n;
  return *(double*)&ix;
}
TF_INLINE float Qrcp_nrootf(const int n, float x) {
  int32_t ix = 0x3f7a68c7 - (*(int32_t*)&x - 0x3f800000) / n;
  return *(float*)&ix;
}
 * 通常，不使用上述估值系数，对不同的n使用其最合适的估值系数，
 * 然後进行方根倒数迭代，最後方根迭代以进行开方计算。
*/

/**
 * 整数方根
 */
double tfroot(const int n, double x) {
  if (n == 1)
    return x;
  if (n == 0 || (n < 0 && x == 0))
    return TF_NAN;
  if (n == -1)
    return 1.0 / x;
  double rx, zx, tx, rcp;
  int64_t sign, ii, ix = *(int64_t *)&x;
  int32_t it, im, ni = n < 0 ? -n : n;
  static const struct {
    int64_t itcst;
    double rcp;
  } itcoefs[15] = {
      {0x5fe6eb507fffffffLL, -1.0 / 2},  // 平方根
      {0x553eee7155555557LL, -1.0 / 3},  // 立方根
      {0x4feb090540000000LL, -1.0 / 4},  // 4次方根
      {0x4cb8a49b33333333LL, -1.0 / 5},  // 5次方根
      {0x4a9716342aaaaaabLL, -1.0 / 6},  // 6次方根
      {0x491160e524924925LL, -1.0 / 7},  // 7次方根
      {0x47ed1df020000000LL, -1.0 / 8},  // 8次方根
      {0x4709c9f01c71c71cLL, -1.0 / 9},  // 9次方根
      {0x4653f00a1999999aLL, -1.0 / 10}, // 10次方根
      {0x45bf24731745d174LL, -1.0 / 11}, // 11次方根
      {0x4543274115555555LL, -1.0 / 12}, // 12次方根
      {0x44da3c1713b13b14LL, -1.0 / 13}, // 13次方根
      {0x44804f1b12492492LL, -1.0 / 14}, // 14次方根
      {0x44325ed611111111LL, -1.0 / 15}, // 15次方根
      {0x43ee2d5d10000000LL, -1.0 / 16}  // 16次方根
  };                                     // 迭代系数
  sign = ix & 0x8000000000000000LL;
  ix &= ~0x8000000000000000LL;
  if ((uint64_t)(ix - 1) >= 0x7fefffffffffffffLL)
    return x;
  if (((uint64_t)sign >> 63) & ~n)
    return TF_NAN;
#if TF_USE_SUBNORM
  int64_t nn = 0;
  while (ix < 0x10000000000000LL) {
    ix <<= 1;
    nn -= 0x10000000000000LL;
  };
  ix += nn;
#endif
  if (ni <= 16) {
    it = (int32_t)(itcoefs[ni - 2].itcst);
    ix = itcoefs[ni - 2].itcst - it * (ix >> 32);
    rcp = itcoefs[ni - 2].rcp;
    ix |= sign;
    rx = *(double *)&ix;
#if TF_HIGH_PRECISION
    int ic = 4;
#else
    int ic = 3;
#endif
    /* 方根倒数迭代,一阶快速开方根迭代 */
    for (; ic > 0; ic--) {
      tx = rx;
      it = ni;
      zx = x;
      while ((it >>= 1) != 1) {
        tx *= tx;
        if (it & 1)
          zx *= tx;
      } // 必须如此计算快速幂,以防上溢或下溢
      zx *= tx;
      if (ni & 1)
        tx *= rx;
      rx += rcp * rx * (-1.0 + zx * tx);
    }
  } else {
#if TF_HIGH_PRECISION
    rcp = -1.0 / ni;
    /* 对数和指数计算幂函数 */
    ii = (ix & 0xfffffffffffffLL) | 0x3fe0000000000000LL;
    if (ii < 0x3fe6a09e667f3bcdLL)
      ii += 0x10000000000000LL;
    it = (ix - ii) >> 52;
    rx = *(double *)&ii;
    rx = (rx - 1.0) / (rx + 1.0);
    rx = _atanh_pnlog(rx, it % ni);
    it /= ni; // 为了精度,额外上阶码
    rx = (rx + rx) * rcp;
    rx = 1.0 + _expm1_lim(rx, &im);
    /* 上阶码和符号 */
    ii = (int64_t)(it - im) << 52;
    ix = (*(int64_t *)&rx - ii) | sign;
    rx = *(double *)&ix;
#else
    /* 对数和指数计算幂函数 */
    ii = (ix & 0xfffffffffffffLL) | 0x3fe0000000000000LL;
    if (ii < 0x3fe6a09e667f3bcdLL)
      ii += 0x10000000000000LL;
    it = (ix - ii) >> 52;
    rx = *(double *)&ii;
    rx = (rx - 1.0) / (rx + 1.0);
    rx = _atanh_pnlog(rx, it % n);
    it /= n; // 为了精度,额外上阶码
    rx = (rx + rx) / n;
    rx = 1.0 + _expm1_lim(rx, &im);
    /* 上阶码和符号 */
    ii = (int64_t)(it + im) << 52;
    ix = (*(int64_t *)&rx + ii) | sign;
    rx = *(double *)&ix;
    return rx;
#endif
  }
#if TF_HIGH_PRECISION
  /* 先计算方根倒数估值 */
  dualdouble rx2 = dmul(rx, x), tmp;
  it = ni - 2;
  if (it & 1)
    rx2 = fdfmul(rx2, rx);
  if ((it >>= 1) != 0) {
    tmp = dsqr(rx);
    if (it & 1)
      rx2 = fdf2mul(rx2, tmp);
    while ((it >>= 1) != 0) {
      tmp = fdfsqr(tmp);
      if (it & 1)
        rx2 = fdf2mul(rx2, tmp);
    }
  }
  /* 进行迭代 */
  zx = fmuladd(rx2.hi, rx, -1.0) + rx2.lo * rx;
  if (n > 0)
    return rx2.hi - ((zx + zx * rcp) * rx2.hi - rx2.lo);
  else
    return rx + rcp * rx * zx;
#else
  if (n > 0) {
    tx = rx;
    it = ni - 1;
    zx = (it & 1) ? tx * x : x;
    while ((it >>= 1) != 0) {
      tx *= tx;
      if (it & 1)
        zx *= tx;
    } // 方根迭代
    return zx - (zx + zx * rcp) * (-1.0 + zx * rx);
  } else {
    tx = rx;
    it = ni;
    zx = x;
    while ((it >>= 1) != 1) {
      tx *= tx;
      if (it & 1)
        zx *= tx;
    } // 方根倒数迭代,需保证计算不溢出
    zx *= tx;
    if (ni & 1)
      tx *= rx;
    return rx + rcp * rx * (-1.0 + zx * tx);
  }
#endif
}

/**
 * 整数方根
 */
float tfrootf(const int n, float x) {
  if (n == 1)
    return x;
  if (n == 0 || (n < 0 && x == 0))
    return (float)TF_NAN;
  if (n == -1)
    return 1.0f / x;
  float rx, zx, tx, rcp;
  int32_t sign, ii, ix = *(int32_t *)&x;
  int32_t it, im, ni = n < 0 ? -n : n;
  static const struct {
    int32_t itcst;
    int32_t inv;
    float rcp;
  } itcoefs[15] = {
      {0x5f375a84, 0x7fffffff, -1.0f / 2},  // 平方根,err=1.75126e-03
      {0x54a21e33, 0x55555557, -1.0f / 3},  // 立方根,err=2.33631e-03
      {0x4f58482a, 0x40000000, -1.0f / 4},  // 4次方根,err=2.42136e-03
      {0x4c2b8b3f, 0x33333333, -1.0f / 5},  // 5次方根,err=2.92157e-03
      {0x4a0e06f6, 0x2aaaaaab, -1.0f / 6},  // 6次方根,err=3.15876e-03
      {0x488b072b, 0x24924925, -1.0f / 7},  // 7次方根,err=3.63909e-03
      {0x4768ef84, 0x20000000, -1.0f / 8},  // 8次方根,err=3.95071e-03
      {0x46873310, 0x1c71c71c, -1.0f / 9},  // 9次方根,err=4.43253e-03
      {0x45d2b382, 0x1999999a, -1.0f / 10}, // 10次方根,err=4.79725e-03
      {0x453ef50c, 0x1745d174, -1.0f / 11}, // 11次方根,err=5.29131e-03
      {0x44c3e4b7, 0x15555555, -1.0f / 12}, // 12次方根,err=5.70192e-03
      {0x445bb952, 0x13b13b14, -1.0f / 13}, // 13次方根,err=6.21533e-03
      {0x440278da, 0x12492492, -1.0f / 14}, // 14次方根,err=6.66999e-03
      {0x43b518d1, 0x11111111, -1.0f / 15}, // 15次方根,err=7.20849e-03
      {0x43716aeb, 0x10000000, -1.0f / 16}  // 16次方根,err=7.70810e-03
  };                                        // 迭代系数
  sign = ix & 0x80000000;
  ix &= ~0x80000000;
  if ((uint32_t)(ix - 1) >= 0x7f7fffff)
    return x;
  if (((uint32_t)sign >> 31) & ~n)
    return (float)TF_NAN;
#if TF_USE_SUBNORM
  int32_t nn = 0;
  while (ix < 0x800000) {
    ix <<= 1;
    nn -= 0x800000;
  };
  ix += nn;
#endif
  if (ni <= 16) {
    it = itcoefs[ni - 2].inv;
    ix = itcoefs[ni - 2].itcst - imulhi_sar(ix, it, 0);
    rcp = itcoefs[ni - 2].rcp;
    ix |= sign;
    rx = *(float *)&ix;
#if TF_HIGH_PRECISION
    int ic = 3;
#else
    int ic = 2;
#endif
    /* 方根倒数迭代,一阶快速开方根迭代 */
    for (; ic > 0; ic--) {
      tx = rx;
      it = ni;
      zx = x;
      while ((it >>= 1) != 1) {
        tx *= tx;
        if (it & 1)
          zx *= tx;
      } // 必须如此计算快速幂,以防上溢或下溢
      zx *= tx;
      if (ni & 1)
        tx *= rx;
      rx += rcp * rx * (-1.0f + zx * tx);
    }
  } else {
#if TF_HIGH_PRECISION
    rcp = -1.0f / ni;
    /* 对数和指数计算幂函数 */
    ii = (ix & 0x7fffff) | 0x3f000000;
    if (ii < 0x3f3504f3)
      ii += 0x800000;
    it = (ix - ii) >> 23;
    rx = *(float *)&ii;
    rx = (rx - 1.0f) / (rx + 1.0f);
    rx = _atanh_pnlogf(rx, it % ni);
    it /= ni; // 为了精度,额外上阶码
    rx = (rx + rx) * rcp;
    rx = 1.0f + _expm1f_lim(rx, &im);
    /* 上阶码和符号 */
    ii = (int32_t)(it - im) << 23;
    ix = (*(int32_t *)&rx - ii) | sign;
    rx = *(float *)&ix;
#else
    /* 对数和指数计算幂函数 */
    ii = (ix & 0x7fffff) | 0x3f000000;
    if (ii < 0x3f3504f3)
      ii += 0x800000;
    it = (ix - ii) >> 23;
    rx = *(float *)&ii;
    rx = (rx - 1.0f) / (rx + 1.0f);
    rx = _atanh_pnlogf(rx, it % n);
    it /= n; // 为了精度,额外上阶码
    rx = (rx + rx) / n;
    rx = 1.0f + _expm1f_lim(rx, &im);
    /* 上阶码和符号 */
    ii = (int32_t)(it + im) << 23;
    ix = (*(int32_t *)&rx + ii) | sign;
    rx = *(float *)&ix;
    return rx;
#endif
  }
#if TF_HIGH_PRECISION
  /* 先计算方根倒数估值 */
  dualfloat rx2 = dmulf(rx, x), tmp;
  it = ni - 2;
  if (it & 1)
    rx2 = fdfmulf(rx2, rx);
  if ((it >>= 1) != 0) {
    tmp = dsqrf(rx);
    if (it & 1)
      rx2 = fdf2mulf(rx2, tmp);
    while ((it >>= 1) != 0) {
      tmp = fdfsqrf(tmp);
      if (it & 1)
        rx2 = fdf2mulf(rx2, tmp);
    }
  }
  /* 进行迭代 */
  zx = fmuladdf(rx2.hi, rx, -1.0f) + rx2.lo * rx;
  if (n > 0)
    return rx2.hi - ((zx + zx * rcp) * rx2.hi - rx2.lo);
  else
    return rx + rcp * rx * zx;
#else
  if (n > 0) {
    tx = rx;
    it = ni - 1;
    zx = (it & 1) ? tx * x : x;
    while ((it >>= 1) != 0) {
      tx *= tx;
      if (it & 1)
        zx *= tx;
    } // 方根迭代
    return zx - (zx + zx * rcp) * (-1.0f + zx * rx);
  } else {
    tx = rx;
    it = ni;
    zx = x;
    while ((it >>= 1) != 1) {
      tx *= tx;
      if (it & 1)
        zx *= tx;
    } // 方根倒数迭代,需保证计算不溢出
    zx *= tx;
    if (ni & 1)
      tx *= rx;
    return rx + rcp * rx * (-1.0f + zx * tx);
  }
#endif
}

double tfhypot(double x, double y) { return 0.0; }

float tfhypotf(float x, float y) { return 0.0f; }

double tfpow(double x, double y) { return 0.0; }

float tfpowf(float x, float y) { return 0.0f; }

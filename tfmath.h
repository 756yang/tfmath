#ifndef _TF_MATH_H_
#define _TF_MATH_H_
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>

#include "dualfloat.h"

/* TF_DIV_SLOW表示浮点除法是否很慢，比如一次除法延迟多于三次乘法则此宏应为1，会根据此宏选定算法
 */
#define TF_DIV_SLOW 1

/* 不实现非规格化浮点数的运算 */
#define TF_USE_SUBNORM 1

#ifdef __GNUC__
#if defined(__x86_64__) || defined(_M_X64)
#define ROUND(i, d) asm("cvtsd2sil %1,%0 \n\t" : "=r"(i) : "x"(d))
#define TRUNC(i, d) asm("cvttsd2sil %1,%0 \n\t" : "=r"(i) : "x"(d))
#define ROUNDF(i, s) asm("cvtss2sil %1,%0 \n\t" : "=r"(i) : "x"(s))
#define TRUNCF(i, s) asm("cvttss2sil %1,%0 \n\t" : "=r"(i) : "x"(s))
#elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
#define ROUND(i, d)                                                            \
  asm("fldl %1 \n\t"                                                           \
      "fistp %0 \n\t"                                                          \
      : "=m"(i)                                                                \
      : "m"(d))
#define TRUNC(i, d)                                                            \
  asm("fldl %1 \n\t"                                                           \
      "fisttp %0 \n\t"                                                         \
      : "=m"(i)                                                                \
      : "m"(d))
#endif
#elif defined(_MSC_VER)
#if defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#include <xmmintrin.h>
#define ROUND(i, d) (i = _mm_cvtsd_si32(_mm_set_sd(d)))
#define TRUNC(i, d) (i = _mm_cvttsd_si32(_mm_set_sd(d)))
#define ROUNDF(i, s) (i = _mm_cvtss_si32(_mm_set_ss(s)))
#define TRUNCF(i, s) (i = _mm_cvttss_si32(_mm_set_ss(s)))
#elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
#define ROUND(i, d) __asm fld d __asm fistp i
#define TRUNC(i, d) __asm fld d __asm fisttp i
#endif
#endif
#ifndef ROUND
#define ROUND(i, d) ((i) = ((d) + 0.5))
#endif
#ifndef TRUNC
#define TRUNC(i, d) ((i) = (d))
#endif
#ifndef ROUNDF
#define ROUNDF(i, s) ROUND(i, s)
#endif
#ifndef TRUNCF
#define TRUNCF(i, s) TRUNC(i, s)
#endif

#define TF_INFINITY (1e300 * 1e300)
#define TF_PRECISION 2.2204460492503130808e-16
#define TF_PRECISIONF 1.1920928955078125e-7
#define TF_NAN (0.0 * TF_INFINITY)
#define TF_MINIMAL 2.2250738585072013831e-308
#define TF_MINIMALF 1.175494350822287508e-38

double tfabs(double x);

float tfabsf(float x);

double tfexp(double x);

float tfexpf(float x);

double tfexpm1(double x);

float tfexpm1f(float x);

double tfexp2(double x);

float tfexp2f(float x);

double tfexp10(double x);

float tfexp10f(float x);

double half_log(double x);

float half_logf(float x);

double tflog(double x);

float tflogf(float x);

double half_log1p(double x);

float half_log1pf(float x);

double tflog1p(double x);

float tflog1pf(float x);

double tflog2(double x);

float tflog2f(float x);

double tflog10(double x);

float tflog10f(float x);

double tfatanh(double x);

float tfatanhf(float x);

double tfasinh(double x);

float tfasinhf(float x);

double tfacosh(double x);

float tfacoshf(float x);

double tfsinh(double x);

float tfsinhf(float x);

double tfcosh(double x);

float tfcoshf(float x);

double tftanh(double x);

float tftanhf(float x);

double tfsin(double x);

float tfsinf(float x);

double tfcos(double x);

float tfcosf(float x);

double tftan(double x);

float tftanf(float x);

double tfasin(double x);

float tfasinf(float x);

double tfacos(double x);

float tfacosf(float x);

double tfatan(double x);

float tfatanf(float x);

double tfatan2(double x, double y);

float tfatan2f(float x, float y);

double tfsqrt(double x);

float tfsqrtf(float x);

double tfcbrt(double x);

float tfcbrtf(float x);

double tfnpow(double x, const int n);

float tfnpowf(float x, const int n);

double tfroot(const int n, double x);

float tfrootf(const int n, float x);

double tfhypot(double x, double y);

float tfhypotf(float x, float y);

double tfpow(double x, double y);

float tfpowf(float x, float y);

#endif
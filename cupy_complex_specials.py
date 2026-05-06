import cupy as cp
import numpy as np

# CUDA source for Faddeeva function and derivatives
# Based on Faddeeva.cc by Steven G. Johnson
_faddeeva_source = r'''
#include <cupy/complex.cuh>
#include <math_constants.h>

#define Inf CUDART_INF
#define NaN CUDART_NAN
#define C cmplx
#define creal(z) (z).real()
#define cimag(z) (z).imag()
#define DBL_EPSILON 2.2204460492503131e-16
#define HUGE_VAL CUDART_INF
#define cexp exp

typedef complex<double> cmplx;

__device__ static inline double sqr(double x) { return x*x; }

__device__ static inline double sinc(double x, double sinx) { 
  return fabs(x) < 1e-4 ? 1 - (0.1666666666666666666667)*x*x : sinx / x; 
}

__device__ static inline double sinh_taylor(double x) {
  return x * (1 + (x*x) * (0.1666666666666666666667
                           + 0.00833333333333333333333 * (x*x)));
}

__device__ static const double expa2n2[] = {
  7.64405281671221563e-01, 3.41424527166548425e-01, 8.91072646929412548e-02,
  1.35887299055460086e-02, 1.21085455253437481e-03, 6.30452613933449404e-05,
  1.91805156577114683e-06, 3.40969447714832381e-08, 3.54175089099469393e-10,
  2.14965079583260682e-12, 7.62368911833724354e-15, 1.57982797110681093e-17,
  1.91294189103582677e-20, 1.35344656764205340e-23, 5.59535712428588720e-27,
  1.35164257972401769e-30, 1.90784582843501167e-34, 1.57351920291442930e-38,
  7.58312432328032845e-43, 2.13536275438697082e-47, 3.51352063787195769e-52,
  3.37800830266396920e-57, 1.89769439468301000e-62, 6.22929926072668851e-68,
  1.19481172006938722e-73, 1.33908181133005953e-79, 8.76924303483223939e-86,
  3.35555576166254986e-92, 7.50264110688173024e-99, 9.80192200745410268e-106,
  7.48265412822268959e-113, 3.33770122566809425e-120, 8.69934598159861140e-128,
  1.32486951484088852e-135, 1.17898144201315253e-143, 6.13039120236180012e-152,
  1.86258785950822098e-160, 3.30668408201432783e-169, 3.43017280887946235e-178,
  2.07915397775808219e-187, 7.36384545323984966e-197, 1.52394760394085741e-206,
  1.84281935046532100e-216, 1.30209553802992923e-226, 5.37588903521080531e-237,
  1.29689584599763145e-247, 1.82813078022866562e-258, 1.50576355348684241e-269,
  7.24692320799294194e-281, 2.03797051314726829e-292, 3.34880215927873807e-304,
  0.0 // underflow
};

#define ispi 0.56418958354775628694807945156

// Forward declarations
__device__ double erfcx_re(double x);
__device__ double w_im_re(double x);
__device__ double erf_re(double x);
__device__ static double erfcx_y100(double y100);
__device__ static double w_im_y100(double y100, double x);

__device__ cmplx w_complex(cmplx z, double relerr)
{
  if (z.real() == 0.0)
    return cmplx(erfcx_re(z.imag()), z.real());
  else if (z.imag() == 0)
    return cmplx(exp(-sqr(z.real())), w_im_re(z.real()));

  double a, a2, c;
  if (relerr <= 2.2204460492503131e-16) {
    relerr = 2.2204460492503131e-16;
    a = 0.518321480430085929872;
    c = 0.329973702884629072537;
    a2 = 0.268657157075235951582;
  } else {
    const double pi = 3.14159265358979323846;
    if (relerr > 0.1) relerr = 0.1;
    a = pi / sqrt(-log(relerr*0.5));
    c = (2.0/pi)*a;
    a2 = a*a;
  }
  const double x = fabs(z.real());
  const double y = z.imag(), ya = fabs(y);
  cmplx ret = 0.;
  double sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0;

  if (ya > 7 || (x > 6 && (ya > 0.1 || (x > 8 && ya > 1e-10) || x > 28))) {
    const double ispi_val = 0.564189583547756286948;
    double xs = y < 0 ? -z.real() : z.real();
    if (x + ya > 4000) {
      if (x + ya > 1e7) {
        if (x > ya) {
          double yax = ya / xs; double denom = ispi_val / (xs + yax*ya);
          ret = cmplx(denom*yax, denom);
        } else if (isinf(ya)) return ((isnan(x) || y < 0) ? cmplx(NaN,NaN) : cmplx(0,0));
        else {
          double xya = xs / ya; double denom = ispi_val / (xya*xs + ya);
          ret = cmplx(denom, denom*xya);
        }
      } else {
        double dr = xs*xs - ya*ya - 0.5, di = 2*xs*ya;
        double denom = ispi_val / (dr*dr + di*di);
        ret = cmplx(denom * (xs*di-ya*dr), denom * (xs*dr+ya*di));
      }
    } else {
      const double c0=3.9, c1=11.398, c2=0.08254, c3=0.1421, c4=0.2023;
      double nu = floor(c0 + c1 / (c2*x + c3*ya + c4));
      double wr = xs, wi = ya;
      for (nu = 0.5 * (nu - 1); nu > 0.4; nu -= 0.5) {
        double denom = nu / (wr*wr + wi*wi);
        wr = xs - wr * denom; wi = ya + wi * denom;
      }
      double denom = ispi_val / (wr*wr + wi*wi);
      ret = cmplx(denom*wi, denom*wr);
    }
    if (y < 0) return 2.0*exp(cmplx((ya-xs)*(xs+ya), 2*xs*y)) - ret;
    else return ret;
  } else if (x < 10) {
    double prod2ax = 1, prodm2ax = 1, expx2;
    if (isnan(y)) return cmplx(y,y);
    if (relerr == 2.2204460492503131e-16) {
      if (x < 5e-4) {
        const double x2 = x*x; expx2 = 1 - x2 * (1 - 0.5*x2);
        const double ax2 = 1.036642960860171859744*x;
        const double exp2ax = 1 + ax2 * (1 + ax2 * (0.5 + 0.166666666666666666667*ax2));
        const double expm2ax = 1 - ax2 * (1 - ax2 * (0.5 - 0.166666666666666666667*ax2));
        for (int n = 1; n <= 50; ++n) {
          const double coef = expa2n2[n-1] * expx2 / (a2*(n*n) + y*y);
          prod2ax *= exp2ax; prodm2ax *= expm2ax;
          sum1 += coef; sum2 += coef * prodm2ax; sum3 += coef * prod2ax;
          sum5 += coef * (2*a) * n * sinh_taylor((2*a)*n*x);
          if (coef * prod2ax < relerr * sum3) break;
        }
      } else {
        expx2 = exp(-x*x);
        const double exp2ax = exp((2*a)*x), expm2ax = 1.0 / exp2ax;
        for (int n = 1; n <= 50; ++n) {
          const double coef = expa2n2[n-1] * expx2 / (a2*(n*n) + y*y);
          prod2ax *= exp2ax; prodm2ax *= expm2ax;
          sum1 += coef; sum2 += coef * prodm2ax; sum4 += (coef * prodm2ax) * (a*n);
          sum3 += coef * prod2ax; sum5 += (coef * prod2ax) * (a*n);
          if ((coef * prod2ax) * (a*n) < relerr * sum5) break;
        }
      }
    } else {
      const double exp2ax = exp((2*a)*x), expm2ax = 1.0 / exp2ax;
      if (x < 5e-4) {
        const double x2 = x*x; expx2 = 1 - x2 * (1 - 0.5*x2);
        for (int n = 1; n <= 100; ++n) {
          const double coef = exp(-a2*(n*n)) * expx2 / (a2*(n*n) + y*y);
          prod2ax *= exp2ax; prodm2ax *= expm2ax;
          sum1 += coef; sum2 += coef * prodm2ax; sum3 += coef * prod2ax;
          sum5 += coef * (2*a) * n * sinh_taylor((2*a)*n*x);
          if (coef * prod2ax < relerr * sum3) break;
        }
      } else {
        expx2 = exp(-x*x);
        for (int n = 1; n <= 100; ++n) {
          const double coef = exp(-a2*(n*n)) * expx2 / (a2*(n*n) + y*y);
          prod2ax *= exp2ax; prodm2ax *= expm2ax;
          sum1 += coef; sum2 += coef * prodm2ax; sum4 += (coef * prodm2ax) * (a*n);
          sum3 += coef * prod2ax; sum5 += (coef * prod2ax) * (a*n);
          if ((coef * prod2ax) * (a*n) < relerr * sum5) break;
        }
      }
    }
    const double expx2erfcxy = y > -6 ? expx2*erfcx_re(y) : 2.0*exp(y*y-x*x);
    double xs = z.real(); const double sinxy = sin(xs*y);
    const double sin2xy = sin(2*xs*y), cos2xy = cos(2*xs*y);
    const double coef1 = expx2erfcxy - c*y*sum1;
    const double coef2 = c*xs*expx2;
    ret = cmplx(coef1 * cos2xy + coef2 * sinxy * sinc(xs*y, sinxy),
                coef2 * sinc(2*xs*y, sin2xy) - coef1 * sin2xy);
  } else {
    if (isnan(x)) return cmplx(x,x);
    if (isnan(y)) return cmplx(y,y);
    ret = exp(-x*x);
    double n0 = floor(x/a + 0.5); double dx = a*n0 - x;
    sum3 = exp(-dx*dx) / (a2*(n0*n0) + y*y);
    sum5 = a*n0 * sum3;
    double exp1 = exp(4*a*dx), exp1dn = 1; int dn;
    for (dn = 1; n0 - dn > 0; ++dn) {
      double np = n0 + dn, nm = n0 - dn;
      double tp = exp(-sqr(a*dn+dx));
      double tm = tp * (exp1dn *= exp1);
      tp /= (a2*(np*np) + y*y); tm /= (a2*(nm*nm) + y*y);
      sum3 += tp + tm; sum5 += a * (np * tp + nm * tm);
      if (a * (np * tp + nm * tm) < relerr * sum5) goto finish;
    }
    for (; dn < 1000; ++dn) {
      double np = n0 + dn;
      double tp = exp(-sqr(a*dn+dx)) / (a2*(np*np) + y*y);
      sum3 += tp; sum5 += a * np * tp;
      if (a * np * tp < relerr * sum5) goto finish;
    }
  }
 finish:
  return ret + cmplx((0.5*c)*y*(sum2+sum3), (0.5*c)*copysign(sum5-sum4, z.real()));
}

__device__ cmplx erf_complex(cmplx z, double relerr) {
    double x = z.real(), y = z.imag();
    double mRe_z2 = (y - x) * (x + y);
    double mIm_z2 = -2.0*x*y;
    if (y == 0) return cmplx(erf_re(x), y);
    if (x == 0) return cmplx(x, y*y > 720 ? (y > 0 ? Inf : -Inf) : exp(y*y) * w_im_re(y));
    if (mRe_z2 < -750) return (x >= 0 ? cmplx(1.0, 0.0) : cmplx(-1.0, 0.0));
    
    if (fabs(x) < 8e-2 && (fabs(y) < 1e-2 || (fabs(mIm_z2) < 5e-3 && fabs(x) < 5e-3))) {
        if (fabs(y) < 1e-2) { // taylor
            cmplx mz2 = cmplx(mRe_z2, mIm_z2);
            return z * (1.1283791670955125739 + mz2 * (0.37612638903183752464 + mz2 * (0.11283791670955125739 + mz2 * (0.026866170645131251760 + mz2 * 0.0052239776254421878422))));
        } else { // taylor_erfi
            double x2 = x*x, y2 = y*y, expy2 = exp(y2);
            return cmplx(expy2 * x * (1.1283791670955125739 - x2 * (0.37612638903183752464 + 0.75225277806367504925*y2) + x2*x2 * (0.11283791670955125739 + y2 * (0.45135166683820502956 + 0.15045055561273500986*y2))),
                         expy2 * (w_im_re(y) - x2*y * (1.1283791670955125739 - x2 * (0.56418958354775628695 + 0.37612638903183752464*y2))));
        }
    }
    if (x >= 0) return 1.0 - exp(mRe_z2) * (cmplx(cos(mIm_z2), sin(mIm_z2)) * w_complex(cmplx(-y, x), relerr));
    else return exp(mRe_z2) * (cmplx(cos(mIm_z2), sin(mIm_z2)) * w_complex(cmplx(y, -x), relerr)) - 1.0;
}

__device__ double erf_re(double x) {
    if (x == 0) return x;
    if (fabs(x) < 0.08) {
        double x2 = x*x;
        return x * (1.1283791670955125739 - x2 * (0.37612638903183752464 - x2 * (0.11283791670955125739 - x2 * (0.026866170645131251760 - x2 * 0.0052239776254421878422))));
    }
    if (x > 27) return 1.0;
    if (x < -27) return -1.0;
    return x >= 0 ? 1.0 - exp(-x*x) * erfcx_re(x) : exp(-x*x) * erfcx_re(-x) - 1.0;
}

__device__ double erfcx_re(double x) {
  if (x >= 0) {
    if (x > 50) {
      const double ispi_val = 0.564189583547756286948;
      if (x > 5e7) return ispi_val / x;
      return ispi_val*((x*x) * (x*x+4.5) + 2) / (x * ((x*x) * (x*x+5) + 3.75));
    }
    return erfcx_y100(400.0/(4.0+x));
  } else return x < -26.7 ? Inf : (x < -6.1 ? 2.0*exp(x*x) : 2.0*exp(x*x) - erfcx_y100(400.0/(4.0-x)));
}

__device__ double w_im_re(double x) {
  if (x >= 0) {
    if (x > 45) {
      const double ispi_val = 0.564189583547756286948;
      if (x > 5e7) return ispi_val / x;
      return ispi_val*((x*x) * (x*x-4.5) + 2) / (x * ((x*x) * (x*x-5) + 3.75));
    }
    return w_im_y100(100.0/(1.0+x), x);
  } else {
    if (x < -45) {
      const double ispi_val = 0.564189583547756286948;
      if (x < -5e7) return ispi_val / x;
      return ispi_val*((x*x) * (x*x-4.5) + 2) / (x * ((x*x) * (x*x-5) + 3.75));
    }
    return -w_im_y100(100.0/(1.0-x), -x);
  }
}
'''

def get_function_source(start, end):
    with open('Faddeeva.cc', 'r') as f:
        lines = f.readlines()
        return "".join(lines[start-1:end])

_faddeeva_source += "__device__ " + get_function_source(1013, 1420) + "\n"
_faddeeva_source += "__device__ " + get_function_source(1458, 1862) + "\n"

_wofz_kernel = cp.ElementwiseKernel('complex128 z', 'complex128 w', 'w = w_complex(z, 0.0);', 'wofz_kernel', preamble=_faddeeva_source)
_erf_kernel = cp.ElementwiseKernel('complex128 z', 'complex128 res', 'res = erf_complex(z, 0.0);', 'erf_kernel', preamble=_faddeeva_source)
_erfc_kernel = cp.ElementwiseKernel('complex128 z', 'complex128 res', 'res = 1.0 - erf_complex(z, 0.0);', 'erfc_kernel', preamble=_faddeeva_source)
_erfcx_kernel = cp.ElementwiseKernel('complex128 z', 'complex128 res', 'res = w_complex(cmplx(-z.imag(), z.real()), 0.0);', 'erfcx_kernel', preamble=_faddeeva_source)
_erfi_kernel = cp.ElementwiseKernel('complex128 z', 'complex128 res', 'cmplx e = erf_complex(cmplx(-z.imag(), z.real()), 0.0); res = cmplx(e.imag(), -e.real());', 'erfi_kernel', preamble=_faddeeva_source)
_dawson_kernel = cp.ElementwiseKernel('complex128 z', 'complex128 res', 'const double spi2 = 0.886226925452758013649; cmplx w = w_complex(z, 0.0); res = cmplx(0, spi2) * (exp(-z*z) - w);', 'dawson_kernel', preamble=_faddeeva_source)

def wofz(z): z = cp.asarray(z); return _wofz_kernel(z.astype(cp.complex128))
def erf(z): z = cp.asarray(z); return _erf_kernel(z.astype(cp.complex128))
def erfc(z): z = cp.asarray(z); return _erfc_kernel(z.astype(cp.complex128))
def erfcx(z): z = cp.asarray(z); return _erfcx_kernel(z.astype(cp.complex128))
def erfi(z): z = cp.asarray(z); return _erfi_kernel(z.astype(cp.complex128))
def dawson(z): z = cp.asarray(z); return _dawson_kernel(z.astype(cp.complex128))

if __name__ == '__main__':
    import scipy.special
    test_cases = [1.0+1.0j, -1.0+1.0j, 1.0-1.0j, -1.0-1.0j, 0j, 10.0, 10j, 100.0+100j, 1e-10+1e-10j, np.inf, np.nan]
    z_cp = cp.asarray(test_cases)
    for name, cp_func, sp_func in [('wofz', wofz, scipy.special.wofz), ('erf', erf, scipy.special.erf), ('erfc', erfc, scipy.special.erfc), ('erfcx', erfcx, scipy.special.erfcx), ('erfi', erfi, scipy.special.erfi), ('dawson', dawson, scipy.special.dawsn)]:
        print(f"\n--- Testing {name} ---")
        w_cp, w_np = cp_func(z_cp), sp_func(np.array(test_cases))
        print(f"{'Input z':<20} | {'CuPy result':<40} | {'SciPy result':<40} | {'Diff'}")
        print("-" * 110)
        for i in range(len(test_cases)):
            v_cp, v_np = w_cp[i].get(), w_np[i]
            print(f"{str(test_cases[i]):<20} | {str(v_cp):<40} | {str(v_np):<40} | {np.abs(v_cp-v_np):.2e}")

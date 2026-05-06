# CuPy Complex Specials

This repository provides a CuPy implementation of special functions for complex arguments, which are currently unsupported in `cupyx.scipy.special`. These functions are based on the [Faddeeva package](http://ab-initio.mit.edu/Faddeeva) by Steven G. Johnson.

## Implemented Functions

The following functions are implemented for complex arguments:

- `wofz(z)`: Faddeeva function, $w(z) = \exp(-z^2) \text{erfc}(-iz)$.
- `erf(z)`: Error function.
- `erfc(z)`: Complementary error function.
- `erfcx(z)`: Scaled complementary error function, $\exp(z^2) \text{erfc}(z)$.
- `erfi(z)`: Imaginary error function, $-i \text{erf}(iz)$.
- `dawson(z)`: Dawson function.

## Installation

Ensure you have `cupy` and `numpy` installed.

```bash
pip install cupy numpy
```

Simply include `cupy_complex_specials.py` and the Faddeeva source files (`Faddeeva.cc`, `Faddeeva.hh`) in your project.

## Usage

You can use these functions as a drop-in replacement for `scipy.special` functions when working with CuPy arrays.

```python
import cupy as cp
import cupy_complex_specials as ccs

z = cp.array([1.0 + 1.0j, -1.0 + 1.0j])
result = ccs.wofz(z)
print(result)
```

## Running Tests

The script `cupy_complex_specials.py` includes a basic test suite that compares the results with `scipy.special`.

```bash
python cupy_complex_specials.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The Faddeeva implementation is also under the MIT License by Steven G. Johnson.

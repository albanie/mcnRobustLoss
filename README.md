## TukeyLoss

The `vl_nntukeyloss` function can be useful as a robust loss for regression
tasks.

This module contains a simplified version of the Tukey loss function described in the paper: 

*Robust Optimization for Deep Regression
V. Belagiannis, C. Rupprecht, G. Carneiro, and N. Navab,
ICCV 2015, Santiago de Chile.*

and it makes use of a subset of the public code for [deepRegression](https://github.com/bazilas/matconvnet-deepReg).


### Install

The module is easiest to install with the `vl_contrib` package manager:

```
vl_contrib('install', 'mcnTukeyLoss', 'contribUrl', 'github.com/albanie/matconvnet-contrib-test/') ;
vl_contrib('setup', 'mcnTukeyLoss', 'contribUrl', 'github.com/albanie/matconvnet-contrib-test/') ;
```

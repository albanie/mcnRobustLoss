function y = vl_nntukeyloss(x, t, varargin)
% VL_NNTUKEYLOSS computes the Tukey Loss 
% This is a slight modification of the robust loss function 
% contained in the deep regression codebase 
%   https://github.com/bazilas/matconvnet-deepReg 
% and described in the paper:

%  Robust Optimization for Deep Regression
%  V. Belagiannis, C. Rupprecht, G. Carneiro, and N. Navab,
%  ICCV 2015, Santiago de Chile.
%  Copyright (C) 2016 Visual Geometry Group, University of Oxford.
%  All rights reserved.
%
% This file is made available under the terms of the BSD license
% (see the COPYING file).
%
% (modified by Samuel Albanie, 2017)

opts.scaleRes = 1 ;
opts.instanceWeights = ones(size(x)) ;
[opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;

% residuals
res = x - t ;

% Median absolute deviation (MAD)
MAD = 1.4826 * mad(res, 1) ;
C = 4.685 ;

% inliers (percentage of inliers)
nonZer = round(100 * sum(abs(res(:)) < C) / numel(res)) ;

% Big V says that this sometimes helps the convergence
MAD = MAD * opts.scaleRes ; 

res = bsxfun(@rdivide, res, MAD) ;

if isempty(dzdy) 
    scale = (C^2) / 6 ;
    yt = scale * (1 - (1 - (res ./ C).^2).^3) ;
    yt(find(abs(res) > C)) = scale ;
    y = opts.instanceWeights(:)' * yt(:) ;
else
    keep = boolean(abs(res) < C) ;
    tukDer = res .* ((1 - (res ./ C).^2).^2) ;
    res = tukDer * dzdy{1} .* keep .* opts.instanceWeights ;
    y = bsxfun(@rdivide, res, MAD) ;
end

function [mx, mt] = vl_nnmasknan(x, t, varargin) 
%VL_NNMASKNAN masks out NaN data from targets
% [MX, MT] = VL_NNMASKNAN(X,T) masks out NaN data present in targets
% T and drops the corresponding entries in X.  Produces masked outputs
% MX and MT where MT consists of the non-NaN entries of T and MX consists
% of the corresponding entries of X. The shape of both MX and MT will be
% Nx1, where N is the number of non-NaN entries of T.

[opts, dzdy] = vl_argparsepos(struct(), varargin) ;

mask = isnan(t) ;

mx = x ;
mt = t ;

if isempty(dzdy)
  mx(mask(:)) = [] ;
  mt(mask(:)) = [] ;
  mx = mx' ; mt = mt' ;
else
  mx = zeros(size(x), 'like', x) ;
  invMask = find(~mask) ;
  mx(invMask) = dzdy{1} ;
end

function Y = vl_nntukeyloss2(X, c, varargin)

opts = struct() ;
[opts, dzdy] = vl_argparsepos(opts, varargin, 'nonrecursive') ;
iter = 100 ;

Y = c ;
%X = reshape(X(1,1,:,:),[size(c{1,1},1),size(c,2)]);
%Y = [c{1,1:size(c,2)}];

%residuals
res=(Y-X);

%Median absolute deviation (MAD)
MAD = 1.4826*mad(res',1)';

%inliers (percentage of inliers)
nonZer = round(100*sum(abs(res(:))<4.685)/numel(res));

if iter<50 %(as in the paper)
%if nonZer<70 %(similar to the above) - test it again
    MAD=MAD*7; %helps the convergence at the first iterations
end

res=bsxfun(@rdivide,res,MAD);
c=4.685;

if isempty(dzdy) %forward
    
    %tukey's beiweight function
    %(http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html)
    yt = (c.^2/6) * (1 - (1-(res./c).^2).^3);
    yt(find(abs(res)>c))=(c^2)/6;
    Y = sum(yt(:));
else
    %derivatives
    tu= -1.*res.*((1-(res./c).^2).^2);
    Y_(1,1,:,:)= tu.*bsxfun(@lt,abs(res),c); % abs(x) < c
    Y = single (Y_ * dzdy{1});   
    Y=squeeze(bsxfun(@rdivide,Y,MAD));
end

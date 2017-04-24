function visualiseImdb

imdbPath = fullfile(vl_rootnn, 'data/datasets/kaggle-keypoints/imdb.mat') ;
imdb = load(imdbPath) ;

% scale up the images for easier visualisation
% from 96 to 288 (a factor of 3)
imSze = 288 ;
imScale = 3 ;
numIms = 4 ;
data = zeros(imSze,imSze,3,numIms) ;

for i = 1:numIms
  im = imdb.images.data(:,:,:,i) ;

  % make things bigger
  im = imresize(im, [imSze imSze]) ;

  % repeat across three channels for compatability
  im_  = repmat(im, [1 1 3]) ;

  landmarks = imdb.images.annotations(i,:) ;

  for j = 1:(numel(landmarks) / 2)
    if ~isnan(landmarks(j))
      x = landmarks(2*j - 1) * imScale ;
      y = landmarks(2*j) * imScale ;
      im_ = insertShape(im_, 'circle', [x y 1], 'LineWidth', 2) ;
      im_ = insertText(im_, [x y], imdb.meta.landmarks{2 *j}, ...
                            'FontSize', 11, 'BoxOpacity', 0) ;
    end
  end
  data(:,:,:,i) = im_ ;
end

% view as tiling if vlfeat is available
if exist('vl_imarraysc')
  vl_imarraysc(data) ;
end

% display in terminal if zsvision available
if exist('zs_dispFig')
  zs_dispFig ;
end

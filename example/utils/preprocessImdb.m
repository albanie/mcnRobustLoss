function imdb = preprocessImdb(opts)
% The original facial keypoint data was collected by the Bengio group
% and made available on Kaggle in 2013:
%
%  Facial Keypoint Detection Competition.
%  Kaggle, 7 May 2013. Web. 31 Dec. 2016.
%  https://www.kaggle.com/c/facial-keypoints-detection
%
% The original dataset requires HTTPS authentication for downloading, so 
% it has also been added to a mirror at the following URL:
%
% This script is only included as a reference, and is not required for the 
% experiments in robust_regression2.m (which downloads a preprocessed imdb file 
% automatically)
rawUrl = 'http://www.robots.ox.ac.uk/~albanie/data/facial-keypoints-mirror/%s.zip' ;
files = {'training', 'test'} ;

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir)
end

for i = 1:numel(files)
  if ~exist(fullfile(opts.dataDir, sprintf('%s.csv', files{i})), 'file')
    url = sprintf(rawUrl, files{i}) ;
    archive = fullfile(opts.dataDir, sprintf('%s.zip', files{i})) ;
    fprintf('downloading %s to %s\n', url, archive) ;
    websave(archive, url) ;
    unzip(archive, opts.dataDir) ;
  end
end

% ------------------------------
% load data into imdb, painfully
% ------------------------------
% training data
training = importdata(fullfile(opts.dataDir, 'training.csv')) ;
data = cellfun(@(x) {strsplit(x,',', 'CollapseDelimiters', 0)}, training) ;

% pop headers
headers = data(1) ; data(1) = [] ; 
headers = strsplit(training{1}, ',') ;

% extract landmarks
landmarks = cellfun(@(x) {x(1:30)}, data) ;

imData = cellfun(@(x) {strsplit(x{end}, ' ')}, data) ;
imdata_ = cellfun(@(x) {cellfun(@(y) str2num(y), x)}, imData) ;
trainImData = vertcat(imdata_{:}) ;
trainImages = permute(reshape(trainImData', ...
                     [96 96 1 size(trainImData, 1)]), [2 1 3 4]) ;
trainLandmarks_ = cellfun(@(x) {cellfun(@(y) str2double(y), x)}, landmarks) ;
trainLandmarks = vertcat(trainLandmarks_{:}) ;
imdb.meta.landmarks = headers(1:end-1) ;

% test data
test = importdata(fullfile(opts.dataDir, 'test.csv')) ;
data = cellfun(@(x) {strsplit(x,',', 'CollapseDelimiters', 0)}, test) ;

% pop headers
headers = data(1) ; data(1) = [] ; 
headers = strsplit(training{1}, ',') ;

imData = cellfun(@(x) {strsplit(x{end}, ' ')}, data) ;
imData_ = cellfun(@(x) {cellfun(@(y) str2num(y), x)}, imData) ;
testImData = vertcat(imdata_{:}) ;
testImages = permute(reshape(testImData', ...
                     [96 96 1 size(testImData, 1)]), [2 1 3 4]) ;

% use nans in place of test landmarks
testLandmarks = repmat({repmat(NaN, 1, 30)}, size(testImages, 4), 1) ;

% Add data to imdb
imdb.images.data = cat(4, trainImages, testImages) ;
imdb.images.set = cat(2, ones(1, size(trainImages, 4)), ...
                     3 * ones(1, size(testImages, 4))) ;
imdb.images.annotations = cat(1, trainLandmarks, testLandmarks)
save(fullfile(opts.dataDir, 'imdb.mat'), '-struct', 'imdb') ;

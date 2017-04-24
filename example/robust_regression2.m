function robust_regression(expDir, varargin) 
% robust_regression2 examines the loss curves produced by
% different loss functions on regression tasks

% we will tackle a slightly more interesting regression task - 
% facial keypoint detection (this dataset was used in a recent
% kaggle competition)

% add AutoNN to the path
run(fullfile(vl_rootnn, 'contrib/autonn/setup_autonn.m')) ; 
rng(0) ;

opts.architecture = 'resnet50' ;
opts.valRatio = 0.2 ;
opts.train.gpus = [2] ;
opts.residualScale = 96 ;
opts.loss = 'euclidean' ;
opts.refreshEvaluationCache = false ;
opts = vl_argparse(opts, varargin) ;

% choose a data location
dataOpts.dataDir = fullfile(vl_rootnn, 'data/datasets/kaggle-keypoints') ;
dataOpts.imdbPath = fullfile(dataOpts.dataDir, 'imdb.mat') ;
dataOpts.valRatio = opts.valRatio ;

% set model options
modelOpts.loss = opts.loss ;
modelOpts.architecture = opts.architecture ;
modelOpts.residualScale = opts.residualScale ; % scale outputs to [0,1]

% training options
train = opts.train ;
train.learningRate = [0.001 * ones(1,20)  ones(1,10) * 0.0001] ;
train.batchSize = 32 ;
train.numEpochs = numel(train.learningRate) ;
opts.train.continue = 1 ;
train.stats = {'regLoss', 'rmse'} ;
train.extractStatsFn = @extractStats ;

% collate options
opts.dataOpts = dataOpts ;
opts.modelOpts = modelOpts ;
opts.train = train ;

% build network
[net, opts] = net_init(opts) ;

% load imdb
imdb = loadImdb(opts) ;

[net, info] = cnn_train_autonn(net, imdb, ...
                                 @(x,y) getBatch(x, y, opts), opts.train, ...
                                 'val', find(imdb.images.set == 2), ...
                                 'expDir', expDir) ;

bestEpoch = findBestCheckpoint(expDir, 'priorityMetric', 'regLoss', ...
                               'prune', true) ;

% make predictions for kaggle submission
generatePredictions(expDir, imdb, bestEpoch, opts) ;

% ----------------------------
function imdb = loadImdb(opts)
% ----------------------------
% The original facial keypoint data was collected by the Bengio group
% and made available on Kaggle in 2013:
%
%  Facial Keypoint Detection Competition.
%  Kaggle, 7 May 2013. Web. 31 Dec. 2016.
%  https://www.kaggle.com/c/facial-keypoints-detection
%
% The original dataset requires HTTPS authentication for downloading, so 
% a preprocessed imdb version has been added to a mirror at the following URL:
domain = 'http://www.robots.ox.ac.uk' ;
url = strcat(domain, '/~albanie/data/facial-keypoints-mirror/imdb.mat') ;

if ~exist(opts.dataOpts.dataDir, 'dir')
  mkdir(opts.dataOpts.dataDir)
end

if ~exist(opts.dataOpts.imdbPath)
    fprintf('downloading %s to %s\n', url, opts.dataOpts.imdbPath) ;
    websave(opts.dataOpts.imdbPath, url) ;
end

imdb = load(opts.dataOpts.imdbPath) ;

% set a portion of the data to be used for validation purposes
numTrain = sum(imdb.images.set == 1) ;
numVal = ceil(numTrain * opts.dataOpts.valRatio) ;
imdb.images.set(numTrain - numVal + 1:numTrain) = 2 ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch, opts)
% --------------------------------------------------------------------
images = single(imdb.images.data(:,:,:,batch)) ;
targets = single(imdb.images.annotations(batch,:))' ;

% scale targets to [0, 1]
targets = targets / opts.modelOpts.residualScale ;

% duplication required for pretrained networks
switch opts.modelOpts.architecture
  case 'resnet50'
    images = repmat(images, [1 1 3 1]) ;
    images = bsxfun(@minus, images, opts.batchOpts.averageImage) ;
    images = imresize(images, opts.batchOpts.imageSize(1:2)) ;
  case 'simple'
    ; % do nothing
  otherwise
    error('architecture %s not recognised', opts.modelOpts.architecture) ;
end

if numel(opts.train.gpus) >= 1
  images = gpuArray(images) ;
end

inputs = {'input', images, 'targets', targets} ;

% -----------------------------------
function [net, opts] = net_init(opts)
% -----------------------------------

% the targets input will be common to all architectures
targets = Input('targets') ;

switch opts.modelOpts.architecture
  case 'resnet50'

    path = '/users/albanie/data/models/matconvnet/resnet50-face-bn.mat' ;
    dag = dagnn.DagNN.loadobj(load(path)) ;

    % remove breaking options
    for i = 1:numel(dag.layers)
      l = dag.layers(i) ;
      if ismember('opts', fieldnames(l.block))
        % remove single option blocks
        if numel(l.block.opts) == 1
          dag.layers(i).block.opts = {} ;
        end
      end
    end

    net = Layer.fromDagNN(dag) ;
    last = net{1}.find(@vl_nnpool, -1) ;
    rawPred = vl_nnconv(last, 'size', [1, 1, 2048, 30]) ;

    opts.batchOpts.imageSize = dag.meta.normalization.imageSize ;
    averageImage = permute(dag.meta.normalization.averageImage, [3 2 1]) ;
    opts.batchOpts.averageImage = averageImage ;
  case 'simple'

    % use the name "input" for consistency with resnets
    input = Input('input') ;

    x = vl_nnconv(input, 'size', [3, 3, 1, 48]) ;
    x = vl_nnbnorm(x, 'learningRate', [2 1 0.05]) ;
    x = vl_nnrelu(x) ;
    x = vl_nnpool(x, [2 2], 'stride', 2, 'pad', 0) ;

    x = vl_nnconv(x, 'size', [3, 3, 48, 96]) ;
    x = vl_nnbnorm(x, 'learningRate', [2 1 0.05]) ;
    x = vl_nnrelu(x) ;
    x = vl_nnpool(x, [2 2], 'stride', 2, 'pad', 0) ;

    x = vl_nnconv(x, 'size', [3, 3, 96, 144]) ;
    x = vl_nnbnorm(x, 'learningRate', [2 1 0.05]) ;
    x = vl_nnrelu(x) ;
    x = vl_nnpool(x, [2 2], 'stride', 2, 'pad', 0) ;

    x = vl_nnconv(x, 'size', [3, 3, 144, 192]) ;
    x = vl_nnbnorm(x, 'learningRate', [2 1 0.05]) ;
    x = vl_nnrelu(x) ;
    x = vl_nnpool(x, [2 2], 'stride', 2, 'pad', 0) ;

    x = vl_nnconv(x, 'size', [4, 4, 192, 30]) ;
    rawPred = reshape(x, 30, []) ;

    opts.batchOpts.imageSize = [96 96] ;
  otherwise
    error('architecture %s not recognised', opts.modelOpts.architecture) ;
end
opts.batchOpts.batchSize = opts.train.batchSize ;

% mask out the NaNs present in the target data so that they do 
% not contribute to the loss, and remove the corresponding predictions
[pred, mTargets] = Layer.create(@vl_nnmasknan, {rawPred, targets}, 'numInputDer', 1) ;

extras = {} ;
switch opts.modelOpts.loss
  case 'euclidean'
    loss_func = @vl_nneuclideanloss ;
  case 'huber'
    loss_func = @vl_nnhuberloss ;
  case 'tukey'
    extras = {'scaleRes', 1} ;
    loss_func = @vl_nntukeyloss ;
  case 'tukey7'
    extras = {'scaleRes', 7} ;
    loss_func = @vl_nntukeyloss ;
  otherwise
    error('loss %s not recognised', opts.modelOpts.loss) ;
end

used = Layer.create(loss_func, {pred, mTargets, extras{:}}, 'numInputDer', 1) ;
used.name = 'regLoss' ;
rmse = Layer.create(@vl_nnrmse, {pred, mTargets, ...
                          'residualScale', opts.modelOpts.residualScale}, ... 
                          'numInputDer', 0) ;
rmse.name = 'rmse' ;

% this is a bit of a hack so that we can easily compare the effect 
% of different loss functions
loss =  used + 0 * rmse ;

Layer.workspaceNames() ;
net = Net(loss) ;

%in = {'input', rand(224,224,3, 'single'), 'targets', rand(30, 1, 'single')} ;
%in = {'input', rand(224,224,3, 'single')} ;
%net.eval(in, 'forward')

% -------------------------------------------------------------------------
function stats = extractStats(stats, net, sel, batchSize)
% -------------------------------------------------------------------------
% modify statistics for accurate RMSE computation
for i = 1:numel(sel)
  name = net.forward(sel(i)).name ;
  if ~isfield(stats, name)
    stats.(name) = 0 ;
  end
  newValue = gather(sum(net.vars{net.forward(sel(i)).outputVar(1)}(:))) ;

  if strcmp(name, 'rmse') % normalisation has already occured
    iter = stats.num / batchSize ;
    stats.(name) = ((iter-1) * stats.(name) + newValue) / iter ;
  else
    % Update running average (same work as dagnn.Loss)
    stats.(name) = ((stats.num - batchSize) * stats.(name) + newValue) / stats.num ;
  end
end


% ---------------------------
function plotResults(results)
% ---------------------------

% set y limits to prevent matlab following anomolies
ylims = [ 0 100 ; 0 10 ] ;

figure(1) ; clf ;
subplot(1,2,1) ;
hold all ;
styles = {'o-', '+--', '+-', '.-', ':'} ;
for i = 1:numel(results)
  semilogy([results(i).info.val.regLoss]', styles{i}) ; 
end
xlabel('Training samples [x 10^3]') ; ylabel('energy') ;
grid on ;
h = legend(results(:).name) ;
set(h,'color','none');
batchSize = results(1).batchSize ;
title(sprintf('regLoss-(bs%d)', batchSize)) ;
ylim(ylims(1,:)) ;
subplot(1,2,2) ;
hold all ;
for i = 1:numel(results)
  plot([results(i).info.val.rmse]',styles{i}) ;
end
h = legend(results(:).name) ;
grid on ;
xlabel('Training samples [x 10^3]'); ylabel('error') ;
set(h,'color','none') ;
title(sprintf('rmse-(bs%d)', batchSize)) ;
ylim(ylims(2,:)) ;
drawnow ;

% this is a function for plotting figures in the terminal
% (the function can be found at https://github.com/albanie/zvision)
% but can be commented out if you are using a normal GUI
zs_dispFig ;

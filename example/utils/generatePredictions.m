function generatePredictions(expDir, imdb, bestEpoch, opts)

% revert structure to prune loss layers
tmp = load(fullfile(expDir, sprintf('net-epoch-%d.mat', bestEpoch))) ;
net = Net(tmp.net) ;
out = Layer.fromCompiledNet(net) ;
net = Net(out{1}.find(@vl_nnconv, -1)) ;

% set up getBatch
opts.modelOpts.get_eval_batch = @getBatchEval ;

% we first compute RMSE on the validation set
valRMSE = getValRMSE(net, imdb, expDir, opts) ;

% now compute the test data submissions 
computeTestPredictions(net, imdb, expDir, opts) ;

% -----------------------------------------------------
function computeTestPredictions(net, imdb, expDir, opts)
% -----------------------------------------------------
path = fullfile(expDir, 'evaluation', 'predictions.csv') ;

if exist(path, 'file') && ~opts.refreshEvaluationCache
  return ;
end

testIdx = find(imdb.images.set == 3) ;
predictions = processImages(net, imdb, testIdx, opts) ;

submissionFilePath = fullfile(opts.dataOpts.dataDir, 'IdLookupTable.csv') ;
assert(logical(exist(submissionFilePath, 'file')), ...
  sprintf(strcat('IdLookupTable not found at %s, please download it', ...
  ' from Kaggle at https://www.kaggle.com/c/facial-keypoints-detection', ...
  ' at place it in the given location'), submissionFilePath)) ;
data = importdata(submissionFilePath) ;

% pop header
header = data{1} ; data(1) = [] ;
tokens = cellfun(@(x) {strsplit(x, ',')}, data) ;
idx = cellfun(@(x) str2num(x{1}), tokens) ;
imageIds = cellfun(@(x) str2num(x{2}), tokens) ;
landmarkIdx = cellfun(@(x) find(strcmp(x{3}, imdb.meta.landmarks)), tokens) ;

fileID = fopen(path, 'w') ;
fprintf(fileID, 'RowId,Location\n') ;

for i = 1:numel(idx)
  pred = predictions(landmarkIdx(i), imageIds(i)) ;
  % clip the values to lie in the required range
  pred = max(min(pred, 96), 0) ;
  fprintf(fileID, '%d,%f\n', idx(i), pred) ; 
end
fclose(fileID) ;

% ----------------------------------------------------
function valRMSE = getValRMSE(net, imdb, expDir, opts) 
% ----------------------------------------------------
scorePath = fullfile(expDir, 'evaluation', 'valRMSE.mat') ;
if ~exist(scorePath, 'file') || opts.refreshEvaluationCache 
  % create evaluation subfolder
  if ~exist(fileparts(scorePath), 'dir')
    mkdir(fileparts(scorePath)) ;
  end

  valIdx = find(imdb.images.set == 2) ;
  predictions = processImages(net, imdb, valIdx, opts) ;
  valRMSE = computeRMSE(predictions, imdb, valIdx) ;
  save(scorePath, 'valRMSE') ;
else
  tmp = load(scorePath) ;
  valRMSE = tmp.valRMSE ;
end

[~,name] = fileparts(expDir) ;
fprintf('Validation RMSE of %s is %f\n', name, valRMSE) ;

% -----------------------------------------------------
function rmse = computeRMSE(predictions, imdb, testIdx) 
% -----------------------------------------------------
targets = imdb.images.annotations(testIdx,:) ;
mask = isnan(targets) ;

% restrict computation to available targets
predictions = predictions(find(~mask)) ;
targets = targets(find(~mask)) ;
rmse = sqrt(mean(predictions - targets).^2) ;

% ------------------------------------------------------------
function predictions = processImages(net, imdb, testIdx, opts) 
% ------------------------------------------------------------

% benchmark speed
num = 0 ;
adjustTime = 0 ;
stats.time = 0 ;
stats.num = num ; 
start = tic ;

if ~isempty(opts.train.gpus)
    net.move('gpu') ;
end

predictions = zeros(30, numel(testIdx), 'single') ; 

for t = 1:opts.batchOpts.batchSize:numel(testIdx) 

    % display progress
    progress = fix((t-1) / opts.batchOpts.batchSize) + 1 ;
    totalBatches = ceil(numel(testIdx) / opts.batchOpts.batchSize) ;
    fprintf('evaluating batch %d / %d: ', progress, totalBatches) ;

    batchSize = min(opts.batchOpts.batchSize, numel(testIdx) - t + 1) ;

    batchStart = t + (labindex - 1) ;
    batchEnd = min(t + opts.batchOpts.batchSize - 1, numel(testIdx)) ;
    batch = testIdx(batchStart : numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = opts.modelOpts.get_eval_batch(imdb, batch, opts) ;

    %net.setInputs('data', inputs{2}) ; 
    net.eval(inputs, 'forward') ;
    positions = arrayfun(@(x) find(testIdx == x, 1), batch) ;
    predictions(:,positions) = gather(net.vars{end-1}) * ...
                                          opts.modelOpts.residualScale ;

    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats.num = num ;
    stats.time = time ;
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;

    if t == 3*opts.batchOpts.batchSize + 1
        % compensate for the first three iterations, which are outliers
        adjustTime = 4*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    fprintf('speed %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
    fprintf('\n') ;
end

net.move('cpu') ;

% --------------------------------------------------------------------
function inputs = getBatchEval(imdb, batch, opts)
% --------------------------------------------------------------------
images = single(imdb.images.data(:,:,:,batch)) ;

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

inputs = {'input', images} ;

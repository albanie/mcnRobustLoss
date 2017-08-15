function robust_regression
% robust_regression is a simple regression example designed
% to demonstrate the use of the Tukey Biweight Robust Loss
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  rng(0) ;

  % First, set up three linear models
  eNet = linearModel('euclidean') ;
  hNet = linearModel('huber') ;
  tNet = linearModel('tukey') ;

  % choose experiment size
  numPoints = 50 ;

  % experiment 1 - no outliers
  runRegression(numPoints, eNet, hNet, tNet, 0) ;

  % experiment 2 - very few outliers
  runRegression(numPoints, eNet, hNet, tNet, 0.1) ;

  % experiment 3 - a healhty dose of outliers
  runRegression(numPoints, eNet, hNet, tNet, 0.3) ;

% -------------------------------------------------------------
function runRegression(numPoints, eNet, hNet, tNet, outlierProb) 
% -------------------------------------------------------------
  % generate data for the regression task with the given probability
  % of outlier occurence
  [data_x, data_y] = generateExpData(numPoints, outlierProb) ;

  % train each model with stochastic gradient descent
  ePreds = SGD(eNet, data_x, data_y) ;
  hPreds = SGD(hNet, data_x, data_y) ;
  tPreds = SGD(tNet, data_x, data_y) ;

  % visualise the predictions of the model against the data
  plotPredictions(data_x, data_y, ePreds, hPreds, tPreds) ;

% -----------------------------------------------------------------
function [data_x, data_y] = generateExpData(numPoints, outlierProb)
% -----------------------------------------------------------------
  % generate data with a simple linear model 
  data_x = (sort(rand(1, numPoints)) - 0.5) * 20 ;
  sigma = 0.5 ;
  w = rand + sigma ; 
  b = rand ;

  % introduce outliers according to the given probability by adding an 
  % extra bias term to the model
  outliers = zeros(size(data_x)) ;
  for i = 1:numel(data_x)
    if rand < outlierProb
      outliers(i) = rand * 50 ;
    end
  end
  data_y = arrayfun(@(x, o) (w * x + b + o + randn * sigma), data_x, outliers) ;

% ---------------------------------------
function preds = SGD(net, data_x, data_y)
% ---------------------------------------
  % set SGD parameters
  learningRate = 1e-3 ;
  numIter = 300 ;

  for iter = 1:numIter

    % sample a minibatch
    idx = randperm(numel(data_y), 20) ;
    
    % evaluate networks and update weights
    net.eval({'x', data_x(idx), 'y', data_y(idx)}) ;

    % perform the update
    net.setValue('w', net.getValue('w') - learningRate * net.getDer('w')) ;
    net.setValue('b', net.getValue('b') - learningRate * net.getDer('b')) ;
  end

  % make predictions with the trained model
  weight = net.getValue('w') ;
  bias = net.getValue('b') ;
  preds = arrayfun(@(x) weight * x + bias, data_x) ;

% ------------------------------
function net = linearModel(loss)
% ------------------------------
  % construct a simple linear predictive model 
  x = Input() ;
  y = Input() ;

  w = Param('value', 0.01 * randn(1, 'single')) ;
  b = Param('value', 0.01 * randn(1, 'single')) ; 

  pred = w * x + b ;

  switch loss
    case 'euclidean'
      loss_func = @vl_nneuclideanloss ;
    case 'huber'
      loss_func = @vl_nnhuberloss ;
    case 'tukey'
      loss_func = @vl_nntukeyloss ;
  end

  loss = Layer.create(loss_func, {pred, y}, 'numInputDer', 1) ;
  Layer.workspaceNames() ;
  net = Net(loss) ;

% --------------------------------------------------------------
function plotPredictions(data_x, data_y, ePreds, hPreds, tPreds)
% --------------------------------------------------------------
  clf ; figure(1) ;
  scatter(data_x, data_y) ; hold on ;
  plot(data_x, ePreds, '.-', 'LineWidth', 2) ;
  plot(data_x, tPreds, '--', 'LineWidth', 2) ;
  plot(data_x, hPreds, ':', 'LineWidth', 2) ;
  legend('data', 'Euclidean', 'Huber', 'Tukey') ;
  xlabel('x') ; ylabel('y') ;

  % display in terminal if zsvision is installed
  if exist('zs_dispFig', 'file'), zs_dispFig ; end

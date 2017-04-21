function test_mcnRobustLoss
% ---------------------------------
% run tests for mcnRobustLoss module
% ---------------------------------

% add tests to path
addpath(fullfile(fileparts(mfilename('fullpath')), 'matlab/xtest')) ;
addpath(fullfile(vl_rootnn, 'matlab/xtest/suite')) ;

% test network layers
run_robustloss_tests('command', 'nn') ;

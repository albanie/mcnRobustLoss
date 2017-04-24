function setup_mcnRobustLoss
%SETUP_MCNROBUSTLOSS Sets up mcnTukeyLoss by adding its folders 
% to the MATLAB path

run(fullfile(vl_rootnn, 'contrib', 'autonn', 'setup_autonn')) ;
root = fileparts(mfilename('fullpath')) ;
addpath(root) ;
addpath(fullfile(root, 'matlab')) ;
addpath(fullfile(root, 'example')) ;
addpath(fullfile(root, 'example/utils')) ;

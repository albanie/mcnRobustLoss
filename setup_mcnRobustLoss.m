function setup_mcnRobustLoss
%SETUP_MCNROBUSTLOSS Sets up mcnTukeyLoss by adding its folders 
% to the MATLAB path
%
% Copyright (C) 2017 Samuel Albanie 
% Licensed under The MIT License [see LICENSE.md for details]

  run(fullfile(vl_rootnn, 'contrib', 'autonn', 'setup_autonn')) ;
  root = fileparts(mfilename('fullpath')) ;
  addpath(root, [root '/matlab'], [root '/example'], [root '/example/utils']) ;

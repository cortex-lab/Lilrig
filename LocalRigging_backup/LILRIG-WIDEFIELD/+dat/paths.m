function p = paths(rig)
%DAT.PATHS Returns struct containing important paths
%   p = DAT.PATHS([RIG])
%
% Part of Rigbox

% 2013-03 CB created

thishost = hostname;

if nargin < 1 || isempty(rig)
  rig = thishost;
end

server1Name = '\\zubjects.cortexlab.net';

%% defaults
% % path containing rigbox config folders
% (this is only used for lab-wide config files)
p.rigbox = fullfile(server1Name, 'code', 'Rigging');
% Repository for local copy of everything generated on this rig
p.localRepository = 'C:\LocalExpData';
% for all data types, under the new system of having data grouped by mouse
% rather than data type
p.mainRepository = fullfile(server1Name, 'Subjects');
% Repository for experiment information
p.expInfoRepository = fullfile(server1Name, 'Subjects');
% Repository for storing eye tracking movies
p.eyeTrackingRepository = fullfile(server1Name, 'Subjects');
% directory for organisation-wide configuration files
p.globalConfig = fullfile(p.rigbox, 'config');
% directory for rig-specific configuration files
p.rigConfig = fullfile(p.globalConfig, rig);

%% load rig-specific overrides from config file, if any  
customPathsFile = fullfile(p.rigConfig, 'paths.mat');
if file.exists(customPathsFile)
  customPaths = loadVar(customPathsFile, 'paths');
  if isfield(customPaths, 'centralRepository')
    % 'centralRepository' is deprecated, remove field, if any
    customPaths = rmfield(customPaths, 'centralRepository');
  end
  % merge paths structures, with precedence on the loaded custom paths
  p = mergeStructs(customPaths, p);
end


end
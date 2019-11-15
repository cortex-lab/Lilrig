

if ~exist(rootDrive)
    rootDrive = 'G:/'; % now need this parameter
end

ops.mouseName = mouseName;
ops.thisDate = thisDate;

% AP 170501 - changed to loop for 1 or 2 colors
for i = 1:length(ops.vids)
    ops.vids(i).fileBase = fullfile(rootDrive, 'data', mouseName, thisDate);
end

for e = 1:length(expNums)
    
    expRefs{e} = dat.constructExpRef(mouseName, thisDate, expNums(e));
    
end
ops.expRefs = expRefs;

ops.localSavePath = fullfile(rootDrive, 'data', mouseName, thisDate);

cd(fullfile(rootDrive, 'data', mouseName, thisDate))
save ops.mat ops
pipelineHereKT


% mouseName = 'Erlanger';
% thisDate = datestr(now, 'yyyy-mm-dd');
% expNums = [2 3];
% 
% 
% addpath(genpath('C:\Users\Experiment\Documents\GitHub\widefield'))
% addpath(genpath('\\zserver.cortexlab.net\Code\Rigging\main'));
% addpath(genpath('\\zserver.cortexlab.net\Code\Rigging\cb-tools'));
% 
% s = svdVid.listen;
% 
% s.ops.mouseName = mouseName;
% s.ops.thisDate = thisDate;
% 
% s.wizard
% 
% s.ops.vids(1).fileBase = fullfile('J:\data', mouseName, thisDate);
% s.ops.vids(2).fileBase = fullfile('J:\data', mouseName, thisDate);
% s.ops.vids(1).rigName = 'kilotrode';
% s.ops.vids(2).rigName = 'kilotrode';
% s.ops.useGPU = true;
% 
% for e = 1:length(expNums)
%     clear ed
%     ed.expRef = dat.constructExpRef(mouseName, thisDate, expNums(e));
%     s.addExpDat(ed)
% end
% 
% s.ops.localSavePath = fullfile('J:\data', mouseName, thisDate);
% 
% cd(fullfile('J:\data', mouseName, thisDate))
% ops = s.ops;
% save ops.mat ops
% pipelineHereKT
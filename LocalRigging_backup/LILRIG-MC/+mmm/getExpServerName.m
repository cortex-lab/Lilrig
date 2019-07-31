
function s = getExpServerName()
[~, rigName] = system('hostname');
rigName = rigName(1:end-1); % removing the Line Feed character

switch rigName
    case 'LILRIG-MC'
        s = '128.40.198.173';
        
    otherwise
        error('getExpServerName:noExpServerDefined',...
            'You must set the expServerName in the function getExpServerName for your rig');
       
end
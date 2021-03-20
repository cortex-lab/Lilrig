function s = findService(id, varargin)
%SRV.FINDSERVICE Returns experiment service(s) with specified id(s)
%   This and EXP.BASICSERVICES has been replaced by SRV.LOADSERVICE. See
%   also SRV.SERVICE, SRV.LOADSERVICE, SRV.BASICSERVICES.
%
% Part of Rigbox

% 2013-06 CB created

timelineHost = iff(any(strcmp(id, 'timeline')), {'lilrig-timeline'}, {''});
micHost = iff(any(strcmp(id, 'microphone')), {'lilrig-timeline'}, {''});
timelinePort = 1001;
micPort = 1002;

remoteHosts = [timelineHost, micHost];
remotePorts = {timelinePort, micPort};

emp = cellfun(@isempty, remoteHosts);

MpepHosts = io.MpepUDPDataHosts(remoteHosts(~emp));
MpepHosts.ResponseTimeout = 30;
MpepHosts.Id = 'MPEP-Hosts';
MpepHosts.Title = 'mPep Data Acquisition Hosts'; % name displayed on startup
MpepHosts.RemotePorts = remotePorts(~emp);
MpepHosts.open();
s = {MpepHosts};
end


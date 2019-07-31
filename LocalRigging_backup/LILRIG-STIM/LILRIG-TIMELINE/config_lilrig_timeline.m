% Create a timeline object to be saved in hardware.mat
% Configured for LILRIG-TIMELINE

% Instantiate the timeline object
timeline = hw.Timeline;

% Set sample rate
timeline.DaqSampleRate = 1000;

% Set up function for configuring inputs
daq_input = @(name, channelID, measurement, terminalConfig) ...
    struct('name', name,...
    'arrayColumn', -1,... % -1 is default indicating unused
    'daqChannelID', channelID,...
    'measurement', measurement,...
    'terminalConfig', terminalConfig, ...
    'axesScale', 1);

% Configure inputs
timeline.Inputs = [...
    daq_input('chrono', 'ai0', 'Voltage', 'SingleEnded')... % for reading back self timing wave
    daq_input('tlExposeClock', 'ai1', 'Voltage', 'SingleEnded'), ...
    daq_input('pcoExposure', 'ai2', 'Voltage', 'SingleEnded')...
    daq_input('camSync', 'ai3', 'Voltage', 'SingleEnded')...
    daq_input('rewardEcho', 'ai4', 'Voltage', 'SingleEnded')...
    daq_input('eyeCamStrobe', 'ai5', 'Voltage', 'SingleEnded')...
    daq_input('faceCamStrobe', 'ai7', 'Voltage', 'SingleEnded')...
    daq_input('blueLEDmonitor', 'ai6', 'Voltage', 'SingleEnded')...
    daq_input('purpleLEDmonitor', 'ai13', 'Voltage', 'SingleEnded')...
    daq_input('audioOut', 'ai8', 'Voltage', 'SingleEnded'), ...
    daq_input('acqLive', 'ai9', 'Voltage', 'SingleEnded'), ...
    daq_input('stimScreen','ai10','Voltage', 'SingleEnded'),...
    daq_input('flipper', 'ai11', 'Voltage', 'SingleEnded'), ...
    daq_input('photoDiode', 'ai12', 'Voltage', 'SingleEnded'), ...
    daq_input('rotaryEncoder','ctr0','Position',[])
    ];

% Activate all defined inputs
timeline.UseInputs = {timeline.Inputs.name};

% Configure outputs (each output is a specialized object)

% (chrono - required timeline self-referential clock)
chronoOutput = hw.TLOutputChrono;
chronoOutput.DaqChannelID = 'port0/line0';

% (acq live output - for external triggering)
acqLiveOutput = hw.TLOutputAcqLive;
acqLiveOutput.Name = 'acqLive'; % rename for legacy compatability
acqLiveOutput.DaqChannelID = 'port0/line1';

% (output to synchronize face camera)
camSyncOutput = hw.TLOutputCamSync;
camSyncOutput.Name = 'camSync'; % rename for legacy compatability
camSyncOutput.DaqChannelID = 'port0/line2';
camSyncOutput.PulseDuration = 0.2;
camSyncOutput.InitialDelay = 0.5;

% (ramp illumination + camera exposure)
rampIlluminationOutput = hw.TLOutputRampIllumination;
rampIlluminationOutput.DaqDeviceID = timeline.DaqIds;
rampIlluminationOutput.exposureOutputChannelID =  'ao0';
rampIlluminationOutput.lightOutputChannelID = 'ao1';
rampIlluminationOutput.triggerChannelID = 'Dev1/PFI1';
rampIlluminationOutput.framerate = 70;
rampIlluminationOutput.lightExposeTime = 6.5;
rampIlluminationOutput.lightRampTime = 1;

% Package the outputs (VERY IMPORTANT: acq triggers illum, so illum must be
% set up BEFORE starting acqLive output)
timeline.Outputs = [chronoOutput,rampIlluminationOutput,acqLiveOutput,camSyncOutput];

% Configure live "oscilliscope"
timeline.LivePlot = true;

% Clear out all temporary variables
clearvars -except timeline

% save to "hardware" file
save('\\zserver.cortexlab.net\Code\Rigging\config\LILRIG-TIMELINE\hardware.mat')
disp('Saved LILRIG-TIMELINE config file')








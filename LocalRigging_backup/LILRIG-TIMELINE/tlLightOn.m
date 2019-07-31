function tlLightOn

disp('Setting up light output...');

% Get rig info
hw = tl.config;

% Set up the light (from tl.start)

sessions.illumControl = daq.createSession(hw.daqVendor);
sessions.illumControl.addAnalogOutputChannel(...
    hw.daqDevice, hw.lightOutputChannelID, 'Voltage');

% Set to off at start
sessions.illumControl.outputSingleScan(0);

% Set max voltage to light (the Cairn box can take 10V and the NIDAQ
% can give 10V, but this is usually way over the light overload)
maxLightVoltage = 5;

% Set the light shape (ramps in beginning and end)
sessions.illumControl.Rate = 40000;
lightShape = maxLightVoltage*ones(round(sessions.illumControl.Rate*(hw.lightExposeTime/1000)),1);
rampSamples = round(sessions.illumControl.Rate*(hw.lightRampTime/1000));
ramp = linspace(0,1,rampSamples)';
lightShape(1:rampSamples) = lightShape(1:rampSamples).*ramp;
lightShape(end:-1:end-rampSamples+1) = lightShape(end:-1:end-rampSamples+1).*ramp;

% Queue light for a given number of seconds to transition smoothly
% (i.e. the number of queued samples has to be an integer)
queueSeconds = hw.framerate./ ...
    gcd(hw.framerate,sessions.illumControl.Rate);
lightQueue_numSamples = sessions.illumControl.Rate*queueSeconds;
lightQueue_t = (0:lightQueue_numSamples-1)/sessions.illumControl.Rate;
% Get the samples which are closest to frame start times
frameStartIdx = diff([Inf,mod(lightQueue_t,1/hw.framerate)]) < 0;
frameStartVector = zeros(lightQueue_numSamples,1);
frameStartVector(frameStartIdx) = 1;
% Drop a light shape at the start of each frame
lightQueueFull = conv(frameStartVector,lightShape);
lightQueue = lightQueueFull(1:lightQueue_numSamples);

sessions.illumControl.TriggersPerRun = 1;
sessions.illumControl.IsContinuous = 1;
addlistener(sessions.illumControl,'DataRequired',@(src,event) src.queueOutputData(lightQueue));

queueOutputData(sessions.illumControl,lightQueue);  
startBackground(sessions.illumControl);

% Turn off on keypress
disp('Press any key to turn off light')
pause;
stop(sessions.illumControl)
sessions.illumControl.outputSingleScan(0);


end
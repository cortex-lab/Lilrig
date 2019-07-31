

function vidObj = openVideoObject(videoID)


[~, rigName] = system('hostname');
rigName = rigName(1:end-1); % removing the Line Feed character

if strcmp(videoID, 'listAll')
    switch rigName
        case 'LILRIG-MC'
            vidObj = {'eye','face'};
            return;
        otherwise
            vidObj =  {'none'};
            return
    end
end

switch [rigName '_' videoID]
    case 'LILRIG-MC_face'
        
        % Set camera device name (from imaqhwinfo)
        cam_DeviceName = 'DMK 23U618';
        
        % Set strobe output (native to camera)
        vidObj = videoinput('tisimaq_r2013_64',cam_DeviceName,'Y800 (640x480)');
        src = getselectedsource(vidObj);
        src.Strobe = 'Enable';
        src.StrobeMode = 'fixed duration';
        src.StrobeDuration = 10000;
        src.FrameRate = '30.00';
        delete(vidObj); clear vidObj src
        
        % Set recording properties (with winvideo)
        vidObj = videoinput('winvideo',cam_DeviceName,'Y800_640x480');
        src = getselectedsource(vidObj);
        src.ExposureMode = 'manual';
        src.Exposure = -6;
        src.GainMode = 'manual';
        src.Gain = 600;
        src.FrameRate = '30.0000';
        
        vidObj.FramesPerTrigger = Inf;
        vidObj.LoggingMode = 'disk';
        
    case 'LILRIG-MC_eye'

        % Set camera device name (from imaqhwinfo)
        cam_DeviceName = 'DMx 21BU04';
        
        % Set strobe output (native to camera)
        vidObj = videoinput('tisimaq_r2013_64',cam_DeviceName,'Y800 (640x480)');
        src = getselectedsource(vidObj);
        src.Strobe = 'Enable';
        src.StrobeMode = 'fixed duration';
        src.StrobeDuration = 10000;
        delete(vidObj); clear vidObj src
        
        % Set recording properties (with winvideo)
        vidObj = videoinput('winvideo',cam_DeviceName,'Y800_640x480');
        src = getselectedsource(vidObj);
        src.ExposureMode = 'manual';
        src.GainMode = 'manual';
        src.Gain = 260;
        src.Exposure = -6;
        src.FrameRate = '30.0000';
        
        vidObj.FramesPerTrigger = Inf;
        vidObj.LoggingMode = 'disk';
        
end


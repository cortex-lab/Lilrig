% lilrig_retinotopy
%
% Generates visual field sign map from retinotopy on lilrig
% stim_program - 'mpep_sparseNoise' or 'signals_sparseNoise'

if exist('Protocol','var')
    if strcmp(Protocol.xfile,'stimSparseNoiseUncorrAsync.x')
        stim_program = 'mpep_sparseNoise';
    else
        error('Unknown MPEP retinotopy protocol');
    end
elseif exist('expDef','var')
    if strcmp(expDef,'AP_sparseNoise') || strcmp(expDef,'sparseNoiseAsync_NS2')
        stim_program = 'signals_sparseNoise';
    elseif strcmp(expDef,'AP_kalatsky')
        stim_program = 'signals_kalatsky';
    else
        error('Unknown signals retinotopy expDef');
    end
end

%% Get photodiode flip times

% Threshold the photodiode trace, find flips
photodiode_thresh = 3;
photodiode_trace = Timeline.rawDAQData(stimScreen_on,photodiode_idx) > photodiode_thresh;
% (medfilt because photodiode can be intermediate value when backlight
% coming on)
photodiode_trace_medfilt = medfilt1(Timeline.rawDAQData(stimScreen_on, ...
    photodiode_idx),3) > photodiode_thresh;
photodiode_flip = find((~photodiode_trace_medfilt(1:end-1) & photodiode_trace_medfilt(2:end)) | ...
    (photodiode_trace_medfilt(1:end-1) & ~photodiode_trace_medfilt(2:end)))+1;
photodiode_flip_times = stimScreen_on_t(photodiode_flip)';

%% ~~~~~~~ Protocol type: sparse noise

switch stim_program
    case {'mpep_sparseNoise','signals_sparseNoise'}
        
        %% Get stimulus squares and times (protocol-dependent)
        
        switch stim_program
            
            %% MPEP sparse noise retinotopy
            case 'mpep_sparseNoise'
                
                % Generate the sparse noise stimuli from the protocol
                myScreenInfo.windowPtr = NaN; % so we can call the stimulus generation and it won't try to display anything
                stimNum = 1;
                ss = eval([Protocol.xfile(1:end-2) '(myScreenInfo, Protocol.pars(:,stimNum));']);
                stim_screen = cat(3,ss.ImageTextures{:});
                ny = size(stim_screen,1);
                nx = size(stim_screen,2);
                
                switch lower(photodiode_type)
                    case 'flicker'
                        
                        if size(stim_screen,3) == length(photodiode_flip_times)
                            % If stim matches photodiode, use directly
                            stim_times = photodiode_flip_times;
                            
                        elseif mod(size(stim_screen,3),2) == 1 && ...
                                length(photodiode_flip_times) == size(stim_screen,3) + 1
                            % Check for case of mismatch between photodiode and stimuli:
                            % odd number of stimuli, but one extra photodiode flip to come back down
                            photodiode_flip_times(end) = [];
                            stim_times = photodiode_flip_times;
                            warning('Odd number of stimuli, removed last photodiode');
                            
                        elseif size(stim_screen,3) ~= length(photodiode_flip_times)
                            % If there's a different kind of mismatch, guess stim times
                            % by interpolation
                            photodiode_flip_times = photodiode_flip_times([1,end]);
                            stim_duration = diff(photodiode_flip_times)/size(stim_screen,3);
                            stim_times = linspace(photodiode_flip_times(1), ...
                                photodiode_flip_times(2)-stim_duration,size(stim_screen,3))';
                            warning('Mismatching stim and photodiode, interpolating start/end')
                        end
                        
                    case 'steady'
                        % If the photodiode is on steady: extrapolate the stim times
                        if length(photodiode_flip_times) ~= 2
                            error('Steady photodiode, but not 2 flips')
                        end
                        stim_duration = diff(photodiode_flip_times)/size(stim_screen,3);
                        stim_times = linspace(photodiode_flip_times(1), ...
                            photodiode_flip_times(2)-stim_duration,size(stim_screen,3))';
                        
                end
                
            case 'signals_sparseNoise'
                
                ny = size(block.events.stimuliOnValues,1);
                nx = size(block.events.stimuliOnValues,2)/ ...
                    size(block.events.stimuliOnTimes,2);
                
                % Get stim (not firsrt: initializes to black on startup)
                stim_screen = reshape(block.events.stimuliOnValues(:,nx+1:end),ny,nx,[]);
                
                % Each photodiode flip is a screen update
                stim_times = photodiode_flip_times;
                
                % (if more stim than times, just try matching the last n stim)
                if size(stim_screen,3) > length(stim_times)
                    warning('More stims than photodiode flips - truncating beginning')
                    stim_screen(:,:,1:(size(stim_screen,3)-length(stim_times))) = [];
                end
                
                % (if times than stim, just try using the last n times)
                if size(stim_screen,3) < length(stim_times)
                    warning('More photodiode flips than stim - truncating beginning')
                    stim_times(1:(length(stim_times)-size(stim_screen,3))) = [];
                end
                
        end
        
        % Check that photodiode times match stim number
        if size(stim_screen,3) ~= length(stim_times)
            error('Mismatching stim number and photodiode times');
        end
        
        
        %% Get average response to each stimulus (bootstrap mean)
        
        surround_window = [0.3,0.5]; % 6s = [0.3,0.5]
        framerate = 1./nanmedian(diff(frame_t));
        surround_samplerate = 1/(framerate*1);
        surround_time = surround_window(1):surround_samplerate:surround_window(2);
        response_n = nan(ny,nx);
        response_grid = cell(ny,nx);
        for px_y = 1:ny
            for px_x = 1:nx
                
                switch stim_program
                    case 'mpep_sparseNoise'
                        % Gray to either black or white: use either
                        align_stims = (stim_screen(px_y,px_x,2:end)~= 0) & ...
                            (diff(stim_screen(px_y,px_x,:),[],3) ~= 0);
                        align_times = stim_times(find(align_stims)+1);
                    case 'signals_sparseNoise'
                        % Black to white
                        align_stims = stim_screen(px_y,px_x,2:end) == 1 & ...
                            stim_screen(px_y,px_x,1:end-1) == -1;
                        align_times = stim_times(find(align_stims)+1);
                end
                
                response_n(px_y,px_x) = length(align_times);
                
                % Don't use times that fall outside of imaging
                align_times(align_times + surround_time(1) < frame_t(2) | ...
                    align_times + surround_time(2) > frame_t(end)) = [];
                
                % Get stim-aligned responses, 2 choices:
                
                % 1) Interpolate times (slow - but supersamples so better)
                %         align_surround_times = bsxfun(@plus, align_times, surround_time);
                %         peri_stim_v = permute(mean(interp1(frame_t,fV',align_surround_times),1),[3,2,1]);
                
                % 2) Use closest frames to times (much faster - not different)
                align_surround_times = align_times + surround_time;
                frame_edges = [frame_t,frame_t(end)+1/framerate];
                align_frames = discretize(align_surround_times,frame_edges);
                
                % Get stim-aligned baseline (at stim onset)
                align_baseline_times = align_times;
                align_frames_baseline = discretize(align_baseline_times,frame_edges);
                
                % Don't use NaN frames (delete, dirty)
                nan_stim = any(isnan(align_frames),2) | isnan(align_frames_baseline);
                align_frames(nan_stim,:) = [];
                align_frames_baseline(nan_stim,:) = [];
                
                % Define the peri-stim V's as subtracting first frame (baseline)
                peri_stim_v = ...
                    reshape(fV(:,align_frames)',size(align_frames,1),size(align_frames,2),[]) - ...
                    nanmean(reshape(fV(:,align_frames_baseline)',size(align_frames_baseline,1),size(align_frames_baseline,2),[]),2);
                
                mean_peri_stim_v = permute(mean(peri_stim_v,2),[3,1,2]);
                
                % Store V's
                response_grid{px_y,px_x} = mean_peri_stim_v;
                
            end
        end
        
        % Get position preference for every pixel
        U_downsample_factor = 1; %2 if max method
        screen_resize_scale = 1; %3 if max method
        filter_sigma = (screen_resize_scale*2);
        
        % Downsample U
        [Uy,Ux,nSV] = size(U);
        Ud = imresize(U,1/U_downsample_factor,'bilinear');
        
        % Convert V responses to pixel responses
        use_svs = 1:500; % de-noises, otherwise size(U,3)
        n_boot = 10;
        
        response_mean_bootstrap = cellfun(@(x) bootstrp(n_boot,@mean,x')',response_grid,'uni',false);
        
        %% Get retinotopy (for each bootstrap)
        
        use_method = 'com'; % max or com
        vfs_boot = nan(size(Ud,1),size(Ud,2),n_boot);
        for curr_boot = 1:n_boot
            
            response_mean = cell2mat(cellfun(@(x) x(:,curr_boot),response_mean_bootstrap(:),'uni',false)');
            stim_im_px = reshape(permute(svdFrameReconstruct(Ud(:,:,use_svs),response_mean(use_svs,:)),[3,1,2]),ny,nx,[]);
            gauss_filt = fspecial('gaussian',[ny,nx],filter_sigma);
            stim_im_smoothed = imfilter(imresize(stim_im_px,screen_resize_scale,'bilinear'),gauss_filt);
            
            switch use_method
                case 'max'
                    % Upsample each pixel's response map and find maximum
                    [~,mi] = max(reshape(stim_im_smoothed,[],size(stim_im_px,3)),[],1);
                    [m_y,m_x] = ind2sub(size(stim_im_smoothed),mi);
                    m_yr = reshape(m_y,size(Ud,1),size(Ud,2));
                    m_xr = reshape(m_x,size(Ud,1),size(Ud,2));
                    
                case 'com'
                    % Conversely, do COM on original^2
                    [xx,yy] = meshgrid(1:size(stim_im_smoothed,2),1:size(stim_im_smoothed,1));
                    m_xr = reshape(sum(sum(bsxfun(@times,stim_im_smoothed.^2,xx),1),2)./sum(sum(stim_im_smoothed.^2,1),2),size(Ud,1),size(Ud,2));
                    m_yr = reshape(sum(sum(bsxfun(@times,stim_im_smoothed.^2,yy),1),2)./sum(sum(stim_im_smoothed.^2,1),2),size(Ud,1),size(Ud,2));
            end
            
            % Calculate and plot sign map (dot product between horz & vert gradient)
            
            % 1) get gradient direction
            [~,Vdir] = imgradient(imgaussfilt(m_yr,1));
            [~,Hdir] = imgradient(imgaussfilt(m_xr,1));
            
            % 3) get sin(difference in direction) if retinotopic, H/V should be
            % orthogonal, so the closer the orthogonal the better (and get sign)
            angle_diff = sind(Vdir-Hdir);
            angle_diff(isnan(angle_diff)) = 0;
            
            vfs_boot(:,:,curr_boot) = angle_diff;
        end
        
%% ~~~~~~~ Protocol type: kalatsky
    case 'signals_kalatsky'
    
        use_u = Uh;
        use_v = fVh;
        
        % Temporally downsample V's (allows integer frames in cycle)
        new_fs = 10;
        [use_v_downsamp,frame_t_resamp] = resample(double(use_v)',th,new_fs);
        use_v_downsamp = single(use_v_downsamp');
        
        % Downsample/blur U
        U_downsample_factor = 10;
        
        % (downsample)
        % Ud = imresize(use_u,1/U_downsample_factor,'bilinear');
        % (gaussian filter)
        % Ud = imgaussfilt(use_u,U_downsample_factor);
        % (local average)
        h = fspecial('disk',U_downsample_factor);
        Ud = convn(use_u,h,'same');
        
        use_v_Ud = ChangeU(use_u,use_v_downsamp,Ud);
        
        % Get stim times from photodiode
        photodiode_trace_medfilt = medfilt1(Timeline.rawDAQData(stimScreen_on, ...
            photodiode_idx),3) > photodiode_thresh;
        photodiode_flip = find((~photodiode_trace_medfilt(1:end-1) & photodiode_trace_medfilt(2:end)) | ...
            (photodiode_trace_medfilt(1:end-1) & ~photodiode_trace_medfilt(2:end)))+1;
        photodiode_flip_times = stimScreen_on_t(photodiode_flip)';
        stimOn_times = [photodiode_flip_times(find(diff(photodiode_flip_times) > 1)+1)];
        
        % Get parameters of stim (with signals protocol AP_kalatsky)
        % (just hardcoding this - parameters are in script)
        n_trials = length(block.paramsValues);
        stim_duration = 100;
        stim_freq = 0.1;
        stim_direction = [1,1,-1,-1];
        stim_orientation = [1,2,1,2];
        stimIDs =  mod(0:(n_trials-1),4)'+1; %  % (cycle of 4 trial types for direction/orientation)
        
        % Get time window for stim
        use_cycles = 5; % split single stim into pieces with n cycles
        
        cycle_split = (stim_freq*stim_duration)/use_cycles;
        framerate = 1./mean(diff(frame_t_resamp));
        surround_window = [0,stim_duration/cycle_split];
        surround_sampletime = 1/(framerate*1); % slight downsample for even numbers
        surround_time = surround_window(1):surround_sampletime:surround_window(2);
        
        % Loop through conditions, get power at stim frequency(bootstrapped)
        n_boot = 20; % (empirical: 5 is bit too little, 50 no difference)
        peri_stim_v_fourier = nan(size(use_u,3),n_boot,length(unique(stimIDs)));
        for curr_condition = unique(stimIDs)'
            
            % Pick stims and get times
            use_stims = find(stimIDs == curr_condition);
            use_stim_onsets = stimOn_times(use_stims);
            
            % Split stims into chunks of n cycles
            cycle_starts = find(mod(surround_time*stim_freq,1) == 0);
            use_cycle_onsets = reshape(transpose(use_stim_onsets + ...
                [0:cycle_split-1].*stim_duration/cycle_split),[],1);
            
            stim_surround_times = bsxfun(@plus, use_cycle_onsets(:), surround_time);
            % (baseline time is just first frame of each stim)
            baseline_times = stim_surround_times(:,1);
            
            % Get activity for stim, baseline subtract
            peri_stim_v_raw = reshape(interp1(frame_t_resamp,use_v_Ud',stim_surround_times), ...
                length(use_cycle_onsets),length(surround_time),[]);
            peri_stim_v_baseline = reshape(interp1(frame_t_resamp,use_v_Ud',baseline_times), ...
                length(use_cycle_onsets),1,[]);
            
            peri_stim_v = permute(peri_stim_v_raw - peri_stim_v_baseline,[3,2,1]);
            
            % Get power at stim frequency
            fourier_phase = 2*exp(-surround_time*2*pi*1i*stim_freq);
            
            % Options for bootstrapping:
            % (picked one empirically)
            
            %         % (if no bootstrap: get power within each rep)
            %             peri_stim_v_fourier(:,:,curr_condition) = ...
            %                 permute(nanmean(peri_stim_v.*fourier_phase,2),[1,3,2]);
            
            %         % (one shake within time across reps)
            %         peri_stim_v_fourier(:,:,curr_condition) = ...
            %                 permute(nanmean(AP_shake(peri_stim_v,3).*fourier_phase,2),[1,3,2]);
            
            %     % (bootstrapped mean in time across reps)
            %     peri_stim_v_bootmean =  permute(reshape(bootstrp(n_boot,@mean, ...
            %         permute(peri_stim_v,[3,1,2])),n_boot, ...
            %         size(peri_stim_v,1),size(peri_stim_v,2)),[2,3,1]);
            %     peri_stim_v_fourier(:,:,curr_condition) = ...
            %         permute(nanmean(peri_stim_v_bootmean.*fourier_phase,2),[1,3,2]);
            
            % (simulating reps by shuffling across reps in time)
            [x,y] = ndgrid(1:size(peri_stim_v),1:size(peri_stim_v,2));
            for curr_boot = 1:n_boot
                curr_boot_sub = randi(size(peri_stim_v,3),size(peri_stim_v,1),size(peri_stim_v,2));
                curr_boot_ind = sub2ind(size(peri_stim_v),x(:),y(:),curr_boot_sub(:));
                peri_stim_v_currshake = reshape( ...
                    peri_stim_v(curr_boot_ind), ...
                    size(peri_stim_v(:,:,1)));
                
                peri_stim_v_fourier(:,curr_boot,curr_condition) = ...
                    nanmean(peri_stim_v_currshake.*fourier_phase,2);
            end
            
        end
        
        vfs_boot = nan(size(Ud,1),size(Ud,2),n_boot);
        amp_boot = nan(size(Ud,1),size(Ud,2),n_boot);
        for curr_rep = 1:n_boot
            ComplexMaps = svdFrameReconstruct(Ud,squeeze(peri_stim_v_fourier(:,curr_rep,:)));
            AbsMaps = abs(ComplexMaps);
            AngleMaps = angle(ComplexMaps);
            
            % Combine maps of same orientation and opposite directions (just hard coded now)
            amp_maps = nan(size(ComplexMaps,1),size(ComplexMaps,2),2);
            angle_maps = nan(size(ComplexMaps,1),size(ComplexMaps,2),2);
            retinotopy_maps = nan(size(ComplexMaps,1),size(ComplexMaps,2),2);
            for curr_orientation = 1:2
                
                curr_stims = find(stim_orientation == curr_orientation);
                
                AbsolutePhaseS = sum(bsxfun(@times,AngleMaps(:,:,curr_stims),permute(stim_direction(curr_stims),[1,3,2])),3);
                DoubleDelayMap = sum(AngleMaps(:,:,curr_stims),3);
                DoubleDelayMap(DoubleDelayMap<0)= DoubleDelayMap(DoubleDelayMap<0) + 2*pi;
                DelayMap = DoubleDelayMap/2;
                
                AbsPhase1 = AngleMaps(:,:,curr_stims(1))-DelayMap;
                AbsPhase2 = AngleMaps(:,:,curr_stims(2))-DelayMap;
                
                AbsPhase1(sign(AbsPhase1) == stim_direction(curr_stims(1))) = AbsPhase1(sign(AbsPhase1) == stim_direction(curr_stims(1))) + 2*pi*-stim_direction(curr_stims(1)); %range=[-2*pi;0]
                AbsPhase1 = AbsPhase1*-stim_direction(curr_stims(1));
                AbsPhase2(sign(AbsPhase2) == stim_direction(curr_stims(2))) = AbsPhase2(sign(AbsPhase2) == stim_direction(curr_stims(2))) + 2*pi*-stim_direction(curr_stims(2)); %range=[-2*pi;0]
                AbsPhase2 = AbsPhase2*-stim_direction(curr_stims(2));
                
                meanAngleMaps = (AbsPhase1 + AbsPhase2)/2;
                meanAmpMaps = (AbsMaps(:,:,curr_stims(1)) + AbsMaps(:,:,curr_stims(2)))/2;
                
                angle_maps(:,:,curr_orientation) = meanAngleMaps;
                amp_maps(:,:,curr_orientation) = meanAmpMaps;
                retinotopy_maps(:,:,curr_orientation) = meanAmpMaps.*exp(meanAngleMaps*sqrt(-1));
            end
            
            % Visual sign map
            
            % 1) get gradient
            [dhdx,dhdy] = imgradientxy(angle_maps(:,:,1));
            [dvdx,dvdy] = imgradientxy(angle_maps(:,:,2));
            
            % 2) get direction of gradient
            [~,Vdir] = imgradient(dvdx,dvdy);
            [~,Hdir] = imgradient(dhdx,dhdy);
            
            % 3) get sin(difference in direction) if retinotopic, H/V should be
            % orthogonal, so the closer the orthogonal the better (and get sign)
            vfs_boot(:,:,curr_rep) = sind(Vdir-Hdir);
            amp_boot(:,:,curr_rep) = nanmean(amp_maps,3);
            
        end
        
end

%% Plot retinotopy (mean across bootstraps)

vfs_boot_mean = imgaussfilt(nanmean(vfs_boot,3),2);

figure('Name',[animal ' ' day]);
ax1 = axes;
subplot(1,2,1,ax1);
imagesc(vfs_boot_mean);
caxis([-1,1]);
axes(ax1); axis image off;
colormap(colormap_BlueWhiteRed)

ax2 = axes;
ax3 = axes;
subplot(1,2,2,ax2);
subplot(1,2,2,ax3);
h1 = imagesc(ax2,avg_im);
colormap(ax2,gray);
caxis(ax2,[0 prctile(avg_im(:),95)]);
h2 = imagesc(ax3,vfs_boot_mean);
colormap(ax3,colormap_BlueWhiteRed);
caxis([-1,1]);
set(ax2,'Visible','off');
axes(ax2); axis image off;
set(ax3,'Visible','off');
axes(ax3); axis image off;
set(h2,'AlphaData',mat2gray(abs(vfs_boot_mean))*0.5);
colormap(ax2,gray);

drawnow;


%% Align retinotopy to master retinotopy
% Plot CCF areas and coordinates aligned to master retinotopy

% Load master VFS
master_vfs_fn = ['lilrig_master_vfs.mat'];
load(master_vfs_fn);
master_align = master_vfs;

% Align animal image to master image
ref_size = size(master_align);

[optimizer, metric] = imregconfig('monomodal');
optimizer = registration.optimizer.OnePlusOneEvolutionary();
optimizer.MaximumIterations = 200;
optimizer.GrowthFactor = 1+1e-6;
optimizer.InitialRadius = 1e-4;

tformEstimate_affine = imregtform(vfs_boot_mean,master_align,'affine',optimizer,metric);
vfs_aligned = imwarp(vfs_boot_mean,tformEstimate_affine,'Outputview',imref2d(ref_size));
avg_im_aligned = imwarp(avg_im,tformEstimate_affine,'Outputview',imref2d(ref_size));

% Plot alignment
figure;
subplot(1,2,1);
imshowpair(master_align,vfs_boot_mean);
title('Unaligned VFS');
subplot(1,2,2);
imshowpair(master_align,vfs_aligned);
title('Aligned VFS');

% Plot average image with CCF overlay
figure;
overlay_color = 'b';

ax1 = axes;
ax2 = axes;
subplot(1,2,1,ax1);
subplot(1,2,1,ax2);
h1 = imagesc(ax1,avg_im_aligned);
colormap(ax1,gray);
caxis(ax1,[0 prctile(avg_im(:),98)]);
h2 = imagesc(ax2,vfs_aligned);
colormap(ax2,colormap_BlueWhiteRed);
caxis([-1,1]);
set(ax1,'Visible','off');
axes(ax1); axis image off;
set(ax2,'Visible','off');
axes(ax2); axis image off;
set(h2,'AlphaData',mat2gray(abs(vfs_aligned))*0.4);
colormap(ax1,gray);
title(ax1,'VFS/CCF overlay');
% (load and plot aligned CCF boundaries)
hold on;
load(['lilrig_cortical_area_boundaries_aligned.mat']);
h = cellfun(@(areas) cellfun(@(outline) plot(outline(:,2),outline(:,1),'color',overlay_color),areas,'uni',false), ...
    cortical_area_boundaries_aligned,'uni',false);

% Plot average image with grid overlay
subplot(1,2,2);
imagesc(avg_im_aligned);
caxis([0 prctile(avg_im(:),98)]);
colormap(gca,'gray');
axis image off
title('0.5mm grid overlay');
% (plot grid)
hold on
um2pixel = 20.6;
bregma = [540,0,570] + 0.5;
ccf_tform_fn = ['lilrig_ccf_tform.mat'];
load(ccf_tform_fn);
bregma_resize = bregma*(10/um2pixel);
bregma_align = [bregma_resize([3,1]),1]*ccf_tform.T;
bregma_offset_x = bregma_align(1);
bregma_offset_y = bregma_align(2);

spacing_um = 500;
spacing_pixels = spacing_um/um2pixel;

xlines_pos = bregma_offset_y + spacing_pixels*(ceil((min(ylim)-bregma_offset_y)./spacing_pixels):floor((max(ylim)-bregma_offset_y)./spacing_pixels));
ylines_pos = bregma_offset_x + spacing_pixels*(ceil((min(xlim)-bregma_offset_x)./spacing_pixels):floor((max(xlim)-bregma_offset_x)./spacing_pixels));

h = struct;

for curr_xline = 1:length(xlines_pos)
    h.xlines(curr_xline) = line(xlim,repmat(xlines_pos(curr_xline),1,2),'color',overlay_color,'linestyle','-');
end

for curr_yline = 1:length(ylines_pos)
    h.ylines(curr_yline) = line(repmat(ylines_pos(curr_yline),1,2),ylim,'color',overlay_color,'linestyle','-');
end

h.bregma = plot(bregma_offset_x,bregma_offset_y,'.r','MarkerSize',30);












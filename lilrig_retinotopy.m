% lilrig_retinotopy
%
% Generates visual field sign map from retinotopy on lilrig
% stim_program - 'mpep' or 'signals'

if exist('Protocol','var')
    if strcmp(Protocol.xfile,'stimSparseNoiseUncorrAsync.x')
        stim_program = 'mpep';
    else
        error('Unknown MPEP retinotopy protocol');
    end
elseif exist('expDef','var')
    if strcmp(expDef,'sparseNoiseAsync_NS2')
        stim_program = 'signals';
    else
        error('Unknown signals retinotopy expDef');
    end
end

switch stim_program
    
    %% MPEP sparse noise retinotopy
    case 'mpep'
        
        [Uy,Ux,nSV] = size(U);
        
        % Generate the sparse noise stimuli from the protocol
        myScreenInfo.windowPtr = NaN; % so we can call the stimulus generation and it won't try to display anything
        stimNum = 1;
        ss = eval([Protocol.xfile(1:end-2) '(myScreenInfo, Protocol.pars(:,stimNum));']);
        stim_screen = cat(3,ss.ImageTextures{:});
        ny = size(stim_screen,1);
        nx = size(stim_screen,2);
        
        % Threshold the photodiode trace, find flips
        photodiode_thresh = 4;
        photodiode_trace = Timeline.rawDAQData(stimScreen_on,photodiode_idx) > photodiode_thresh;
        % (medfilt because photodiode can be intermediate value when backlight
        % coming on)
        photodiode_trace_medfilt = medfilt1(Timeline.rawDAQData(stimScreen_on, ...
            photodiode_idx),3) > photodiode_thresh;
        photodiode_flip = find((~photodiode_trace_medfilt(1:end-1) & photodiode_trace_medfilt(2:end)) | ...
            (photodiode_trace_medfilt(1:end-1) & ~photodiode_trace_medfilt(2:end)))+1;
        photodiode_flip_times = stimScreen_on_t(photodiode_flip)';       
        
        switch lower(photodiode_type)
            case 'flicker'
                % Check for case of mismatch between photodiode and stimuli:
                % odd number of stimuli, but one extra photodiode flip to come back down
                if mod(size(stim_screen,3),2) == 1 && ...
                        length(photodiode_flip_times) == size(stim_screen,3) + 1
                    photodiode_flip_times(end) = [];
                    stim_times = photodiode_flip_times;
                    warning('Odd number of stimuli, removed last photodiode');
                    
                % If there's a different kind of mismatch, guess stim times
                % by interpolation
                elseif size(stim_screen,3) ~= length(photodiode_flip_times)
                    photodiode_flip_times = photodiode_flip_times([1,end]);
                    stim_duration = diff(photodiode_flip_times)/size(stim_screen,3);
                    stim_times = linspace(photodiode_flip_times(1), ...
                        photodiode_flip_times(2)-stim_duration,size(stim_screen,3))';
                end                                         
                
                % (this was a started attempt to fix dropped frames, hasn't
                % happened in a long time so not finished)
%                 if size(stim_screen,3) ~= length(photodiode_flip_times)
%                     warning([num2str(size(stim_screen,3)) ' stimuli, ', ...
%                         num2str(length(photodiode_flip_times)) ' photodiode pulses']);
%                     
%                     % Try to estimate which stim were missed by time difference
%                     photodiode_diff = diff(photodiode_flip_times);
%                     max_regular_diff_time = prctile(diff(photodiode_flip_times),99);
%                     skip_cutoff = max_regular_diff_time*2;
%                     photodiode_skip = find(photodiode_diff > skip_cutoff);
%                     est_n_pulse_skip = ceil(photodiode_diff(photodiode_skip)/max_regular_diff_time)-1;
%                     stim_skip = cell2mat(arrayfun(@(x) photodiode_skip(x):photodiode_skip(x)+est_n_pulse_skip(x)-1, ...
%                         1:length(photodiode_skip),'uni',false));
%                     
%                     % Interpolate stim times from start/end
%                     if isempty(est_n_pulse_skip) || length(photodiode_flip_times) + sum(est_n_pulse_skip) ~= size(stim_screen,3)
%                         error('Can''t match photodiode events to stimuli, trying interpolation')                                           
%                     end
%                 end             
                
            case 'steady'
                % If the photodiode is on steady: extrapolate the stim times
                if length(photodiode_flip_times) ~= 2
                    error('Steady photodiode, but not 2 flips')
                end
                stim_duration = diff(photodiode_flip_times)/size(stim_screen,3);
                stim_times = linspace(photodiode_flip_times(1), ...
                    photodiode_flip_times(2)-stim_duration,size(stim_screen,3))';
                
        end
        
        % Get average response to each stimulus
        surround_window = [0.1,0.5]; % 6s = [0.1,0.5], 6f = [0.05,0.2]
        framerate = 1./nanmedian(diff(frame_t));
        surround_samplerate = 1/(framerate*1);
        surround_time = surround_window(1):surround_samplerate:surround_window(2);
        response_n = nan(ny,nx);
        response_grid = cell(ny,nx);
        for px_y = 1:ny
            for px_x = 1:nx
                
                % Use first frame of dark or light stim
                align_stims = (stim_screen(px_y,px_x,2:end)~= 0) & ...
                    (diff(stim_screen(px_y,px_x,:),[],3) ~= 0);
                align_times = stim_times(find(align_stims)+1);
                
                %         error('What the hell is this? throw away half of the data?')
                %         align_times = align_times(round(length(align_times)/2):end);
                
                response_n(px_y,px_x) = length(align_times);
                
                % Don't use times that fall outside of imaging
                align_times(align_times + surround_time(1) < frame_t(2) | ...
                    align_times + surround_time(2) > frame_t(end)) = [];
                
                % Get stim-aligned responses, 2 choices:
                
                % 1) Interpolate times (slow - but supersamples so better)
                %         align_surround_times = bsxfun(@plus, align_times, surround_time);
                %         peri_stim_v = permute(mean(interp1(frame_t,fV',align_surround_times),1),[3,2,1]);
                
                % 2) Use closest frames to times (much faster - not different)
                align_surround_times = bsxfun(@plus, align_times, surround_time);
                frame_edges = [frame_t,frame_t(end)+1/framerate];
                align_frames = discretize(align_surround_times,frame_edges);
                
                align_frames(any(isnan(align_frames),2),:) = [];
                
                % Define the peri-stim V's as subtracting first frame (baseline)
                peri_stim_v = bsxfun(@minus, ...
                    reshape(fV(:,align_frames)',size(align_frames,1),size(align_frames,2),[]), ...
                    reshape(fV(:,align_frames(:,1))',size(align_frames(:,1),1),size(align_frames(:,1),2),[]));
                
                mean_peri_stim_v = permute(mean(peri_stim_v,2),[3,1,2]);
                
                % Save V's
                response_grid{px_y,px_x} = mean_peri_stim_v;
                
            end
        end
        
        % Get position preference for every pixel
        U_downsample_factor = 1; %2 if max method
        screen_resize_scale = 1; %3 if max method
        filter_sigma = (screen_resize_scale*2);
        
        % Downsample U
        use_u_y = 1:Uy;
        Ud = imresize(U(use_u_y,:,:),1/U_downsample_factor,'bilinear');
        
        % Convert V responses to pixel responses
        use_svs = 1:size(U,3);
        n_boot = 10;
        
        response_mean_boostrap = cellfun(@(x) bootstrp(n_boot,@mean,x')',response_grid,'uni',false);
        
        % (to split trials instead of bootstrap)
        %split_trials = cellfun(@(x) shake(discretize(1:size(x,2),round(linspace(1,size(x,2),n_boot+1)))),response_grid,'uni',false);
        %response_mean_boostrap = cellfun(@(x,y) grpstats(x',y','mean')',response_grid,split_trials,'uni',false);
        use_method = 'com'; % max or com
        vfs_boot = nan(size(Ud,1),size(Ud,2),n_boot);
        for curr_boot = 1:n_boot
            
            response_mean = cell2mat(cellfun(@(x) x(:,curr_boot),response_mean_boostrap(:),'uni',false)');
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
        
    %% Signals sparse noise retinotopy
    case 'signals'
        
        [Uy,Ux,nSV] = size(U);
        
        ny = size(block.events.stimuliOnValues,1);
        nx = size(block.events.stimuliOnValues,2)/ ...
            size(block.events.stimuliOnTimes,2);
        stim_screen = reshape(block.events.stimuliOnValues,ny,nx,[]);
        
        % Get average response to each stimulus
        surround_window = [0,0.5]; % 6s = [0.1,0.5], 6f = [0.05,0.2]
        framerate = 1./nanmean(diff(frame_t));
        surround_samplerate = 1/(framerate*1);
        surround_time = surround_window(1):surround_samplerate:surround_window(2);
        response_n = nan(ny,nx);
        response_grid = cell(ny,nx);
        for px_y = 1:ny
            for px_x = 1:nx
                
                % Get all square flips to white only
                align_stims = (stim_screen(px_y,px_x,2:end) - ...
                    stim_screen(px_y,px_x,1:end-1)) > 0;
                
                % Get corresponding stim times (don't add 1: first screen blank
                % with no photodiode flip)
                align_times = stimOn_times(find(align_stims));
                
                response_n(px_y,px_x) = length(align_times);
                
                % Don't use times that fall outside of imaging
                align_times(align_times + surround_time(1) < frame_t(2) | ...
                    align_times + surround_time(2) > frame_t(end)) = [];
                
                % Get stim-aligned responses, 2 choices:
                
                % 1) Interpolate times (slow - but supersamples so better)
                %         align_surround_times = bsxfun(@plus, align_times, surround_time);
                %         peri_stim_v = permute(mean(interp1(frame_t,fV',align_surround_times),1),[3,2,1]);
                
                % 2) Use closest frames to times (much faster - not different)
                align_surround_times = bsxfun(@plus, align_times, surround_time);
                frame_edges = [frame_t,frame_t(end)+1/framerate];
                align_frames = discretize(align_surround_times,frame_edges);
                
                align_frames(any(isnan(align_frames),2),:) = [];
                
                % Define the peri-stim V's as subtracting first frame (baseline)
                peri_stim_v = bsxfun(@minus, ...
                    reshape(fV(:,align_frames)',size(align_frames,1),size(align_frames,2),[]), ...
                    reshape(fV(:,align_frames(:,1))',size(align_frames(:,1),1),size(align_frames(:,1),2),[]));
                
                mean_peri_stim_v = permute(mean(peri_stim_v,2),[3,1,2]);
                
                % Save V's
                response_grid{px_y,px_x} = mean_peri_stim_v;
                
            end
        end
        
        % Get position preference for every pixel
        U_downsample_factor = 1; %2 if max method
        screen_resize_scale = 1; %3 if max method
        filter_sigma = (screen_resize_scale*2);
        
        % Downsample U
        use_u_y = 1:Uy;
        Ud = imresize(U(use_u_y,:,:),1/U_downsample_factor,'bilinear');
        
        % Convert V responses to pixel responses
        use_svs = 1:size(U,3);
        n_boot = 10;
        
        response_mean_boostrap = cellfun(@(x) bootstrp(n_boot,@mean,x')',response_grid,'uni',false);
        
        % (to split trials instead of bootstrap)
        %split_trials = cellfun(@(x) shake(discretize(1:size(x,2),round(linspace(1,size(x,2),n_boot+1)))),response_grid,'uni',false);
        %response_mean_boostrap = cellfun(@(x,y) grpstats(x',y','mean')',response_grid,split_trials,'uni',false);
        use_method = 'max'; % max or com
        vfs_boot = nan(size(Ud,1),size(Ud,2),n_boot);
        for curr_boot = 1:n_boot
            
            response_mean = cell2mat(cellfun(@(x) x(:,curr_boot),response_mean_boostrap(:),'uni',false)');
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
end

%% Plot retinotopy

vfs_median = imgaussfilt(nanmedian(vfs_boot,3),2);

figure('Name',[animal ' ' day]);
ax1 = axes;
subplot(1,2,1,ax1);
imagesc(vfs_median);
caxis([-1,1]);
axes(ax1); axis image off;
colormap(colormap_BlueWhiteRed)

ax2 = axes;
ax3 = axes;
subplot(1,2,2,ax2);
subplot(1,2,2,ax3);
h1 = imagesc(ax2,avg_im(use_u_y,:));
colormap(ax2,gray);
caxis(ax2,[0 prctile(avg_im(:),95)]);
h2 = imagesc(ax3,vfs_median);
colormap(ax3,colormap_BlueWhiteRed);
caxis([-1,1]);
set(ax2,'Visible','off');
axes(ax2); axis image off;
set(ax3,'Visible','off');
axes(ax3); axis image off;
set(h2,'AlphaData',mat2gray(abs(vfs_median))*0.3);
colormap(ax2,gray);

drawnow;
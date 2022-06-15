% Batch preprocess local widefield data on Lilrig

% Find all animal/day folders with data

% Set drives/folders where data could be
data_drives = {'C:\Data','G:\Data'};

% Loop through drives, find folders besides 'test'
process_paths = cell(0);
for curr_drive = data_drives
    curr_drive = cell2mat(curr_drive);
    curr_drive_dir = dir(curr_drive);
    animal_paths = ~contains({curr_drive_dir.name},{'.','test'});
    for animal_path = {curr_drive_dir(animal_paths).name}
        animal_path = cell2mat(animal_path);
        animal_dir = dir(fullfile(curr_drive,animal_path));
        % Find day paths of the form YYYY-MM-DD
        day_paths = cellfun(@(x) ...
            ~isempty(regexp(x,'\d\d\d\d-\d\d-\d\d')), ...
            {animal_dir.name}) &...
            [animal_dir.isdir];
        % Add those paths to the list of paths-to-process
        process_paths(end+1:end+sum(day_paths)) = ...
            cellfun(@(x) fullfile(curr_drive,animal_path,x), ...
            {animal_dir(day_paths).name},'uni',false);
    end
end

%% Preprocess all local data

for curr_data_path = process_paths
    curr_data_path = cell2mat(curr_data_path);
    
    curr_data_path_sep = regexp(curr_data_path,filesep,'split');
    animal = curr_data_path_sep{end-1};
    day = curr_data_path_sep{end};
    
    
    %% Run SVD on day's data
    % Single U for day, V split by recording, average im by color, frame info
    
    [U,Vrec,im_avg_color,frame_info] = AP_preprocess_widefield_pco(curr_data_path);
    
    
    %% Match frames to recording number
    % Determine correct recording folder by closest folder and frame time
    
    % Get recordings (numbered folders) in experiment path
    % (this dat package is legacy)
    expPath = dat.expPath(animal,day,1,'main');
    experiment_path = fileparts(expPath{2});
    experiment_dir =  dir(experiment_path);
    recording_dir_idx = cellfun(@(x) ~isempty(x), regexp({experiment_dir.name},'^\d*$'));
    recording_dir = experiment_dir(recording_dir_idx);
    
    % Find creation time of recording folders
    rec_dir_starttime = nan(size(recording_dir));
    for curr_recording = 1:length(recording_dir)
        curr_t = System.IO.File.GetCreationTime( ...
            fullfile(recording_dir(curr_recording).folder, ...
            recording_dir(curr_recording).name));
        
        rec_dir_starttime(curr_recording) = ...
            datenum(double([curr_t.Year,curr_t.Month,curr_t.Day, ...
            curr_t.Hour,curr_t.Minute,curr_t.Second]));
    end
    
    % Get nearest recording folder time for each imaging recording start
    [~,rec_frame_start_idx] = unique([frame_info.rec_idx]);
    frame_timestamp_cat = vertcat(frame_info.timestamp);
    rec_im_time = frame_timestamp_cat(rec_frame_start_idx);
    
    [~,im_rec_idx] = min(abs(rec_im_time - rec_dir_starttime'),[],2);
    
    % Sanity check: same number of recording numbers as V recordings
    if length(im_rec_idx) ~= size(Vrec,1)
        error('Different number of V''s and associated recordings');
    end
    
    
    %% Save preprocessed widefield data on server
    
    % Set number of components to save
    max_components_save = 2000;
    n_components_save = min(max_components_save,size(U{1},3));
    
    % Assume 2 colors in order of blue/purple
    color_names = {'blue','purple'};
    
    % Save frame information in experiment folder
    frame_info_fn = fullfile(experiment_path,'widefield_frame_info');
    save(frame_info_fn,'frame_info','-v7.3');
    
    % Save mean images in experiment folder by color
    for curr_color = 1:length(color_names)
        curr_mean_im_fn = fullfile(experiment_path, ...
            sprintf('meanImage_%s.npy',color_names{curr_color}));
        writeNPY(im_avg_color(:,:,curr_color),curr_mean_im_fn);
    end
    
    % Save spatial components in experiment (animal/day) folder by color
    for curr_color = 1:length(color_names)
        curr_U_fn = fullfile(experiment_path, ...
            sprintf('svdSpatialComponents_%s.npy',color_names{curr_color}));
        writeNPY(U{curr_color}(:,:,1:n_components_save),curr_U_fn);
    end
    
    % Save temporal components in associated recording folders
    for curr_recording = 1:size(Vrec,1)
        for curr_color = 1:length(color_names)
            curr_V_fn = fullfile(experiment_path, ...
                recording_dir(im_rec_idx(curr_recording)).name, ...
                sprintf('svdTemporalComponents_%s.npy',color_names{curr_color}));
            writeNPY(Vrec{curr_recording,curr_color}(1:n_components_save,:)',curr_V_fn);
        end
    end
    
    %% Move raw data to Lugaro for tape archiving
    
    % Move from local to staging folder (takes time)
    % (temporary location to prevent tape-archiving partial data)
    tapedrive_staging_path = ...
        fullfile('\\lugaro.cortexlab.net\bigdrive\staging', ...
        sprintf('%s_%s',animal,day));
    mkdir(tapedrive_staging_path);
    staging_status = movefile(curr_data_path,tapedrive_staging_path);
    
    % Move from staging to toarchive folder (instantaneous)
    % (contents of this folder are regularly moved to tape)
    tapedrive_toarchive_path = ...
        fullfile('\\lugaro.cortexlab.net\bigdrive\toarchive', ...
        sprintf('%s_%s',animal,day));
    toarchive_status = movefile(tapedrive_staging_path,tapedrive_toarchive_path);
    
    
end































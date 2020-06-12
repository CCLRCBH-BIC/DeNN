clear
clc
rng(1);
load C:\Users\zyang\Dropbox\DeNN_simu\002_S_4213.mat
load C:\Users\zyang\Dropbox\DeNN_simu\X_setv2
mask = squeeze(mean(fMRIdata,1))>0.05*max(fMRIdata(:));
c23T1 = (c2T1_erode+c3T1_erode)>0.1;

c1T1_float = c1T1>0.9;
tdim = 135;TR=3;deltaT = 0.1;
%% simulate signal
signal_random = zeros((tdim*TR+15)/deltaT,8);
ind = randi(size(signal_random,1),30,8);
for i = 1:8
    for j = 1:size(ind,1)
        signal_random(ind(j,i):min(ind(j,i)+3/deltaT,size(signal_random,1)),i) = 1;
    end
end
signal_weight = [0.5 0.5 zeros(1,6);
    0 0 0.5 0.5 0 0 0 0;
    0 0 0 0 0.5 0.5 0 0;
    0 0 0 0 0 0 0.5 0.5;
    0.5 0 0.5 0 0 0 0 0;
    0 0.5 0 0.5 0 0 0 0;
    0 0 0 0 0.5 0 0.5 0;
    0 0 0 0 0 0.5 0 0.5];
xyzdim = size(fMRIdata);xyzdim = xyzdim(2:end);
TR_array = TR*(1:tdim)+15;
deltaT_array = deltaT*(1:size(signal_random,1));
signal_TR = zeros(tdim,8);
for i = 1:8
    signal_temp = conv(signal_random(:,i),hrf_set(:,1));
    signal_TR(:,i) = interp1(deltaT_array,signal_temp(1:size(signal_random,1)),...
        TR_array,'pchip','extrap');
end
%% simulate noise
noise_q = zscore(fMRIdata(:,c23T1>0));
nvoxel = sum(mask(:));
ind = randi(size(noise_q,2),nvoxel,1);


signal_xyz = zeros([tdim,xyzdim]);
noise_xyz = zeros([tdim,xyzdim]);
noise_xyz(:,mask>0) = zscore(noise_q(:,ind)+0.05*randn(tdim,nvoxel));
for x = 1:xyzdim(1)
    for y = 1:xyzdim(2)
        for z = 1:xyzdim(3)
            if mask(x,y,z)>0
                ind = randperm(10,1); 
                if ind<9 && c1T1_float(x,y,z)>0% only 4/5 gray matter have signal
                    signal_xyz(:,x,y,z) = zscore(signal_TR*(signal_weight(ind,:)+0.1*randn(1,8))');
                end
            end
                
        end
    end
end
noise_median = median(median(abs(noise_xyz(:,mask>0))));
signal_median = median(abs(signal_xyz(signal_xyz~=0)));

% noise_xyz = noise_xyz + 0.1*noise_median*randn(size(noise_xyz));
figure;
subplot(3,1,1)
imagesc(zscore(noise_xyz(:,mask>0))');caxis([-3 3]);
subplot(3,1,3)
imagesc(zscore(fMRIdata(:,mask>0))');caxis([-3 3]);
temp = signal_xyz(:,mask>0);
subplot(3,1,2)
imagesc(zscore(temp(:,temp(1,:)~=0))');

fMRIdata = 0.2*signal_xyz+0.8*noise_xyz;
save G:\ADNI2_analysis\Denoise\ssegdata_simu\002_S_4213_simu.mat c1T1 c23T1 c2T1 c2T1_erode c3T1...
    c3T1_erode CSFcomp fMRIdata mask GSWMCSF_TS motion_parameter WMcomp signal_xyz noise_xyz -v7.3
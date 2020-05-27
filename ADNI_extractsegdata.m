clear
clc
[num,txt,raw] = xlsread('G:\ADNI2_analysis\Denoise\ssegdata_result\ADNI_fALFF_data.xlsx');
datadir = 'G:\ADNI2_data';TR=3;
savepath = 'G:\ADNI2_analysis\Denoise\ssegdata';
sz = [91,109,91];
WM_mask=ROImask([33,58,54],sz,[],[],4);
CSF_mask=ROImask([36,47,46],sz,[],[],4);
for i = 2:size(raw,1)
    subname = raw{i,1}
    if exist([savepath,'/',subname,'.mat'])==0
        acqdate = raw{i,9};
        ind = strfind(acqdate,'/');
        acqdate = [acqdate(ind(2)+1:end),'-',sprintf('%02d',str2num(acqdate(1:ind(1)-1))),'-',...
            sprintf('%02d',str2num(acqdate(ind(1)+1:ind(2)-1)))];
        fMRIname = dir([datadir,'/',subname,'/rest_state/',acqdate,'*']);
        fMRIname = fMRIname(1).name;
        T1name = dir([datadir,'/',subname,'/MPRAGE/',acqdate,'*']);
        T1name = T1name(1).name;
        
        motion_file = selectImageDir([datadir,'/',raw{i,1},'/rest_state/',fMRIname],'rp_*.txt');
        motion_parameter = textread(motion_file{1});
        motion_parameter = motion_parameter(6:end,:);
        fMRIdata = ZY_fmrimerge([datadir,'/',subname,'/rest_state/',...
            fMRIname,'/std_raResting_State_*.nii']);
        
        T1dir = [datadir,'/',subname,'/MPRAGE/',T1name];
        c1T1 = load_untouch_nii([T1dir,'/SyNT12MNIc1MPRAGE.nii']);
        c2T1 = load_untouch_nii([T1dir,'/SyNT12MNIc2MPRAGE.nii']);
        c3T1 = load_untouch_nii([T1dir,'/SyNT12MNIc3MPRAGE.nii']);
        
        c1T1 = c1T1.img;
        c2T1 = c2T1.img;
        c3T1 = c3T1.img;
        
        c3T1_erode = spm_erode(c3T1);
        c23T1 = 3*(spm_erode(c3T1)>0.5);
        c2T1_erode = spm_erode(spm_erode(c2T1));
        c23T1(c2T1_erode>0.5) = 2;
        
        GSWMCSF_TS = [mean(fMRIdata(:,c1T1+c2T1+c3T1>0.5*max(c1T1(:))),2,'omitnan'),mean(fMRIdata(:,c23T1==2),2,'omitnan'),...
            mean(fMRIdata(:,c23T1==3),2,'omitnan')];
        GSWMCSF_TS2 = [mean(fMRIdata(:,c1T1+c2T1+c3T1>0.5*max(c1T1(:))),2,'omitnan'),mean(fMRIdata(:,WM_mask>0),2,'omitnan'),...
            mean(fMRIdata(:,CSF_mask>0),2,'omitnan')];
        [U,S,V] = svd(fMRIdata(:,c23T1==2),'econ');
        WMcomp = U(:,1:5);
        [U,S,V] = svd(fMRIdata(:,c23T1==3),'econ');
        CSFcomp = U(:,1:5);
        fMRIdata_c1 = fMRIdata(:,c1T1>0);
        fMRIdata_c23 = fMRIdata(:,c23T1>0);
        fMRIdata(:,c1T1+c2T1+c3T1<0.05*max(c1T1(:)))=0;
        save([savepath,'/',subname,'.mat'],...
            'fMRIdata','c1T1','c2T1','c3T1','c2T1_erode','c3T1_erode','c23T1','GSWMCSF_TS',...
            'WMcomp','CSFcomp','motion_parameter','-v7.3');
    end
end

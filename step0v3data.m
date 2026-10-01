
clc; clear; close all;

fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('   Step 0: 统一采样\n');
fprintf('═══════════════════════════════════════════════════════════════\n');
fprintf('开始时间: %s\n\n', datestr(now));

%% ── 选择数据集 ──────────────────────────────────────────────────────────
DATASET_NAME = 'Trento';  % ★Trento★  可选: Houston2013, Houston2018, Augsburg, MUUFL, Trento

fprintf('[0/4] 配置数据集: %s\n', DATASET_NAME);
CONFIG = get_dataset_config(DATASET_NAME);

%% ── 采样参数 ────────────────────────────────────────────────────────────
SAMPLES_PER_CLASS = 10;
MIN_SAMPLES       = 10;

% ★Trento★ 预定义划分开关（true = 直接用 TR/TE; false = 从全标注随机采样）
USE_PREDEFINED_SPLIT = false;

fprintf('  每类采样数目: %d\n', SAMPLES_PER_CLASS);

%% ── 加载真值 ────────────────────────────────────────────────────────────
fprintf('\n[1/4] 加载数据...\n');

% ★Trento★ GT_MODE=='split' 时合并 TR+TE
if isfield(CONFIG,'GT_MODE') && strcmp(CONFIG.GT_MODE,'split')
    tr_d = load(CONFIG.GT_TRAIN_FILE);  ftr = fieldnames(tr_d);
    te_d = load(CONFIG.GT_TEST_FILE);   fte = fieldnames(te_d);
    tr_gt = double(tr_d.(ftr{1}));
    te_gt = double(te_d.(fte{1}));
    gt    = max(tr_gt, te_gt);          % 两图非重叠, 取max即合并
    [H, W] = size(gt);
    fprintf('  GT 来自 %s + %s 合并\n', CONFIG.GT_TRAIN_FILE, CONFIG.GT_TEST_FILE);
else
    gt_data = load(CONFIG.GT_FILE);
    gt_fields = fieldnames(gt_data);
    gt = double(gt_data.(gt_fields{1}));
    [H, W] = size(gt);
end

fprintf('  真值尺寸: %d × %d\n', H, W);
fprintf('  细类数: %d\n', CONFIG.N_FINE_CLASSES);

%% ── 采样 ────────────────────────────────────────────────────────────────
fprintf('\n[2/4] 采样...\n');

rng(42);
gt_vec = gt(:);

% ★Trento★ 预定义划分模式
if USE_PREDEFINED_SPLIT && isfield(CONFIG,'GT_MODE') && strcmp(CONFIG.GT_MODE,'split')
    fprintf('  使用预定义 TR/TE 划分\n');
    train_px_idx = find(tr_gt(:) > 0);
    y_fine_train = tr_gt(train_px_idx);
    test_px_idx  = find(te_gt(:) > 0);
    y_fine_test  = te_gt(test_px_idx);

    fprintf('  %-4s %-25s %8s %8s\n', 'ID', '类别', '训练', '测试');
    fprintf('  %s\n', repmat('─',1,50));
    for c = 1:CONFIG.N_FINE_CLASSES
        ntr = sum(y_fine_train==c); nte = sum(y_fine_test==c);
        fprintf('  %-4d %-25s %8d %8d\n', c, CONFIG.FINE_NAMES{c}, ntr, nte);
    end
else
    % 原始随机采样逻辑
    train_px_idx = [];
    y_fine_train = [];

    fprintf('  %-4s %-25s %8s %8s %8s\n', 'ID', '类别', '总数', '训练', '测试');
    fprintf('  %s\n', repmat('─',1,60));

    for c = 1:CONFIG.N_FINE_CLASSES
        class_idx = find(gt_vec == c);
        n_total = length(class_idx);
        if n_total == 0
            fprintf('  %-4d %-25s %8d %8d %8d  [跳过]\n', ...
                c, CONFIG.FINE_NAMES{c}, 0, 0, 0); continue;
        end
        if n_total >= SAMPLES_PER_CLASS
            n_train = SAMPLES_PER_CLASS;
        elseif n_total >= MIN_SAMPLES * 2
            n_train = floor(n_total / 2);
        else
            n_train = min(MIN_SAMPLES, n_total);
        end
        perm = randperm(n_total);
        train_idx_c = class_idx(perm(1:n_train));
        train_px_idx = [train_px_idx; train_idx_c];
        y_fine_train = [y_fine_train; ones(n_train,1)*c];
        status=''; if n_total<SAMPLES_PER_CLASS, status='[不足]'; end
        fprintf('  %-4d %-25s %8d %8d %8d  %s\n', c, CONFIG.FINE_NAMES{c}, ...
            n_total, n_train, n_total-n_train, status);
    end

    all_labeled_idx = find(gt_vec > 0);
    test_px_idx  = setdiff(all_labeled_idx, train_px_idx);
    y_fine_test  = gt_vec(test_px_idx);
end

fprintf('\n  总训练样本: %d\n', length(train_px_idx));
fprintf('  总测试样本: %d\n', length(test_px_idx));

%% ── 粗类标签 ────────────────────────────────────────────────────────────
fprintf('\n[3/4] 生成粗类标签...\n');

y_coarse_train = CONFIG.FINE2COARSE(y_fine_train)';
y_coarse_test  = CONFIG.FINE2COARSE(y_fine_test)';

for c = CONFIG.COARSE_IDS_USED
    n = sum(y_coarse_train == c);
    if n > 0
        fprintf('    粗类 %d (%s): %d 样本\n', c, CONFIG.COARSE_NAMES{c}, n);
    end
end

%% ── 保存 ────────────────────────────────────────────────────────────────
fprintf('\n[4/4] 保存...\n');
checksum = sum(train_px_idx) + sum(y_fine_train);
save('shared_samples.mat', ...
    'train_px_idx','test_px_idx', ...
    'y_fine_train','y_fine_test', ...
    'y_coarse_train','y_coarse_test', ...
    'H','W','CONFIG','DATASET_NAME', ...
    'SAMPLES_PER_CLASS','checksum');

fprintf('  校验值: %.0f\n', checksum);
fprintf('\n═══ 采样完成 - %s  训练: %d ═══\n', DATASET_NAME, length(train_px_idx));


%% =========================================================================
%  数据集配置函数
%% =========================================================================
function CONFIG = get_dataset_config(dataset_name)
    CONFIG = struct();
    CONFIG.DATASET_NAME = dataset_name;
    CONFIG.N_COARSE_CLASSES = 4;
    CONFIG.COARSE_NAMES = {'Vegetation','Water','Soil','Impervious'};
    CONFIG.COARSE_IDS_USED = 1:4;
    CONFIG.COARSE_COLORS = [
        0.20 0.72 0.20;
        0.10 0.45 0.90;
        0.65 0.50 0.25;
        0.70 0.40 0.40];

    switch dataset_name

        % ═════════════════════════ ★Trento★ ═════════════════════════
        case 'Trento'
            CONFIG.N_BANDS  = 63;        % AISA Eagle: 400-990nm
            CONFIG.HAS_SWIR = false;
            CONFIG.LIDAR_TYPE = 'DSM';
            CONFIG.N_LIDAR_CHANNELS = 1;

            CONFIG.HSI_FILE   = 'HSI.mat';
            CONFIG.LIDAR_FILES = {'LiDAR.mat'};

            % ★ GT 来自 TR+TE 两个文件 ★
            CONFIG.GT_FILE       = '';
            CONFIG.GT_MODE       = 'split';
            CONFIG.GT_TRAIN_FILE = 'TR.mat';
            CONFIG.GT_TEST_FILE  = 'TE.mat';

            % 波段索引（AISA Eagle 63ch, ~9.4nm/band, 起始~400nm）
            CONFIG.BANDS = struct();
            CONFIG.BANDS.BLUE  = 6;   % ~456nm
            CONFIG.BANDS.GREEN = 15;  % ~541nm
            CONFIG.BANDS.RED   = 29;  % ~672nm
            CONFIG.BANDS.RE    = 36;  % ~738nm
            CONFIG.BANDS.NIR   = 50;  % ~870nm

            % 6 个细类
            CONFIG.N_FINE_CLASSES = 6;
            CONFIG.FINE_NAMES = {
                'Apples',       ...  % 1 → Vegetation
                'Buildings',    ...  % 2 → Impervious
                'Ground',       ...  % 3 → Soil
                'Woods',        ...  % 4 → Vegetation
                'Vineyard',     ...  % 5 → Vegetation
                'Roads'};            % 6 → Impervious

            %  细类:         1  2  3  4  5  6
            CONFIG.FINE2COARSE = [1, 4, 3, 1, 1, 4];

            CONFIG.COARSE_TO_FINE = {
                [1, 4, 5],     ...  % 1-Vegetation (Apples, Woods, Vineyard)
                [],            ...  % 2-Water (无)
                [3],           ...  % 3-Soil  (Ground)
                [2, 6]};            % 4-Impervious (Buildings, Roads)

            CONFIG.COARSE_IDS_USED = [1, 3, 4];   % 无 Water 类

            % ★ 空间特征文件名（Trento 特有）★
            CONFIG.SPATIAL_FEATURES.hsi_emap  = 'HSI_EMAPOutput.mat';
            CONFIG.SPATIAL_FEATURES.lid_emap  = 'LiDAREMAPOutput.mat';
            CONFIG.SPATIAL_FEATURES.hsi_gabor = 'HSIGABOR_PCA2.mat';
            CONFIG.SPATIAL_FEATURES.lid_gabor = 'LiDARGABOR.mat';
            CONFIG.SPATIAL_FEATURES.hsi_glcm  = 'HSIGLCMP10.mat';
            CONFIG.SPATIAL_FEATURES.lid_glcm  = 'LiDARGLCM.mat';


        case 'Houston2013'
            CONFIG.N_BANDS = 144;
            CONFIG.HAS_SWIR = false;
            CONFIG.LIDAR_TYPE = 'DSM';
            CONFIG.N_LIDAR_CHANNELS = 1;
            CONFIG.HSI_FILE = 'casi.mat';
            CONFIG.LIDAR_FILES = {'LiDAR.mat'};
            CONFIG.GT_FILE = 'wholecasigt2013.mat';
            CONFIG.BANDS = struct();
            CONFIG.BANDS.BLUE=15; CONFIG.BANDS.GREEN=39;
            CONFIG.BANDS.RED=65; CONFIG.BANDS.RE=75; CONFIG.BANDS.NIR=105;
            CONFIG.N_FINE_CLASSES = 15;
            CONFIG.FINE_NAMES = {
                'Healthy Grass','Stressed Grass','Synthetic Grass','Trees',...
                'Soil','Water','Residential','Commercial','Road','Highway',...
                'Railway','Parking Lot 1','Parking Lot 2','Tennis Court','Running Track'};
            CONFIG.FINE2COARSE = [1,1,4,1,3,2,4,4,4,4,4,4,4,4,4];
            CONFIG.COARSE_TO_FINE = {[1,2,4],[6],[5],[3,7,8,9,10,11,12,13,14,15]};
            CONFIG.SPATIAL_FEATURES.hsi_emap='HSI_EMAPOutput.mat';
            CONFIG.SPATIAL_FEATURES.lid_emap='LiDAREMAPOutput.mat';
            CONFIG.SPATIAL_FEATURES.hsi_gabor='HUG_PCA2.mat';
            CONFIG.SPATIAL_FEATURES.lid_gabor='LiDARGABOR.mat';
            CONFIG.SPATIAL_FEATURES.hsi_glcm='GLCM_multiscale.mat';
            CONFIG.SPATIAL_FEATURES.lid_glcm='LIDARGLCM_multiscale.mat';

        case 'Houston2018'
            CONFIG.N_BANDS=48; CONFIG.HAS_SWIR=false;
            CONFIG.LIDAR_TYPE='MS_LiDAR'; CONFIG.N_LIDAR_CHANNELS=3;
            CONFIG.HSI_FILE='huston2018.mat';
            CONFIG.LIDAR_FILES={'DSM_no_scaled.mat'};
            CONFIG.GT_FILE='huston2018_downgt.mat';
            CONFIG.BANDS=struct();
            CONFIG.BANDS.BLUE=5;CONFIG.BANDS.GREEN=15;CONFIG.BANDS.RED=25;
            CONFIG.BANDS.RE=35;CONFIG.BANDS.NIR=45;
            CONFIG.N_FINE_CLASSES=20;
            CONFIG.FINE_NAMES={'Healthy Grass','Stressed Grass','Artificial Turf',...
                'Evergreen Trees','Deciduous Trees','Bare Earth','Water',...
                'Residential','Commercial','Road','Sidewalk','Crosswalk',...
                'Major Thoroughfares','Highway','Railway','Paved Parking',...
                'Unpaved Parking','Cars','Trains','Stadium Seats'};
            CONFIG.FINE2COARSE=[1,1,4,1,1,3,2,4,4,4,4,4,4,4,4,4,3,4,4,4];
            CONFIG.COARSE_TO_FINE={[1,2,4,5],[7],[6,17],[3,8,9,10,11,12,13,14,15,16,18,19,20]};
            CONFIG.SPATIAL_FEATURES.hsi_emap='HSI_EMAPOutput.mat';
            CONFIG.SPATIAL_FEATURES.lid_emap='LiDAREMAPOutput.mat';
            CONFIG.SPATIAL_FEATURES.hsi_gabor='HUG_PCA2.mat';
            CONFIG.SPATIAL_FEATURES.lid_gabor='LiDARGABOR.mat';
            CONFIG.SPATIAL_FEATURES.hsi_glcm='GLCM_multiscale.mat';
            CONFIG.SPATIAL_FEATURES.lid_glcm='LIDARGLCM_multiscale.mat';

        case 'Augsburg'
            CONFIG.N_BANDS=180; CONFIG.HAS_SWIR=true;
            CONFIG.LIDAR_TYPE='DSM'; CONFIG.N_LIDAR_CHANNELS=1;
            CONFIG.HSI_FILE='Augsburg_HSI.mat';
            CONFIG.LIDAR_FILES={'Augsburg_DSM.mat'};
            CONFIG.GT_FILE='Augsburggt.mat';
            CONFIG.BANDS=struct();
            CONFIG.BANDS.BLUE=10;CONFIG.BANDS.GREEN=30;CONFIG.BANDS.RED=50;
            CONFIG.BANDS.RE=70;CONFIG.BANDS.NIR=100;
            CONFIG.BANDS.SWIR1=150;CONFIG.BANDS.SWIR2=170;
            CONFIG.N_FINE_CLASSES=7;
            CONFIG.FINE_NAMES={'Forest','Residential Area','Industrial Area',...
                'Low Plants','Allotment','Commercial Area','Water'};
            CONFIG.FINE2COARSE=[1,4,4,1,1,4,2];
            CONFIG.COARSE_TO_FINE={[1,4,5],[7],[],[2,3,6]};
            CONFIG.COARSE_IDS_USED=[1,2,4];
            CONFIG.SPATIAL_FEATURES.hsi_emap='HSI_EMAPOutput.mat';
            CONFIG.SPATIAL_FEATURES.lid_emap='LiDAREMAPOutput.mat';
            CONFIG.SPATIAL_FEATURES.hsi_gabor='HUG_PCA2.mat';
            CONFIG.SPATIAL_FEATURES.lid_gabor='LiDARGABOR.mat';
            CONFIG.SPATIAL_FEATURES.hsi_glcm='GLCM_multiscale.mat';
            CONFIG.SPATIAL_FEATURES.lid_glcm='LIDARGLCM_multiscale.mat';

        case 'MUUFL'
            CONFIG.N_BANDS=64; CONFIG.HAS_SWIR=false;
            CONFIG.LIDAR_TYPE='HeightIntensity'; CONFIG.N_LIDAR_CHANNELS=2;
            CONFIG.HSI_FILE='mufflHSI.mat';
            CONFIG.LIDAR_FILES={'LiDAR1.mat','LiDAR2.mat'};
            CONFIG.GT_FILE='muffl_gt.mat';
            CONFIG.BANDS=struct();
            CONFIG.BANDS.BLUE=5;CONFIG.BANDS.GREEN=15;CONFIG.BANDS.RED=30;
            CONFIG.BANDS.RE=45;CONFIG.BANDS.NIR=60;
            CONFIG.N_FINE_CLASSES=11;
            CONFIG.FINE_NAMES={'Trees','Grass Ground','Mixed Ground','Dirt and Sand',...
                'Road','Water','Building Shadow','Building','Sidewalk','Yellow Curb','Cloth Panels'};
            CONFIG.FINE2COARSE=[1,1,1,3,4,2,4,4,4,4,4];
            CONFIG.COARSE_TO_FINE={[1,2,3],[6],[4],[5,7,8,9,10,11]};
            CONFIG.SPATIAL_FEATURES.hsi_emap='HSI_EMAPOutput.mat';
            CONFIG.SPATIAL_FEATURES.lid_emap='LiDAREMAPOutput.mat';
            CONFIG.SPATIAL_FEATURES.hsi_gabor='HUG_PCA2.mat';
            CONFIG.SPATIAL_FEATURES.lid_gabor='LiDARGABOR.mat';
            CONFIG.SPATIAL_FEATURES.hsi_glcm='GLCM_multiscale.mat';
            CONFIG.SPATIAL_FEATURES.lid_glcm='LIDARGLCM_multiscale.mat';

        otherwise
            error('未知数据集: %s', dataset_name);
    end

    % 生成细类颜色
    CONFIG.FINE_COLORS = generate_fine_colors(CONFIG);
end


function colors = generate_fine_colors(CONFIG)
    n_classes = CONFIG.N_FINE_CLASSES;
    colors = zeros(n_classes, 3);
    base_hsv = [
        0.33 0.70 0.72;
        0.58 0.78 0.90;
        0.08 0.62 0.65;
        0.00 0.55 0.85];
    for c = 1:4
        if c<=length(CONFIG.COARSE_TO_FINE)
            fine_ids=CONFIG.COARSE_TO_FINE{c}; n_fine=length(fine_ids);
            if n_fine==0, continue; end
            base=base_hsv(c,:);
            for j=1:n_fine
                f=fine_ids(j);
                if f<=n_classes
                    h_var=(j-(n_fine+1)/2)/max(n_fine,1)*0.08;
                    v_var=(j-(n_fine+1)/2)/max(n_fine,1)*0.15;
                    hsv_c=base;
                    hsv_c(1)=mod(hsv_c(1)+h_var,1);
                    hsv_c(3)=min(1,max(0.3,hsv_c(3)+v_var));
                    colors(f,:)=hsv2rgb(hsv_c);
                end
            end
        end
    end
end
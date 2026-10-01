clc; clear; close all;
t_start = tic;
%% ─── 参数 ────────────────────────────────────────────────────────────
P.N_PCA_HSI      = 8;
P.PCA_K          = 15;

P.CAND_WIN       = 5;
P.CAND_SPEC_THR  = 0.99;
P.CAND_LIDAR_THR = 0.02;

P.RF_GLOBAL      = 500;
P.RF_MINLEAF     = 2;
P.RF_NUMPRED     = 0.5;

P.USE_MRF        = true;
P.MRF_N_ITER     = 5;
P.SOBEL_N_PC     = 2;

OUT_DIR = 'results_paper_consensus';
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

%% ═══ [0] 加载原始数据 ════════════════════════════════════════════════
fprintf('\n[0] 加载...\n');
load('shared_samples.mat');
train_px_idx=train_px_idx(:); y_fine_train=y_fine_train(:);
test_px_idx=test_px_idx(:);   y_fine_test=y_fine_test(:);
eps_v=1e-8; NFC=CONFIG.N_FINE_CLASSES; N_PIX=H*W;
B=CONFIG.BANDS; NB=CONFIG.N_BANDS;
assert(isempty(intersect(train_px_idx,test_px_idx)), '训练/测试交叠');

hsi=norm01(double(lv(CONFIG.HSI_FILE)));
[~,~,NB_actual]=size(hsi);
hsi_2d=reshape(hsi,N_PIX,[]); hsi_norms=sqrt(sum(hsi_2d.^2,2)+eps_v);

lidar_ch=[];
for i=1:length(CONFIG.LIDAR_FILES)
    f=CONFIG.LIDAR_FILES{i};
    if exist(f,'file'), lidar_ch=cat(3,lidar_ch,norm01(double(lv(f)))); end
end
N_LIDAR=size(lidar_ch,3);
if N_LIDAR>=1, lidar_ht_vec=reshape(lidar_ch(:,:,1),N_PIX,1);
else,          lidar_ht_vec=zeros(N_PIX,1); end

gt_fine = load_gt(DATASET_NAME, CONFIG);

fprintf('  数据集: %s | 训练:%d 测试:%d NFC=%d 波段:%d LiDAR:%d通道\n',...
    DATASET_NAME,length(train_px_idx),length(test_px_idx),NFC,NB_actual,N_LIDAR);

%% ═══ [1] BASE 特征 ════════════════════════════════════════════════════
fprintf('\n[1] BASE 特征构建...\n');

mu_hsi=mean(hsi_2d,1); hc=hsi_2d-mu_hsi;
[~,~,Vh]=svds(hc,P.N_PCA_HSI); M1=single(hc*Vh);
for k=1:P.N_PCA_HSI
    mn=min(M1(:,k)); mx=max(M1(:,k));
    if mx>mn, M1(:,k)=(M1(:,k)-mn)/(mx-mn); end
end

M2=[]; M2_names={};
if isfield(B,'NIR')&&B.NIR<=NB_actual && isfield(B,'RED')&&B.RED<=NB_actual
    nir=hsi(:,:,B.NIR); red=hsi(:,:,B.RED);
    M2=[M2, reshape((nir-red)./(nir+red+eps_v),[],1)]; M2_names{end+1}='NDVI';
end
if isfield(B,'GREEN')&&B.GREEN<=NB_actual && isfield(B,'NIR')&&B.NIR<=NB_actual
    grn=hsi(:,:,B.GREEN); nir2=hsi(:,:,B.NIR);
    M2=[M2, reshape((grn-nir2)./(grn+nir2+eps_v),[],1)]; M2_names{end+1}='NDWI';
end
if isfield(B,'SWIR1')&&B.SWIR1<=NB_actual && isfield(B,'NIR')&&B.NIR<=NB_actual
    swir=hsi(:,:,B.SWIR1); nir3=hsi(:,:,B.NIR);
    M2=[M2, reshape((swir-nir3)./(swir+nir3+eps_v),[],1)]; M2_names{end+1}='NDBI';
end
M2=single(M2); M2(~isfinite(M2))=0;

M3=[]; M3_names={};
if N_LIDAR>=1
    ht=lidar_ch(:,:,1);
    M3=[M3,ht(:)];                                          M3_names{end+1}='Height';
    ht_min=ordfilt2(ht,1,ones(21));
    M3=[M3,reshape(ht-ht_min,[],1)];                        M3_names{end+1}='nDSM';
    [gx,gy]=gradient(ht);
    M3=[M3,reshape(sqrt(gx.^2+gy.^2),[],1)];               M3_names{end+1}='Gradient';
end
if N_LIDAR>=2, M3=[M3,reshape(lidar_ch(:,:,2),[],1)]; M3_names{end+1}='Intensity'; end
M3=single(M3); M3(~isfinite(M3))=0;

BASE=[double(M1),double(M2),double(M3)]; BASE(~isfinite(BASE))=0;
fprintf('  M1(HSI_PC)=%dD  M2(%s)=%dD  M3(%s)=%dD → BASE=%dD\n',...
    size(M1,2),strjoin(M2_names,'/'),size(M2,2),...
    strjoin(M3_names,'/'),size(M3,2),size(BASE,2));

%% ═══ [2] 六族空间特征 ═══════════════════════════════════════════════
fprintf('\n[2] 空间特征加载...\n');

switch DATASET_NAME
    case 'Houston2013'
        SF={'HSI_EMAPOutput.mat','LiDAREMAP.mat','HUG_PCA2.mat',...
            'LiDARGABOR.mat','HSIGLCM.mat','GLCM_LiDAR_all.mat'};
    case 'Houston2018'
        SF={'hu2018EMAP.mat','LiDAREMAP.mat','hu2018GABOR.mat',...
            'LiGABOR.mat','HSIGLCM.mat','GLCM_LiDAR_all.mat'};
    case 'Trento'
        SF={'HSI_EMAPOutput.mat','LiDAREMAPOutput.mat','HSIGABOR_PCA2.mat',...
            'LiDARGABOR.mat','HSIGLCM.mat','GLCM_LiDAR_all.mat'};
    case 'Augsburg'
        SF={'AugEMAP.mat','AugLiEMAP.mat','AugGABOR_PCA3.mat',...
            'LiGABOR.mat','HSIGLCM.mat','GLCM_LiDAR_all.mat'};
    otherwise
        SF={'HSI_EMAP.mat','LiDAR_EMAP.mat','HSI_Gabor.mat',...
            'LiDAR_Gabor.mat','HSI_GLCM.mat','LiDAR_GLCM.mat'};
end
sf_tags={'HSI_EMAP','LID_EMAP','HSI_GABOR','LID_GABOR','HSI_GLCM','LID_GLCM'};

RAW=struct();
for si=1:6
    RAW.(sf_tags{si}) = load_spatial(SF{si}, N_PIX, H, W);
end

rng(0); pidx=randperm(N_PIX,min(50000,N_PIX));
HE=pcr(RAW.HSI_EMAP,pidx,P.PCA_K);  LE=pcr(RAW.LID_EMAP,pidx,P.PCA_K);
HG=pcr(RAW.HSI_GABOR,pidx,P.PCA_K); LG=pcr(RAW.LID_GABOR,pidx,P.PCA_K);
HC=pcr(RAW.HSI_GLCM,pidx,P.PCA_K);  LC=pcr(RAW.LID_GLCM,pidx,P.PCA_K);

X_FULL=[BASE,HE,LE,HG,LG,HC,LC]; X_FULL(~isfinite(X_FULL))=0;
N_FEAT=size(X_FULL,2);
fprintf('  X_FULL: %dD = BASE(%d) + 6×PCA(%d)\n',N_FEAT,size(BASE,2),P.PCA_K);

%% ═══ [3] Sobel多源梯度图 ════════════════════════════════════════════
if P.USE_MRF
    fprintf('\n[3] Sobel梯度图...\n');
    [~,~,Vs]=svds(hc, P.SOBEL_N_PC);
    hsi_pc_3d=reshape(hc*Vs, H, W, P.SOBEL_N_PC);
    HGradIm = compute_sobel_grad(hsi_pc_3d);
    fprintf('  HSI: %d PC, mean=%.4f\n', P.SOBEL_N_PC, mean(HGradIm(:)));
    if N_LIDAR>=1
        LGradIm = compute_sobel_grad(lidar_ch);
        fprintf('  LiDAR: %d通道, mean=%.4f\n', N_LIDAR, mean(LGradIm(:)));
    else
        LGradIm = ones(H,W);
    end
end

fprintf('\n[4] Label Consensus 伪标签筛选 \n');
fprintf('    spec≥%.2f, Δh≤%.3f, win=%d, 无 K_max, 无 r_max\n',...
    P.CAND_SPEC_THR, P.CAND_LIDAR_THR, P.CAND_WIN);

half_w = floor(P.CAND_WIN/2);
train_set = false(N_PIX,1); train_set(train_px_idx) = true;

% ═════════════════════════════════════════════════════════════════════
% Phase 1: 收集所有 (candidate_pixel, anchor_idx, anchor_label, similarity)
%          对应论文 §2.3 中的 G_i 构造 + 并集 Γ_cand = ∪ G_i
% ═════════════════════════════════════════════════════════════════════
fprintf('\n  ── Phase 1: 物理筛选 (Eq.6) ──\n');
t_p1 = tic;

% 预分配 (每个锚点最多窗口大小-1 个候选)
max_cands = length(train_px_idx) * (P.CAND_WIN^2 - 1);
all_cands = zeros(max_cands, 4);
n_filled = 0;
rej_spec = 0; rej_lidar = 0; n_checked = 0;

for ti=1:length(train_px_idx)
    px = train_px_idx(ti); lab = y_fine_train(ti);
    [ri,ci] = ind2sub([H,W], px);
    s_tr = hsi_2d(px,:); n_tr = hsi_norms(px); ht_tr = lidar_ht_vec(px);

    for dr=-half_w:half_w
        for dc=-half_w:half_w
            if dr==0 && dc==0, continue; end
            nr = ri + dr; nc_ = ci + dc;
            if nr<1 || nr>H || nc_<1 || nc_>W, continue; end
            npx = sub2ind([H,W], nr, nc_);
            if train_set(npx), continue; end
            n_checked = n_checked + 1;

            cs = (s_tr * hsi_2d(npx,:)') / (n_tr*hsi_norms(npx) + eps_v);
            if cs < P.CAND_SPEC_THR, rej_spec = rej_spec + 1; continue; end

            if N_LIDAR>=1 && abs(ht_tr - lidar_ht_vec(npx)) > P.CAND_LIDAR_THR
                rej_lidar = rej_lidar + 1; continue;
            end

            n_filled = n_filled + 1;
            all_cands(n_filled, :) = [npx, ti, lab, cs];
        end
    end
end
all_cands = all_cands(1:n_filled, :);   % 截断到实际填充长度
fprintf('  检查%d → 光谱拒%d, LiDAR拒%d → 通过%d (含重复) [%.1fs]\n',...
    n_checked, rej_spec, rej_lidar, n_filled, toc(t_p1));

% ═════════════════════════════════════════════════════════════════════
% Phase 2: Label Consensus
%   对每个唯一候选像素 q, 检查所有选中它的 anchor 标签是否一致
%   - 全部一致 → 保留, 标签 = 该一致标签, 相似度 = max
%   - 存在冲突 → 丢弃
% ═════════════════════════════════════════════════════════════════════
fprintf('\n  ── Phase 2: Label Consensus 冲突剔除 ──\n');
t_p2 = tic;

[unique_q, ~, q_group] = unique(all_cands(:,1));
n_unique = length(unique_q);

pseudo_idx = zeros(n_unique, 1);
pseudo_lab = zeros(n_unique, 1);
pseudo_sim = zeros(n_unique, 1);
n_pseudo = 0;

% 诊断变量
n_singleton      = 0;   % 仅被 1 个锚点选中
n_multi_agree    = 0;   % 多锚点一致
n_multi_conflict = 0;   % 多锚点冲突 (丢弃)
n_anchors_per_q  = zeros(n_unique, 1);
conflict_matrix  = zeros(NFC, NFC);

for k = 1:n_unique
    mask = (q_group == k);
    labs_for_q = all_cands(mask, 3);
    sims_for_q = all_cands(mask, 4);
    n_a = length(labs_for_q);
    n_anchors_per_q(k) = n_a;

    unique_labs = unique(labs_for_q);

    if length(unique_labs) == 1
        % 一致 → 保留
        n_pseudo = n_pseudo + 1;
        pseudo_idx(n_pseudo) = unique_q(k);
        pseudo_lab(n_pseudo) = unique_labs(1);
        pseudo_sim(n_pseudo) = max(sims_for_q);

        if n_a == 1, n_singleton = n_singleton + 1;
        else,        n_multi_agree = n_multi_agree + 1; end
    else
        % 冲突 → 丢弃, 记录类对
        n_multi_conflict = n_multi_conflict + 1;
        for ii = 1:length(unique_labs)
            for jj = ii+1:length(unique_labs)
                a = unique_labs(ii); b = unique_labs(jj);
                ami = min(a,b); amx = max(a,b);
                conflict_matrix(ami, amx) = conflict_matrix(ami, amx) + 1;
            end
        end
    end
end
pseudo_idx = pseudo_idx(1:n_pseudo);
pseudo_lab = pseudo_lab(1:n_pseudo);
pseudo_sim = pseudo_sim(1:n_pseudo);
fprintf('  Phase 2 完成 [%.1fs]\n', toc(t_p2));

% ═════════════════════════════════════════════════════════════════════
% Phase 2 诊断输出
% ═════════════════════════════════════════════════════════════════════
fprintf('\n  ┌─── Label Consensus 诊断 ─────────────────────────────────┐\n');
fprintf('  │ 唯一候选像素总数:               %5d                       │\n', n_unique);
fprintf('  │ ① 单锚点选中 (无冲突可能):      %5d (%5.1f%%)              │\n',...
    n_singleton, 100*n_singleton/max(n_unique,1));
fprintf('  │ ② 多锚点一致 (consensus 通过):  %5d (%5.1f%%)              │\n',...
    n_multi_agree, 100*n_multi_agree/max(n_unique,1));
fprintf('  │ ③ 多锚点冲突 (剔除):            %5d (%5.1f%%)              │\n',...
    n_multi_conflict, 100*n_multi_conflict/max(n_unique,1));
fprintf('  │                                                            │\n');
fprintf('  │ ★ Consensus 保留总数 (①+②):    %5d                       │\n',...
    n_singleton + n_multi_agree);
fprintf('  │ ★ Consensus 剔除占比:           %5.1f%%                     │\n',...
    100*n_multi_conflict/max(n_unique,1));
fprintf('  └────────────────────────────────────────────────────────────┘\n');

% 多锚点重合度分布
n_multi = sum(n_anchors_per_q > 1);
fprintf('\n  锚点重合度: %d/%d (%.1f%%) 候选被多于一个锚点选中\n',...
    n_multi, n_unique, 100*n_multi/max(n_unique,1));
if n_multi > 0
    max_overlap = max(n_anchors_per_q);
    fprintf('  最大重合度: %d 个锚点同时选中同一像素\n', max_overlap);
end

% 冲突类对 top-N
if n_multi_conflict > 0
    fprintf('\n  最常见的类间冲突 (top 10):\n');
    [a_arr, b_arr] = find(conflict_matrix > 0);
    v_arr = arrayfun(@(i,j) conflict_matrix(i,j), a_arr, b_arr);
    [v_sorted, sidx] = sort(v_arr, 'descend');
    n_show = min(10, length(v_sorted));
    fprintf('    %-20s ↔  %-20s %6s\n', '类 A', '类 B', '冲突数');
    fprintf('    %s\n', repmat('-', 1, 56));
    for kk = 1:n_show
        a = a_arr(sidx(kk)); b = b_arr(sidx(kk));
        fprintf('    %-20s ↔  %-20s %6d\n',...
            CONFIG.FINE_NAMES{a}, CONFIG.FINE_NAMES{b}, v_sorted(kk));
    end
end

% 逐类伪样本统计
fprintf('\n  逐类伪样本统计:\n');
fprintf('    %-4s %-22s %8s %8s %8s\n','ID','类别','原始','保留','放大倍数');
fprintf('    %s\n', repmat('-', 1, 60));
for f = 1:NFC
    n_orig = sum(y_fine_train == f);
    n_kept = sum(pseudo_lab == f);
    if n_orig == 0 && n_kept == 0, continue; end
    if n_orig > 0
        amp = n_kept / n_orig;
        fprintf('    %-4d %-22s %8d %8d %7.1fx\n',...
            f, CONFIG.FINE_NAMES{f}, n_orig, n_kept, amp);
    end
end

% ═════════════════════════════════════════════════════════════════════
% 构建 L1.5 训练集
% ═════════════════════════════════════════════════════════════════════
L15i = [train_px_idx; pseudo_idx];
L15l = [y_fine_train;  pseudo_lab];

fprintf('\n  L1=%d → L1.5=%d (+%d伪样本)\n',...
    length(train_px_idx), length(L15i), n_pseudo);

% 纯度评估
gt_vec = double(gt_fine(:));
if n_pseudo > 0
    m = (gt_vec(pseudo_idx) > 0);
    if any(m)
        purity = mean(gt_vec(pseudo_idx(m)) == pseudo_lab(m));
        fprintf('  纯度: %.2f%% (n=%d)\n', purity*100, sum(m));
    end
end

%% ═══ [5] RF 训练 (普通 RF) ══════════════════════════════════
fprintf('\n[5] RF 训练 (普通 RF)...\n');

Xs1=stdf(X_FULL,train_px_idx);
rf1=trf(Xs1, train_px_idx, y_fine_train, P);
P1=brfp(rf1, Xs1, NFC);

Xs15=stdf(X_FULL,L15i);
rf15=trf(Xs15, L15i, L15l, P);
P15=brfp(rf15, Xs15, NFC);

%% ═══ [6] Sobel-dual MRF (纯邻域加权传播) ══════════════════════
if P.USE_MRF
    fprintf('\n[6] Sobel-dual MRF (iter=%d, 纯邻域加权传播)...\n',...
        P.MRF_N_ITER);
    Pf = mrf_sobel_dual(P15, HGradIm, LGradIm, H, W, P.MRF_N_ITER);
else
    Pf = P15;
end

%% ═══ [7] 三阶段评估 ═════════════════════════════════════════════════
fprintf('\n[7] 三阶段评估...\n');

gt_vec = double(gt_fine(:));
eval_mask = (gt_vec > 0);
eval_mask(train_px_idx) = false;
eval_idx   = find(eval_mask);
eval_label = gt_vec(eval_idx);
n_eval = length(eval_idx);

n_pseudo_in_eval = sum(ismember(pseudo_idx, eval_idx));
fprintf('  评估集: %d像素 (全图GT - %d原始训练)\n', n_eval, length(train_px_idx));
fprintf('  其中%d个伪样本在评估集中 (参与训练但仍参与精度评估)\n', n_pseudo_in_eval);

fprintf('\n  ── (a) L1: 仅原始训练 (%d样本) ──\n', length(train_px_idx));
[~, pv_L1] = max(P1, [], 2);
pred_L1 = reshape(pv_L1(:), H, W);
yp_L1 = pv_L1(eval_idx);
[OA_L1, AA_L1, Kap_L1, CA_L1, UA_L1, CM_L1] = accf(eval_label, yp_L1, 1:NFC);
print_acc('L1', OA_L1, AA_L1, Kap_L1, CA_L1, UA_L1, NFC, CONFIG, eval_label);

fprintf('\n  ── (b) L1.5: 论文版 Consensus 扩充 (%d样本, +%d伪) ──\n', length(L15i), n_pseudo);
[~, pv_L15] = max(P15, [], 2);
pred_L15 = reshape(pv_L15(:), H, W);
yp_L15 = pv_L15(eval_idx);
[OA_L15, AA_L15, Kap_L15, CA_L15, UA_L15, CM_L15] = accf(eval_label, yp_L15, 1:NFC);
print_acc('L1.5', OA_L15, AA_L15, Kap_L15, CA_L15, UA_L15, NFC, CONFIG, eval_label);

fprintf('\n  ── (c) Final: + Sobel-dual MRF ──\n');
[~, pv_final] = max(Pf, [], 2);
pred_final = reshape(pv_final(:), H, W);
yp_final = pv_final(eval_idx);
[OA_final, AA_final, Kap_final, CA_final, UA_final, CM_final] = accf(eval_label, yp_final, 1:NFC);
print_acc('Final', OA_final, AA_final, Kap_final, CA_final, UA_final, NFC, CONFIG, eval_label);

fprintf('\n  ╔══════════════════════════════════════════════════════╗\n');
fprintf('  ║   %s三阶段精度汇总   ║\n', DATASET_NAME);
fprintf('  ╠════════════════╦════════╦════════╦════════════════╣\n');
fprintf('  ║ 阶段           ║   OA   ║   AA   ║   Kappa        ║\n');
fprintf('  ╠════════════════╬════════╬════════╬════════════════╣\n');
fprintf('  ║ (a) L1         ║ %5.2f%% ║ %5.2f%% ║ %.4f         ║\n', OA_L1*100, AA_L1*100, Kap_L1);
fprintf('  ║ (b) L1.5       ║ %5.2f%% ║ %5.2f%% ║ %.4f         ║\n', OA_L15*100, AA_L15*100, Kap_L15);
fprintf('  ║ (c) Final      ║ %5.2f%% ║ %5.2f%% ║ %.4f  ★      ║\n', OA_final*100, AA_final*100, Kap_final);
fprintf('  ╚════════════════╩════════╩════════╩════════════════╝\n');

fprintf('\n  逐类PA对比:\n');
fprintf('  %-4s %-24s %8s %8s %8s %6s\n','ID','类别','L1','L1.5','Final','n');
fprintf('  %s\n', repmat('-',1,64));
for f=1:NFC
    ns=sum(eval_label==f); if ns==0, continue; end
    fprintf('  %-4d %-24s %7.2f%% %7.2f%% %7.2f%% %6d\n',...
        f, CONFIG.FINE_NAMES{f}, CA_L1(f)*100, CA_L15(f)*100, CA_final(f)*100, ns);
end

%% ═══ [8] 保存 ══════════════════════════════════════════════════════
fprintf('\n[8] 保存...\n');

pred_L1    = uint8(pred_L1);
pred_L15   = uint8(pred_L15);
pred_final = uint8(pred_final);

ACC_L1    = struct('OA',OA_L1, 'AA',AA_L1, 'Kappa',Kap_L1, 'CA',CA_L1, 'UA',UA_L1, 'CM',CM_L1);
ACC_L15   = struct('OA',OA_L15, 'AA',AA_L15, 'Kappa',Kap_L15, 'CA',CA_L15, 'UA',UA_L15, 'CM',CM_L15);
ACC_final = struct('OA',OA_final, 'AA',AA_final, 'Kappa',Kap_final, 'CA',CA_final, 'UA',UA_final, 'CM',CM_final);

CONSENSUS_DIAG = struct(...
    'n_unique',          n_unique,...
    'n_singleton',       n_singleton,...
    'n_multi_agree',     n_multi_agree,...
    'n_multi_conflict',  n_multi_conflict,...
    'conflict_rate',     n_multi_conflict/max(n_unique,1),...
    'n_pseudo_kept',     n_pseudo,...
    'conflict_matrix',   conflict_matrix,...
    'n_anchors_per_q',   n_anchors_per_q);

imwrite(lbl2rgb(double(pred_L1),1:NFC,CONFIG.FINE_COLORS),...
    fullfile(OUT_DIR,sprintf('pred_%s_L1.png',DATASET_NAME)));
imwrite(lbl2rgb(double(pred_L15),1:NFC,CONFIG.FINE_COLORS),...
    fullfile(OUT_DIR,sprintf('pred_%s_L15.png',DATASET_NAME)));
imwrite(lbl2rgb(double(pred_final),1:NFC,CONFIG.FINE_COLORS),...
    fullfile(OUT_DIR,sprintf('pred_%s_final.png',DATASET_NAME)));

save(fullfile(OUT_DIR,sprintf('results_%s_paper.mat',DATASET_NAME)),...
    'pred_L1','pred_L15','pred_final',...
    'gt_fine',...
    'ACC_L1','ACC_L15','ACC_final',...
    'CONSENSUS_DIAG',...
    'eval_idx','eval_label',...
    'yp_L1','yp_L15','yp_final',...
    'P1','P15','Pf',...
    'L15i','L15l','n_pseudo','pseudo_idx','pseudo_lab','pseudo_sim',...
    'n_pseudo_in_eval',...
    'HGradIm','LGradIm',...
    'CONFIG','DATASET_NAME','H','W',...
    'train_px_idx','test_px_idx','P','-v7.3');

fprintf('\n  已保存 results_%s_paper.mat (含 CONSENSUS_DIAG 诊断结构体)\n', DATASET_NAME);
fprintf('  %s 完成 — Final OA=%.2f%%  L1(%d)→L1.5(%d) [Consensus] 耗时%.1fs\n',...
    DATASET_NAME, OA_final*100, length(train_px_idx), length(L15i), toc(t_start));


%% ═════════════════════════════════════════════════════════════════════
%%  辅助函数
%% ═════════════════════════════════════════════════════════════════════

function print_acc(tag, OA, AA, Kap, CA, UA, NFC, CONFIG, eval_label)
    fprintf('    OA=%.2f%%  AA=%.2f%%  Kappa=%.4f\n', OA*100, AA*100, Kap);
    fprintf('    %-4s %-24s %7s %7s %6s\n','ID','类别','PA(CA)','UA','n');
    for f=1:NFC
        ns=sum(eval_label==f); if ns==0, continue; end
        fl=''; if CA(f)<0.80, fl=' ←'; end
        fprintf('    %-4d %-24s %6.2f%% %6.2f%% %6d%s\n',...
            f, CONFIG.FINE_NAMES{f}, CA(f)*100, UA(f)*100, ns, fl);
    end
end

function d=lv(f)
    s=load(f); fn=fieldnames(s); d=s.(fn{1});
end

function gt = load_gt(ds, CONFIG)
    switch ds
        case 'Houston2013'
            d=load(CONFIG.GT_FILE); fn=fieldnames(d); gt=double(d.(fn{1}));
        case 'Houston2018'
            load('huston2018_downgt.mat','huston2018_downgt');
            gt=double(huston2018_downgt);
        case 'Trento'
            load('trento_gt.mat','trento_gt'); gt=double(trento_gt);
        case 'Augsburg'
            load('Augsburggt.mat','Augsburggt'); gt=double(Augsburggt);
        otherwise
            d=load(CONFIG.GT_FILE); fn=fieldnames(d); gt=double(d.(fn{1}));
    end
end

function X = load_spatial(fn, NP, H, W)
    X=zeros(NP,0);
    if ~exist(fn,'file'), fprintf('    ⚠ %s 未找到\n',fn); return; end
    s=load(fn); f=fieldnames(s); raw=double(s.(f{1}));
    if ndims(raw)==3
        [h,w,~]=size(raw);
        if h==H&&w==W, raw=reshape(raw,NP,[]);
        elseif w==H&&h==W, raw=reshape(permute(raw,[2,1,3]),NP,[]);
        else, fprintf('    ⚠ %s 尺寸不匹配\n',fn); return; end
    elseif ismatrix(raw)
        if size(raw,1)~=NP
            if size(raw,2)==NP, raw=raw';
            else, fprintf('    ⚠ %s 行列不匹配N_PIX\n',fn); return; end
        end
    end
    raw(~isfinite(raw))=0; X=raw;
    fprintf('    + %s: %d维\n',fn,size(X,2));
end

function Xp=pcr(X,fi,k)
    [~,d]=size(X); if d==0, Xp=zeros(size(X,1),0); return; end
    k=min(k,d); mu=mean(X(fi,:),1);
    if d<=k, Xp=bsxfun(@minus,X,mu); return; end
    [~,~,V]=svds(bsxfun(@minus,X(fi,:),mu),k);
    Xp=bsxfun(@minus,X,mu)*V;
end

function Xs=stdf(X,ri)
    mu=mean(X(ri,:),1); sg=std(X(ri,:),0,1); sg(sg<1e-8)=1;
    Xs=bsxfun(@rdivide,bsxfun(@minus,X,mu),sg); Xs(~isfinite(Xs))=0;
end

function rf=trf(Xs,idx,y,P)
    y=y(:);
    np=max(2,round(P.RF_NUMPRED*sqrt(size(Xs,2))));
    rng(42);
    rf=TreeBagger(P.RF_GLOBAL,Xs(idx,:),y,'Method','classification',...
        'MinLeafSize',P.RF_MINLEAF,'NumPredictorsToSample',np,...
        'OOBPrediction','on');
    fprintf('    RF: n=%d, %dD (普通RF, 无加权)\n',length(y),size(Xs,2));
end

function Po=brfp(rf,X,NFC)
    BS=30000; N=size(X,1); cn=str2double(rf.ClassNames);
    Po=zeros(N,NFC,'single');
    for s=1:BS:N, e=min(s+BS-1,N);
        [~,sc]=rf.predict(X(s:e,:));
        for k=1:length(cn), c=cn(k);
            if c>=1&&c<=NFC, Po(s:e,c)=single(sc(:,k)); end
        end
    end
    Z=sum(Po,2); Z(Z<1e-12)=1; Po=bsxfun(@rdivide,Po,Z);
end

function GradIm = compute_sobel_grad(data3d)
    [~, ~, nb] = size(data3d);
    GIh = zeros(size(data3d,1), size(data3d,2), nb);
    GIv = zeros(size(data3d,1), size(data3d,2), nb);
    for i = 1:nb
        GIh(:,:,i) = double(edge(data3d(:,:,i), 'sobel', 'horizontal'));
        GIv(:,:,i) = double(edge(data3d(:,:,i), 'sobel', 'vertical'));
    end
    GradIm = 0.5 * (exp(-sum(GIh, 3)) + exp(-sum(GIv, 3)));
    mx = max(GradIm(:));
    if mx > 0, GradIm = GradIm / mx; end
end

function Po = mrf_sobel_dual(Pi, HGrad, LGrad, H, W, ni)
    K = size(Pi, 2);
    Pu = reshape(Pi, H, W, K);
    Pc = Pu;
    off = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];

    for it = 1:ni
        ma = zeros(H, W, K, 'single');
        wa = zeros(H, W, 'single');
        for oi = 1:8
            di = off(oi,1); dj = off(oi,2);
            Pb = circshift(Pc, [-di,-dj,0]);
            Hn = circshift(HGrad, [-di,-dj]);
            Ln = circshift(LGrad, [-di,-dj]);

            vl = true(H,W);
            if di<0, vl(end+di+1:end,:)=false; end
            if di>0, vl(1:di,:)=false; end
            if dj<0, vl(:,end+dj+1:end)=false; end
            if dj>0, vl(:,1:dj)=false; end

            w_h = min(HGrad, Hn);
            w_l = min(LGrad, Ln);
            ww  = single(w_h .* w_l .* vl);

            wa = wa + ww;
            for k=1:K
                ma(:,:,k) = ma(:,:,k) + ww .* Pb(:,:,k);
            end
        end

        ws = wa + 1e-8;
        for k=1:K, ma(:,:,k) = ma(:,:,k) ./ ws; end

        % 纯邻域加权传播 (无 λ): P^{t+1} = (∑ ω·P^t) / (∑ ω)
        % 边界处 (无有效邻居) 保持原值不变
        nb = wa < 1e-6;
        Pn = ma;
        for k=1:K
            t=Pn(:,:,k); u=Pu(:,:,k); t(nb)=u(nb); Pn(:,:,k)=t;
        end

        Z = sum(Pn,3) + 1e-8;
        for k=1:K, Pn(:,:,k) = Pn(:,:,k) ./ Z; end
        Pc = Pn;
    end
    Po = reshape(Pc, H*W, K);
end

function [oa,aa,kap,pa,ua,cm]=accf(yt,yp,cls)
    yt=yt(:); yp=yp(:); n=length(cls); cm=zeros(n);
    for i=1:n, for j=1:n, cm(i,j)=sum(yt==cls(i)&yp==cls(j)); end; end
    N=sum(cm(:)); oa=trace(cm)/max(N,1);
    pe=sum(sum(cm,2).*sum(cm,1)')/max(N^2,1);
    kap=(oa-pe)/max(1-pe,1e-10);
    pa=diag(cm)./max(sum(cm,2),1e-10);
    ua=diag(cm)./max(sum(cm,1)',1e-10);
    aa=mean(pa(pa>0));
end

function d=norm01(d)
    mn=min(d(:)); mx=max(d(:));
    if mx>mn, d=(d-mn)/(mx-mn); else, d=zeros(size(d)); end
end

function rgb=lbl2rgb(lm,ids,c)
    [H,W]=size(lm); rgb=zeros(H,W,3,'uint8');
    for k=1:length(ids)
        if k<=size(c,1), m=(lm==ids(k));
            for ch=1:3, l=rgb(:,:,ch); l(m)=uint8(c(k,ch)*255); rgb(:,:,ch)=l; end
        end
    end
end

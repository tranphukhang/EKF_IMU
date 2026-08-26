clc;
clear;
close all;

%% ============================================================
%  ĐỌC DỮ LIỆU
% =============================================================
fileName = 'esekf_simulation.csv';
data = readtable(fileName);

t = data.time;

%% ============================================================
%  GROUND TRUTH - MUJOCO IMU SITE
% =============================================================

% Position
gt_px = data.gt_px;
gt_py = data.gt_py;
gt_pz = data.gt_pz;

% Velocity
gt_vx = data.gt_vx;
gt_vy = data.gt_vy;
gt_vz = data.gt_vz;

% Quaternion: [qw qx qy qz]
gt_qw = data.gt_qw;
gt_qx = data.gt_qx;
gt_qy = data.gt_qy;
gt_qz = data.gt_qz;

%% ============================================================
%  ESEKF ESTIMATE
% =============================================================

% Position
est_px = data.est_px;
est_py = data.est_py;
est_pz = data.est_pz;

% Velocity
est_vx = data.est_vx;
est_vy = data.est_vy;
est_vz = data.est_vz;

% Quaternion
est_qw = data.est_qw;
est_qx = data.est_qx;
est_qy = data.est_qy;
est_qz = data.est_qz;


%% ============================================================
%  TẠO FIGURE
% =============================================================

figure('Name', 'IMU site trajectory', ...
       'Color', 'w', ...
       'Position', [100 80 1450 800]);

tl = tiledlayout(2,2, ...
    'TileSpacing', 'compact', ...
    'Padding', 'compact');


%% ============================================================
%  1. QUỸ ĐẠO 3D
% =============================================================

nexttile;
hold on;
grid on;
box on;

plot3(gt_px, gt_py, gt_pz, ...
    'r-', ...
    'LineWidth', 1.6);

plot3(est_px, est_py, est_pz, ...
    '--', ...
    'Color', [1.0 0.45 0.0], ...
    'LineWidth', 1.6);

% Đánh dấu vị trí cuối
plot3(gt_px(end), gt_py(end), gt_pz(end), ...
    'ko', ...
    'MarkerFaceColor', 'k', ...
    'MarkerSize', 6, ...
    'HandleVisibility', 'off');

xlabel('X [m]', 'FontSize', 12);
ylabel('Y [m]', 'FontSize', 12);
zlabel('Z [m]', 'FontSize', 12);

title('Quỹ đạo IMU trong hệ world', ...
    'FontSize', 14, ...
    'FontWeight', 'normal');

legend('imu\_right\_foot', ...
       'ESEKF estimate', ...
       'Location', 'best');

view(3);
axis tight;

set(gca, 'FontSize', 11);


%% ============================================================
%  2. VỊ TRÍ THEO THỜI GIAN
% =============================================================

nexttile;
hold on;
grid on;
box on;

% Ground truth
plot(t, gt_px, 'r-', 'LineWidth', 1.4);
plot(t, gt_py, 'g-', 'LineWidth', 1.4);
plot(t, gt_pz, 'b-', 'LineWidth', 1.4);

% ESEKF
plot(t, est_px, 'r--', 'LineWidth', 1.4);
plot(t, est_py, 'g--', 'LineWidth', 1.4);
plot(t, est_pz, 'b--', 'LineWidth', 1.4);

xlabel('Thời gian [s]', 'FontSize', 12);
ylabel('Vị trí [m]', 'FontSize', 12);

title('Vị trí IMU theo thời gian', ...
    'FontSize', 14, ...
    'FontWeight', 'normal');

legend('x', 'y', 'z', ...
       'x+', 'y+', 'z+', ...
       'Location', 'best');

xlim([t(1) t(end)]);

set(gca, 'FontSize', 11);


%% ============================================================
%  3. VẬN TỐC IMU TRONG HỆ WORLD
% =============================================================

nexttile;
hold on;
grid on;
box on;

% Ground truth
plot(t, gt_vx, 'r-', 'LineWidth', 1.2);
plot(t, gt_vy, 'g-', 'LineWidth', 1.2);
plot(t, gt_vz, 'b-', 'LineWidth', 1.2);

% ESEKF estimate
plot(t, est_vx, 'r--', 'LineWidth', 1.2);
plot(t, est_vy, 'g--', 'LineWidth', 1.2);
plot(t, est_vz, 'b--', 'LineWidth', 1.2);

xlabel('Thời gian [s]', 'FontSize', 12);
ylabel('Vận tốc [m/s]', 'FontSize', 12);

title('Vận tốc IMU trong hệ world', ...
    'FontSize', 14, ...
    'FontWeight', 'normal');

legend('vx', 'vy', 'vz', ...
       'vx+', 'vy+', 'vz+', ...
       'Location', 'best');

xlim([t(1) t(end)]);

set(gca, 'FontSize', 11);


%% ============================================================
%  4. QUATERNION IMU -> WORLD
% =============================================================

nexttile;
hold on;
grid on;
box on;

% Ground truth
plot(t, gt_qw, 'k-', 'LineWidth', 1.3);
plot(t, gt_qx, 'r-', 'LineWidth', 1.3);
plot(t, gt_qy, 'g-', 'LineWidth', 1.3);
plot(t, gt_qz, 'b-', 'LineWidth', 1.3);

% ESEKF estimate
plot(t, est_qw, 'k--', 'LineWidth', 1.3);
plot(t, est_qx, 'r--', 'LineWidth', 1.3);
plot(t, est_qy, 'g--', 'LineWidth', 1.3);
plot(t, est_qz, 'b--', 'LineWidth', 1.3);

xlabel('Thời gian [s]', 'FontSize', 12);
ylabel('Giá trị quaternion', 'FontSize', 12);

title('Quaternion IMU \rightarrow world', ...
    'FontSize', 14, ...
    'FontWeight', 'normal');

legend('qw', 'qx', 'qy', 'qz', ...
       'qw+', 'qx+', 'qy+', 'qz+', ...
       'Location', 'best');

xlim([t(1) t(end)]);
ylim([-1.05 1.05]);

set(gca, 'FontSize', 11);
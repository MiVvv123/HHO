  
%% 激励轨迹优化 - 基于8参数tau123_global回归矩阵
% 使用 tau123_global.m 生成的 3x8 Regressor_Y_Final.mat
% 8个参数: Theta_const, Theta_22, Theta_33, Theta_cross, Theta_L4, Theta_L5, Mu_2, Mu_3
%
% 优化目标: 最小化观测矩阵条件数，提高参数辨识精度

clear;
close all
% p=parpool(4);
tic     %tic——toc用于计时
opt.T = 2;          % 信号时长 (缩短到2秒以支持高加速度)
opt.wf = 2*pi/opt.T;    % 信号频率 = π rad/s
opt.t_smp = 0.01;  % 采样时间 (缩短以保持采样点数)
opt.t = 0:opt.t_smp:opt.T;  % 时间序列 (1001点)
opt.N = 5;          % 谐波次数
opt.NL = 3;         %连杆数量
opt.n_params = 8;   % 参数数量
% 关节安全设置
% opt.q_min = -deg2rad([160  130   170  107   170  80 170]');
% opt.q_max =  deg2rad([160  130  170  154  170  260 170]');
opt.q_min = deg2rad([-80 -80 10]');
opt.q_max =  deg2rad([40 -10 100]');
opt.dq_max =  deg2rad([150 150 150]');        % 速度限制: ±140 deg/s (恢复原值，给加速度更多空间)
opt.ddq_max = deg2rad([1500 1500 1500]');    % 加速度限制: ±1500 deg/s²

q_offset = (opt.q_min+opt.q_max)/2;
% q_offset = deg2rad([150 120 230]');
% opt.q0 = deg2rad([0 90 0 90 0 0 ]');      %初始位置，可以是未知的，已知时：变量个数6*2*N，未知，变量个数：6*（2*N+1）
opt.q0 = q_offset;                          %将初始位置设为关节限位的中间值
opt.min = 1000;                             %用于保存最小的条件数和对应的傅里叶参数（暂时没有用到）    
opt.xmin = zeros(opt.NL*2*opt.N,1);
%% 遗传算法
% optns_ga = optimoptions('ga');
%     optns_ga.Display = 'iter';
%     optns_ga.PlotFcn = 'gaplotbestf'; % {'gaplotbestf', 'gaplotscores'}
%     optns_ga.MaxGenerations = 50;
%     optns_ga.PopulationSize = 400; % in each generation.
%     optns_ga.InitialPopulationRange = [-100; 100];
%     optns_ga.SelectionFcn = 'selectionroulette';
% A = []; b = [];
% Aeq = []; beq = [];
% lb = []; ub = [];
% load data x
% x0 = x;
% % x0 = -1*rand(opt.NL*2*opt.N,1)+2;
% [x,fval] = ga(@(x)calfitness(x, opt),6*2*opt.N,A,b,Aeq,beq,lb,ub,@(x)restraints(x, opt),optns_ga);

%% patternsearch求解
%     optns_pttrnSrch = optimoptions('patternsearch');
%     optns_pttrnSrch.Display = 'iter';
%     optns_pttrnSrch.StepTolerance = 1e-1;
%     optns_pttrnSrch.FunctionTolerance = 10;
%     optns_pttrnSrch.ConstraintTolerance = 1e-6;
%     optns_pttrnSrch.MaxTime = inf;
%     optns_pttrnSrch.MaxFunctionEvaluations = 1e+6;
%     A = []; b = [];
%     Aeq = []; beq = [];
%     lb = []; ub = [];
%     x0 = -1*rand(opt.NL*2*opt.N,1)+2;
%     [x,fval] = patternsearch(@(x)calfitness(x,opt), x0, ...
%                              A, b, Aeq, beq, lb, ub, ...
%                              @(x)restraints(x,opt), optns_pttrnSrch);

%% fmincon求解
optns_fi = optimoptions("fmincon");
optns_fi.Algorithm = 'sqp';           % SQP算法更稳定
optns_fi.Display = 'iter';
optns_fi.PlotFcn = 'optimplotfval';
optns_fi.StepTolerance = 1e-8;        % 放宽步长容差
optns_fi.OptimalityTolerance = 1e-6;  % 最优性容差
optns_fi.ConstraintTolerance = 1e-6;  % 约束容差
optns_fi.MaxFunctionEvaluations = 100000;
optns_fi.MaxIterations = 2000;

A = []; b = [];
Aeq = []; beq = [];
lb = []; ub = [];

% 随机初始化（周期改变后需要重新优化）
x0 = randn(opt.NL*2*opt.N, 1) * 0.5;

% 如需使用上次结果，取消下面注释：
% load data x
% x0 = x;

[x,fval,exitflag] = fmincon(@(x)calfitness(x, opt),x0,A,b,Aeq,beq,lb,ub,@(x)restraints(x, opt),optns_fi);

T = opt.T ;         % 信号时长
    wf = opt.wf ;    % 信号频率
    t= opt.t;  % 时间序列
    N= opt.N;          % 谐波次数
    NL = opt.NL;
    q0= opt.q0;     
    for i=1:NL
        for j=1:N
             %傅里叶轨迹参数
            a(i,j)=x(N*(i-1)+j);
            b(i,j)=x(opt.NL*opt.N+N*(i-1)+j);
        end
    end
%     a
%     b
save data x opt
[path,q,dq,ddq] = plotracjetory(x, opt);  %绘制轨迹

load data_q_33 q dq ddq
Rizon = creat_robot(); %创建JAKA机械臂

is_use = 1;
% for i =1:size(q,2)
% %      判断是否能碰到桌面
%          p1=Rizon.links(1).A(q(1,i));
%          p2=p1*Rizon.links(2).A(q(2,i));
%          p3=p2*Rizon.links(3).A(q(3,i));
%      if p1.t(3)<0.1||p2.t(3)<0.1||p3.t(3)<0.1
%          disp("*********************************会碰到桌子*******************************")
%          is_use = 0;
%      end
% end
if is_use==1
    fit = calfitness(x, opt);
    data = "data" + num2str(opt.N)+"_" + num2str(opt.T)+"_"+num2str(opt.t_smp)+"_"+num2str(fit)+".mat";
    save(data, 'x', 'opt')
    
    % === 修改这一行 ===
    % 旧代码：csvwrite(path,q');
    % 新代码：
    t_column = opt.t';
    q_deg = rad2deg(q');
    data_to_save = [t_column, q_deg];
    csvwrite(path, data_to_save);
    fprintf('✓ 轨迹已保存: %s (时间+位置角度)\n', path);
    % === 修改结束 ===
end


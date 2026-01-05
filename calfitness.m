%计算适应度函数（条件数优化）
%使用9参数最小参数集（来自tau123_global_minimal_fixed.m）
function out =  calfitness(x, opt)

    T = opt.T ;         % 信号时长
    wf = opt.wf ;       % 信号频率
    t= opt.t;           % 时间序列
    N= opt.N;           % 谐波次数
    NL = opt.NL;
    q0= opt.q0;     
    a=zeros(NL,N);
    b=zeros(NL,N);
    for i=1:NL
        for j=1:N
             %傅里叶轨迹参数
            a(i,j)=x(N*(i-1)+j);
            b(i,j)=x(opt.NL*opt.N+N*(i-1)+j);
        end
    end

    q=zeros(NL,length(t));dq=zeros(NL,length(t));ddq=zeros(NL,length(t));
    %傅里叶轨迹表达式
      for i=1:NL
        for k=1:N
            if(k==1)
                q(i,:) = q(i,:) + q0(i);
            end
            q(i,:) = q(i,:) + (a(i,k)*sin(k*wf*t) - b(i,k)*cos(k*wf*t))/(k*wf);
            dq(i,:) = dq(i,:) + (a(i,k)*cos(k*wf*t) + b(i,k)*sin(k*wf*t));
            ddq(i,:) = ddq(i,:) + wf*k*(-a(i,k)*sin(k*wf*t) + b(i,k)*cos(k*wf*t));
        end
      end

    n_params = 8;
    if isfield(opt, 'n_params')
        n_params = opt.n_params;
    end
    Yf = zeros(length(t)*NL, n_params);
    for j=1:length(t)
        Yf(NL*(j-1)+1:NL*j,:) = calYP_8params(q(:,j), dq(:,j), ddq(:,j));
    end
%     toc
%      tic
%      Yf = zeros(length(t)*NL,57);
%     Yff =zeros(NL,57,length(t));
%     parfor (j=1:length(t))
%         Yff(:,:,j) = calYP(q(:,j),dq(:,j),ddq(:,j));
% %         Yf(NL*(j-1)+1:NL*j,:)=calYP(q(:,j),dq(:,j),ddq(:,j));
%     end
%     
%      for j=1:length(t)
%         Yf(NL*(j-1)+1:NL*j,:)= Yff(:,:,j);
%     end
% toc
    %   disp("计算适应度")
    
    % 使用SVD计算有效条件数（处理秩亏矩阵）
    [~, S, ~] = svd(Yf, 'econ');
    singular_values = diag(S);
    
    % 设置奇异值阈值（相对于最大奇异值的比例）
    tol = max(size(Yf)) * eps(max(singular_values));
    effective_sv = singular_values(singular_values > tol);
    
    % 有效条件数 = 最大有效奇异值 / 最小有效奇异值
    if length(effective_sv) >= 2
        cond_val = effective_sv(1) / effective_sv(end);
    else
        cond_val = 1e10;  % 秩太低，给一个大惩罚
    end
    
    % D-最优准则：log(det(W'*W)) = 2*sum(log(singular_values))
    % 越大越好，取负值作为最小化目标
    log_det = 2 * sum(log(effective_sv + 1e-10));
    
    % 简化目标函数：仅最小化有效条件数
    % 加速度激励通过约束函数restraints.m中的加速度约束实现
    out = cond_val;

%     if out<=opt.min
%         opt.x = x;
%         opt.min = out;
%         save datamin x
%     end
end
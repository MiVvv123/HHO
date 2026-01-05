%Լ������
function [c,ceq] =  restraints(x, opt)
    T = opt.T ;         % �ź�ʱ��
    wf = opt.wf ;    % �ź�Ƶ��
    t= opt.t;  % ʱ������
    N= opt.N;          % г������
    q0= opt.q0;    
    NL = opt.NL;
    a=zeros(NL,N);
    b=zeros(NL,N);
    for i=1:NL
        for j=1:N
             %����Ҷ�켣����
            a(i,j)=x(N*(i-1)+j);
            b(i,j)=x(opt.NL*opt.N+N*(i-1)+j);
        end
    end
% 
    q=zeros(NL,length(t));dq=zeros(NL,length(t));ddq=zeros(NL,length(t));
%     ����Ҷ�켣����ʽ
%     parfor i=1:NL
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
% % 约束写法1 - 包含加速度约束
    c=zeros(1,4*NL);ceq=zeros(1,3*NL);
    c(1:NL) = opt.q_min - min(q,[],2); 
    c(NL+1:2*NL) = max(q,[],2) - opt.q_max; 
    c(2*NL+1:3*NL) = max(abs(dq),[],2) - opt.dq_max; % max joint velocity const
    c(3*NL+1:4*NL) = max(abs(ddq),[],2) - opt.ddq_max; % max joint acceleration constr (已启用)
%     c(4*NL+1:4*NL+length(x)) = 0.0001 - abs(x);      %��ֹx������0
    
    ceq(1:NL)=q(:,1) - q0;
    ceq(NL+1:2*NL)=dq(:,length(t));
    ceq(2*NL+1:3*NL)=ddq(:,1);
%     ceq(3*NL+1:4*NL)=ddq(:,length(t));


%Լ��д��2���ӿ��ٶȣ�ʹ����ѧ֪ʶ��������
%     sum(a,2)        %λ�ã����ٶȳ�ʼΪ0
%     sum(b,2)        %�ٶȣ���ʼΪ0

    %��֤�켣��λ�á��ٶȡ����ٶȳ�ʼ����Ϊ��
%     ceq = zeros(1,3*NL);
%     for i=1:NL
%         for j=1:N
%             ceq(i) =( ceq(i) + b(i,j)/(wf*j) );    %q
%             ceq(NL+i) =( ceq(NL+i) + a(i,j));     %dq
%             ceq(2*NL+i) = ( ceq(2*NL+i) + wf*j*b(i,j));   %ddq
%         end
%     end
%     %��֤�Ƕ�λ�á��ٶȡ����ٶ�Լ��
%     c = zeros(1,3*NL+length(x));   
%     for i=1:NL
%         for j=1:N
%             c(i) = c(i) + sqrt(a(i,j)^2 + b(i,j)^2)/(j*wf);   
%             c(NL+i) = c(NL+i) + sqrt(a(i,j)^2 + b(i,j)^2);
%             c(2*NL+i) = c(2*NL+i) + j*wf*sqrt(a(i,j)^2 + b(i,j)^2);
%         end
% %          c(3*NL+i) = (opt.q_max(i)-opt.q_min(i))/3 - c(i);  %�����˶��Ƕȣ�����׼ȷ��
%          c(i) =( c(i) - (opt.q_max(i)-opt.q_min(i)));   
%          c(NL+i) =( c(NL+i) - opt.dq_max(i));
%          c(2*NL+i) =( c(2*NL+i) - opt.ddq_max(i)); 
%     end
%      c(3*NL+1:3*NL+length(x)) = -abs(x); 

    %��ֹa��bϵ��Ϊ0
    
    
    
    
end
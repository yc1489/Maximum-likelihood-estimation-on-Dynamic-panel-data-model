rep = 2500;         % number of replications
list_T = [ 5 10];     % list of time, T
list_N = [ 50 250 ];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
list_power = [-0.20 -0.10 0.00 0.10 0.20]; % for power computation
bias_mean_gam=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));  % creat a space for saving bias in different setting
std_gam=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));        % creat a space for saving standard error in different setting
rmse_gam=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));       % creat a space for saving RMSE in different setting
se_gam=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));         % creat a space for saving bias in different setting
power1_gamma=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));    % creat a space for saving power(gamma-0.05) in different setting
power2_gamma=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));    % creat a space for saving power(gamma+0.05)  in different setting
power4_gamma=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));    % creat a space for saving power(gamma-0.05) in different setting
power5_gamma=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));    % creat a space for saving power(gamma+0.05)  in different setting
size_gamma=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));     % creat a space for saving size(gamma-0) in different setting

for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T


for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N

TT= (T0+1); % T* (because we want to discard first 50 units)
T = TT-1;        
sigmau=1; % seeting the variance of disturbance
size_power_gam_nsml   = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
delta_GMM0=zeros(rep,1);
se_GMM0=zeros(rep,1);
sml=1;
while sml<=rep
alpha=normrnd(0,sigmau,[N,1]);
y=zeros(N,TT);  % creat a space for saving data

y(:,1)=alpha+normrnd(0,1/(1-gamma^2), [N,1]);         % setting the initial value of data
for tt=2:TT
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,TT-T0:TT)';    

y  = Y_NT(2:T+1,:) ;
y1 = Y_NT(1:T,:) ;
K = 0;

% [T, N] = size(y);
T1=T-1;
T2=T-2;

m_DIF1 = T*(T-1)/2 + (T*(T+1)/2-1)*K;

% disp([m_DIF m_SYS]);

D = [-eye(T1) zeros(T1,1)] + [zeros(T1,1) eye(T1) ];
DD = D*D';
L = [zeros(T1,1) eye(T1)];

Zy_DIF1 = zeros(m_DIF1,1);  ZX_DIF1 = zeros(m_DIF1,K+1);  ZZ_DIF1 = zeros(m_DIF1,m_DIF1);      
Dy_N = zeros(T1,1,N);    DX_N = zeros(T1,K+1,N);
Sy_N = zeros(2*T1,1,N);  SX_N = zeros(2*T1,K+1,N);
Z_DIF1_N  = zeros(T1,m_DIF1,N);       
Dy = D*y;   Dy1 = D*y1;  
Ly = L*y;   Ly1 = L*y1;   

for i=1:N;
    Dyi = Dy(:,i);   DXi = [Dy1(:,i) ];
    Lyi = Ly(:,i);   LXi = [Ly1(:,i)];
    Syi = [Dyi; Lyi];   SXi =  [DXi; LXi];
    Dy_N(:,:,i) = Dyi;  DX_N(:,:,i) = DXi;
    Sy_N(:,:,i) = Syi;  SX_N(:,:,i) = SXi;

    Zi_DIF10 = 0;
    Zi_DIF20 = 0;
    Zi_LEV0 = 0;
   
    for t=1:T1
        if t==1; lag=1; end
        if t>=2; lag=2; end
        Zi_DIF10 = blkdiag(Zi_DIF10, [y1(1:t,i)']);         
    end

    
    Zi_DIF1 = Zi_DIF10(2:end,2:end) ;    
    Z_DIF1_N(:,:,i) = Zi_DIF1;
    

    Zy_DIF1 = Zy_DIF1 + Zi_DIF1'*Dyi;  
    ZX_DIF1 = ZX_DIF1 + Zi_DIF1'*DXi;  
    ZZ_DIF1 = ZZ_DIF1 + Zi_DIF1'*DD*Zi_DIF1;           

end;
ZX_DIF1 = ZX_DIF1/N;  
Zy_DIF1 = Zy_DIF1/N;  
ZZ_DIF1 = ZZ_DIF1/N;   


% 1step GMM
if m_DIF1 < N; invZZ_DIF1 = inv(ZZ_DIF1); GMM_DIF1_1step = (ZX_DIF1'*invZZ_DIF1*ZX_DIF1)\(ZX_DIF1'*invZZ_DIF1*Zy_DIF1);
else GMM_DIF1_1step = NaN(K+1,1); 
end

% 2step GMM
Zu_DIF1_1step_N = zeros(N,m_DIF1);

for i=1:N
    Dyi = Dy_N(:,:,i) ;    DXi = DX_N(:,:,i) ;
    Syi = Sy_N(:,:,i) ;    SXi = SX_N(:,:,i) ;
    Zi_DIF1 = Z_DIF1_N(:,:,i);
   
    ui_DIF1_1step = Dyi - DXi*GMM_DIF1_1step;
     
    Zu_DIF1_1step_N(i,:) = (Zi_DIF1'*ui_DIF1_1step)';    
end
Zu_DIF1_1step = mean(Zu_DIF1_1step_N)';   

Ome_DIF1_1step = Zu_DIF1_1step_N'*Zu_DIF1_1step_N/N - Zu_DIF1_1step*Zu_DIF1_1step';


if m_DIF1 < N; invOme_DIF1_1step = inv(Ome_DIF1_1step); 
GMM_DIF1_2step = (ZX_DIF1'*invOme_DIF1_1step*ZX_DIF1)\(ZX_DIF1'*invOme_DIF1_1step*Zy_DIF1);
else GMM_DIF1_2step =NaN(K+1,1); end

option = optimset('TolCon',0.001,'TolFun',0.0001,'MaxFunEvals', 2000,'LargeScale', 'off','Display','off');

if m_DIF1 < N;  [GMM_DIF1_CUE,fval,exitflag,output,grad,Hhat_DIF1]   = fminunc( @(beta)Obj_CUE( beta, Dy_N, DX_N, Z_DIF1_N ), GMM_DIF1_1step, option); else GMM_DIF1_CUE = NaN(K+1,1); end

% GMM_DIF_CUE   = GMM_DIF_2step;
% GMM_SYS_CUE   = GMM_SYS_2step;

coef_GMM_DIF1 = [GMM_DIF1_1step   GMM_DIF1_2step   GMM_DIF1_CUE];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute standard errors
Zu_DIF1_2step_N = zeros(N,m_DIF1);   
Zu_DIF1_CUE_N = zeros(N,m_DIF1);     
dW_DIF1 = zeros(m_DIF1,m_DIF1,K+1);   
ZXuZ_DIF1 = zeros(m_DIF1, m_DIF1,K+1);   

for i=1:N
    Dyi = Dy_N(:,:,i) ;    DXi = DX_N(:,:,i) ;
    Syi = Sy_N(:,:,i) ;    SXi = SX_N(:,:,i) ;
    ui_DIF1_1step = Dyi - DXi*GMM_DIF1_1step;
    
    ui_DIF1_2step = Dyi - DXi*GMM_DIF1_2step;
    
    ui_DIF1_CUE = Dyi - DXi*GMM_DIF1_CUE;
 
    Zi_DIF1 = Z_DIF1_N(:,:,i);    
    
    Zu_DIF1_2step_N(i,:) = (Zi_DIF1'*ui_DIF1_2step)';
    
    Zu_DIF1_CUE_N(i,:) = (Zi_DIF1'*ui_DIF1_CUE)';
    
for j=1:K+1
    dW_DIF1(:,:,j) = dW_DIF1(:,:,j) + Zi_DIF1'*(DXi(:,j)*ui_DIF1_1step' + ui_DIF1_1step*DXi(:,j)')*Zi_DIF1;
    
   ZXuZ_DIF1(:,:,j) = ZXuZ_DIF1(:,:,j) + Zi_DIF1'*DXi(:,j)*ui_DIF1_CUE'*Zi_DIF1;
end
end
Zu_DIF1_2step = mean(Zu_DIF1_CUE_N)';   

dW_DIF1 = -dW_DIF1/N;
ZXuZ_DIF1 = ZXuZ_DIF1/N;    


if m_DIF1 < N; 
D_DIF1_2step = zeros(K+1,1);
for j=1:K+1
    D_DIF1_2step = [D_DIF1_2step, -(inv(ZX_DIF1'*invOme_DIF1_1step*ZX_DIF1)*(ZX_DIF1'*invOme_DIF1_1step*dW_DIF1(:,:,j)*invOme_DIF1_1step*Zu_DIF1_2step))];    
end
D_DIF1_2step = D_DIF1_2step(:,2:end);
var_DIF1_2step = (1/N)*inv(ZX_DIF1'*invOme_DIF1_1step*ZX_DIF1) ;
else
    var_DIF1_2step   = NaN(K+1,K+1);   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



delta_GMM0(sml,1)  = GMM_DIF1_2step;  % saving gamma in est_gam_nsml
se_GMM0(sml,1)  =abs(sqrt(var_DIF1_2step));   % saving standard error of gamma in se_gam_nsml


for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(delta_GMM0(sml,1) - (gamma-b))/se_GMM0(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end  

sml=sml+1;

  
end

mean_gam= nanmean(delta_GMM0);       % get mean of estimator base on all replication 
bias_mean_gam(idx_T, idx_N, idx_gam) = mean_gam - gamma;     % get bias of estimator in different setting
std_gam(idx_T, idx_N, idx_gam) = nanstd(delta_GMM0);       % get standard error of estimator in different setting  
rmse_gam(idx_T, idx_N, idx_gam) = sqrt( nanmean( (delta_GMM0-gamma).^2) );  % get root mean square error in different setting
se_gam(idx_T, idx_N, idx_gam)= nanmean(se_GMM0);                         % get mean of standard error of gamma in different setting
size_power_gam= nanmean(size_power_gam_nsml,2);                             % get mean of test result
size_gamma(idx_T, idx_N, idx_gam) = size_power_gam(3,1)';                   % get size in different setting in different setting
power1_gamma(idx_T, idx_N, idx_gam) = size_power_gam(1,1)';                 % get power in different setting in different setting
power2_gamma(idx_T, idx_N, idx_gam)=size_power_gam(2,1)';                   % get power in different setting in different setting
power4_gamma(idx_T, idx_N, idx_gam)=size_power_gam(4,1)';  
power5_gamma(idx_T, idx_N, idx_gam)=size_power_gam(5,1)';  


end
end
end


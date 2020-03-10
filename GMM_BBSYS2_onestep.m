rep = 2500;         % number of replications
list_T = [ 5 10];     % list of time, T
list_N = [ 50 500 ];   % list of N 
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
m_DIF2 = (T1*2 - 1) + ((T1*3 - 1)*K);

m_SYS2 = m_DIF2 + (K+1)*(T-1);

% disp([m_DIF m_SYS]);

D = [-eye(T1) zeros(T1,1)] + [zeros(T1,1) eye(T1) ];
DD = D*D';
L = [zeros(T1,1) eye(T1)];

  
 

Zy_SYS2 = zeros(m_SYS2,1);  ZX_SYS2 = zeros(m_SYS2,K+1);  ZZ_SYS2 = zeros(m_SYS2,m_SYS2);  

Dy_N = zeros(T1,1,N);    DX_N = zeros(T1,K+1,N);
Sy_N = zeros(2*T1,1,N);  SX_N = zeros(2*T1,K+1,N);
 Z_DIF2_N  = zeros(T1,m_DIF2,N);
Z_SYS2_N  = zeros(2*T1,m_SYS2,N); 
Dy = D*y;   Dy1 = D*y1;  
Ly = L*y;   Ly1 = L*y1;   

for i=1:N;
    Dyi = Dy(:,i);   DXi = [Dy1(:,i)];
    Lyi = Ly(:,i);   LXi = [Ly1(:,i) ];
    Syi = [Dyi; Lyi];   SXi =  [DXi; LXi];
    Dy_N(:,:,i) = Dyi;  DX_N(:,:,i) = DXi;
    Sy_N(:,:,i) = Syi;  SX_N(:,:,i) = SXi;

    Zi_DIF20 = 0;
    Zi_LEV0 = 0;
   
    for t=1:T1
        if t==1; lag=1; end
        if t>=2; lag=2; end
        Zi_DIF20 = blkdiag(Zi_DIF20, [y1(t-lag+1:t,i)'] ); 
        Zi_LEV0  = blkdiag(Zi_LEV0, [Dy1(t,i)' ]); 
    end

    

    Zi_DIF2 = Zi_DIF20(2:end,2:end) ;
    Zi_LEV  = Zi_LEV0(2:end,2:end);
    Zi_SYS2 = blkdiag(Zi_DIF2, Zi_LEV);
    Z_DIF2_N(:,:,i) = Zi_DIF2;
    Z_SYS2_N(:,:,i) = Zi_SYS2;
    
            
       
    Zy_SYS2 = Zy_SYS2 + Zi_SYS2'*Syi;    ZX_SYS2 = ZX_SYS2 + Zi_SYS2'*SXi;    ZZ_SYS2 = ZZ_SYS2 + blkdiag(Zi_DIF2'*DD*Zi_DIF2, Zi_LEV'*Zi_LEV);

end;
        ZX_SYS2 = ZX_SYS2/N;
      Zy_SYS2 = Zy_SYS2/N;
     ZZ_SYS2 = ZZ_SYS2/N;


% 1step GMM

if m_SYS2 < N; invZZ_SYS2 = inv(ZZ_SYS2); GMM_SYS2_1step = (ZX_SYS2'*invZZ_SYS2*ZX_SYS2)\(ZX_SYS2'*invZZ_SYS2*Zy_SYS2);
else GMM_SYS2_1step = NaN(K+1,1); 
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2step GMM


Zu_SYS2_1step_N = zeros(N,m_SYS2);

for i=1:N
    Dyi = Dy_N(:,:,i) ;    DXi = DX_N(:,:,i) ;
    Syi = Sy_N(:,:,i) ;    SXi = SX_N(:,:,i) ;

    Zi_SYS2 = Z_SYS2_N(:,:,i);
    
  
    ui_SYS2_1step = Syi - SXi*GMM_SYS2_1step;
    
    Zu_SYS2_1step_N(i,:) = (Zi_SYS2'*ui_SYS2_1step)';

end

 
Zu_SYS2_1step = mean(Zu_SYS2_1step_N)';   

Ome_SYS2_1step = Zu_SYS2_1step_N'*Zu_SYS2_1step_N/N - Zu_SYS2_1step*Zu_SYS2_1step';




if m_SYS2 < N; invOme_SYS2_1step = inv(Ome_SYS2_1step); GMM_SYS2_2step = (ZX_SYS2'*invOme_SYS2_1step*ZX_SYS2)\(ZX_SYS2'*invOme_SYS2_1step*Zy_SYS2);
else GMM_SYS2_2step =NaN(K+1,1); 
end



% GMM_DIF_CUE   = GMM_DIF_2step;
% GMM_SYS_CUE   = GMM_SYS_2step;


coef_GMM_SYS2 = [GMM_SYS2_1step ];

% compute standard errors

if m_SYS2 < N; 
var_SYS2_1step = (1/N)*inv(ZX_SYS2'*invZZ_SYS2*ZX_SYS2)*(ZX_SYS2'*invZZ_SYS2*Ome_SYS2_1step*invZZ_SYS2*ZX_SYS2)*inv(ZX_SYS2'*invZZ_SYS2*ZX_SYS2);
else
    var_SYS2_1step   = NaN(K+1,K+1);   
end

delta_GMM0(sml,1)  = GMM_SYS2_1step;  % saving gamma in est_gam_nsml
se_GMM0(sml,1)  =abs(sqrt( var_SYS2_1step ));   % saving standard error of gamma in se_gam_nsml


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










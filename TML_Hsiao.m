rep = 10;         % number of replications
list_T = [10];     % list of time, T
list_N = [500];   % list of N 
list_gam= [0.5 ];       % list of gamma 
list_power = [-0.20 -0.10 0.00 0.10 0.20]; % for power computation
bias_mean_gam=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));  % creat a space for saving bias in different setting
meanbun=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));  % creat a space for saving bias in different setting
std_gam=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));        % creat a space for saving standard error in different setting
rmse_gam=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));       % creat a space for saving RMSE in different setting
se_gam=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));         % creat a space for saving bias in different setting
power1_gamma=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));    % creat a space for saving power(gamma-0.05) in different setting
power2_gamma=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));    % creat a space for saving power(gamma+0.05)  in different setting
power4_gamma=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));    % creat a space for saving power(gamma-0.05) in different setting
power5_gamma=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));    % creat a space for saving power(gamma+0.05)  in different setting
size_gamma=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));     % creat a space for saving size(gamma-0) in different setting
Med=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));
IQR=zeros(size(list_T,2), size(list_N,2), size(list_gam,2));





K=0;
for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N
TT= (T0+1); % T* (because we want to discard first 50 units)
T = TT-1; 
sigmau=1;      % seeting the variance of disturbance

size_power_gam_nsml   = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);

sml=1;
while sml<=rep
alpha=normrnd(0,sigmau,[N,1]);
y=zeros(N,TT);  % creat a space for saving data

y(:,1)=alpha+normrnd(2,10/(1-gamma^2), [N,1]);         % setting the initial value of data
for tt=2:TT
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,TT-T0:TT)'; % discard first 50 observation 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T1=T-1;        %  previous perod of data
y = Y_NT(:,:); 
y1 = Y_NT(1:end-1,:);  
Dy = y(2:T+1,:) - y(1:T,:);   % diference data


DW = zeros(T,N,2);
        for i=1:N
            DW(:,i,:) =  [1,   0;
                        zeros(T1,1), Dy(1:T1,i) ]; 
            
        end


Omega = 2*eye(T)+diag(-1*ones(T-1,1),1)+diag(-1*ones(T-1,1),-1);
Phi = sort((1:T)','descend')*sort((1:T),'descend');
% compute starting values for MLE
% GMM using stacked IVs
D = [-eye(T-1) zeros(T-1,1)] + [zeros(T-1,1) eye(T-1) ];
DD = D*D';
m_ML = 3; 
Zy_ML = zeros(m_ML,1);  ZX_ML = zeros(m_ML,1);  ZZ_ML = zeros(m_ML,m_ML);

DX_GMM_N = zeros(T1,N,1);

for i=1:N;
    Dyi_GMM = Dy(2:T,i);   
  
       
        DXi_GMM = Dy(1:T-1,i);
        Zi_ML = [Y_NT(1:T-1,i) [zeros(1,1); Y_NT(1:T-2,i)] [zeros(2,1); Y_NT(1:T-3,i)] ];
    

          
    DX_GMM_N(:,i,:) = DXi_GMM;
    Zy_ML = Zy_ML + Zi_ML'*Dyi_GMM;   
    ZX_ML = ZX_ML + Zi_ML'*DXi_GMM;   
    ZZ_ML = ZZ_ML + Zi_ML'*DD*Zi_ML;
end;

delta_GMM0 = (ZX_ML'*inv(ZZ_ML)*ZX_ML)\(ZX_ML'*inv(ZZ_ML)*Zy_ML);
du = zeros(T1,N);
du = Dy(2:T,:) - Dy(1:T-1,:)*delta_GMM0(1);


sig2_u_ini = sum(sum(du.*du))/(2*N*(T-2));

y1 = Dy(1,:)';
    X1 = ones(N,1);
b_pi_ini = (X1'*X1)\(X1'*y1);

phi_ini = [b_pi_ini; delta_GMM0(1)];


MDE_omega = (T-1)/T;
for i=1:N
    Dyi = Dy(:,i);
    DWi = squeeze(DW(:,i,:));
    ri =  Dyi - DWi*phi_ini;
    MDE_omega = MDE_omega + ri'*Phi*ri/(sig2_u_ini*N*T^2);
end
Omega(1,1) = MDE_omega;
invOmega = inv(Omega);
 
num=0;  den=0;
 for i=1:N
    Dyi = Dy(:,i);
    DWi = squeeze(DW(:,i,:));
     num = num + DWi'*invOmega*Dyi;
     den = den + DWi'*invOmega*DWi;
 end
MDE_phi = den\num;
 
MDE_sigma = 0;
 for i=1:N
    Dyi = Dy(:,i);
    DWi = squeeze(DW(:,i,:));
    ri =  Dyi - DWi*MDE_phi;
     MDE_sigma = MDE_sigma+ ri'*invOmega*ri/(N*T);
 end
 
% var_MDE_phi = mde_sigma*inv(den);
% correct MDE
if MDE_omega < ((T-1)/T);  MDE_omega = ((T-1)/T)+0.001; end;
if MDE_phi(2,1) > 1; MDE_phi(2,1) = 0.999; end
 



   theta_ini = [MDE_phi;  MDE_omega;  MDE_sigma];

    lq = [ -inf;   -inf;    (T-1)/T;   0.0001 ];
    uq = [ inf;     inf;       inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = TMLopt(N,T,K,Dy,DW,Phi,Omega,theta_ini,lq,uq);

dim_phi = size(DW,3);

phi_ML   = theta(1:dim_phi);
omega_ML = theta(dim_phi+1,1);
sigma_ML = theta(dim_phi+2,1);
Omega(1,1) = omega_ML;
invOmega = pinv(Omega);

sig2_uhat = sigma_ML;

% compute asymptotic variance
A11 = zeros(dim_phi,dim_phi);
A21 = zeros(1,dim_phi);
B11 = zeros(dim_phi,dim_phi);
B22 = zeros(1,1);
B33 = zeros(1,1);
B21 = zeros(1,dim_phi);
B31 = zeros(1,dim_phi);
B32 = zeros(1,1);

q_num=0;
q_den=0;

for i=1:N
    Dyi = Dy(:,i);
    DWi = squeeze(DW(:,i,:));
    ri =  Dyi - DWi*phi_ML;
    q_num = q_num + ri'*ri; 
    q_den = q_den + (ri'*invOmega*ri)^2; 

    A11 = A11 + DWi'*invOmega*DWi;
    A21 = A21 + ri'*Phi*DWi;   
    B11 = B11 + DWi'*invOmega*(ri*ri')*invOmega*DWi;
    B22 = B22 + (ri'*Phi*ri/T)^2;
    B33 = B33 + (ri'*invOmega*ri/T)^2;
    B21 = B21 + (ri'*invOmega*DWi)*(ri'*Phi*ri);
    B31 = B31 + (ri'*invOmega*DWi)*(ri'*invOmega*ri);
    B32 = B32 + (ri'*Phi*ri/T)*(ri'*invOmega*ri/T);
end
g = 1+T*(omega_ML-1);

A11 = (1/(N*sig2_uhat))*A11;
A22 = T^2/(2*g^2);
A33 = T/(2*sig2_uhat^2);
A21 = (1/(g^2*N*sig2_uhat))*A21;
A12 = A21';
A31 = zeros(1,dim_phi);
A13 = A31';
A32 = T/(2*g*sig2_uhat);
A23 = A32';
A = [A11 A12 A13; 
     A21 A22 A23; 
     A31 A32 A33];


B11 = (1/(N*sig2_uhat^2))*B11;
B22 = (T^2/(4*g^4*sig2_uhat^2))*((1/N)*B22 - g^2*sig2_uhat^2);
B33 = (T^2/(4*sig2_uhat^4))*((1/N)*B33 - sig2_uhat^2);
B21 = (1/(2*N*g^2*sig2_uhat^2))*B21;
B31 = (1/(2*N*sig2_uhat^3))*B31;
B32 = (T^2/(4*g^2*sig2_uhat^3))*((1/N)*B32 - g*sig2_uhat^2);
B12 = B21';
B13 = B31';
B23 = B32';
B = [B11 B12 B13; 
     B21 B22 B23;
     B31 B32 B33];
var_theta= pinv(A)*B*pinv(A)/(N);

coef_ML    = theta(2,1);
var_ML = var_theta(2,2);



est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml


for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    



sml=sml+1;

  
end


IQR(idx_T, idx_N, idx_gam)=iqr(est_gam_nsml);
Med(idx_T, idx_N, idx_gam)=median(est_gam_nsml);
meanbun(idx_T, idx_N, idx_gam) = nanmean(est_gam_nsml);     % get bias of estimator in different setting
mean_gam= nanmean(est_gam_nsml);       % get mean of estimator base on all replication 
bias_mean_gam(idx_T, idx_N, idx_gam) = mean_gam - gamma;     % get bias of estimator in different setting
std_gam(idx_T, idx_N, idx_gam) = nanstd(est_gam_nsml);       % get standard error of estimator in different setting  
rmse_gam(idx_T, idx_N, idx_gam) = sqrt( nanmean( (est_gam_nsml-gamma).^2) );  % get root mean square error in different setting
se_gam(idx_T, idx_N, idx_gam)= nanmean(se_gam_nsml);                         % get mean of standard error of gamma in different setting
size_power_gam= nanmean(size_power_gam_nsml,2);                             % get mean of test result
size_gamma(idx_T, idx_N, idx_gam) = size_power_gam(3,1)';                   % get size in different setting in different setting
power1_gamma(idx_T, idx_N, idx_gam) = size_power_gam(1,1)';                 % get power in different setting in different setting
power2_gamma(idx_T, idx_N, idx_gam)=size_power_gam(2,1)';                   % get power in different setting in different setting
power4_gamma(idx_T, idx_N, idx_gam)=size_power_gam(4,1)';  
power5_gamma(idx_T, idx_N, idx_gam)=size_power_gam(5,1)';  
end
end
end

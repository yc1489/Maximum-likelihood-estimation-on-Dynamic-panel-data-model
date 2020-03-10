rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100  500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=0;
TT=T0; % T* (because we want to discard first 50 units) 
sigmau=1;      % seeting the variance of disturbance
size_power_gam_nsml  = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);

sml=1;
while sml<=rep
alpha=normrnd(0,sigmau,[N,1]);
y=zeros(N,TT);  % creat a space for saving data

y(:,1)=0;          % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);

% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('FEm_0r_1.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;

rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=0;
TT=T0; % T* (because we want to discard first 50 units) 
sigmau=10;      % seeting the variance of disturbance
size_power_gam_nsml  = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);

sml=1;
while sml<=rep
alpha=normrnd(0,sigmau,[N,1]);
y=zeros(N,TT);  % creat a space for saving data

y(:,1)=0;          % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);

% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('FEm_0r_10.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;

rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=1;
TT=T0; % T* (because we want to discard first 50 units) 
sigmau=1;      % seeting the variance of disturbance
size_power_gam_nsml  = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);

sml=1;
while sml<=rep
alpha=normrnd(0,sigmau,[N,1]);
y=zeros(N,TT);  % creat a space for saving data

y(:,1)=0;          % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);

% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('FEm_1r_1.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;


rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=1;
TT=T0; % T* (because we want to discard first 50 units) 
sigmau=10;      % seeting the variance of disturbance
size_power_gam_nsml  = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);

sml=1;
while sml<=rep
alpha=normrnd(0,sigmau,[N,1]);
y=zeros(N,TT);  % creat a space for saving data

y(:,1)=0;          % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);

% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('FEm_1r_10.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;


rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=5;
TT=T0; % T* (because we want to discard first 50 units) 
sigmau=1;      % seeting the variance of disturbance
size_power_gam_nsml  = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);

sml=1;
while sml<=rep
alpha=normrnd(0,sigmau,[N,1]);
y=zeros(N,TT);  % creat a space for saving data

y(:,1)=0;          % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);

% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('FEm_5r_1.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;


rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=5;
TT=T0; % T* (because we want to discard first 50 units) 
sigmau=10;      % seeting the variance of disturbance
size_power_gam_nsml  = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);

sml=1;
while sml<=rep
alpha=normrnd(0,sigmau,[N,1]);
y=zeros(N,TT);  % creat a space for saving data

y(:,1)=0;          % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);

% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('FEm_5r_10.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;


rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=0;
TT=T0; % T* (because we want to discard first 50 units) 
sigmau=1;      % seeting the variance of disturbance
size_power_gam_nsml  = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);

sml=1;
while sml<=rep
alpha=normrnd(0,sigmau,[N,1]);
y=zeros(N,TT);  % creat a space for saving data

y(:,1)=alpha+normrnd(0,1/(1-gamma^2), [N,1]);          % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);

% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('FESr_1.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;




rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=0;
TT=T0; % T* (because we want to discard first 50 units) 
sigmau=10;      % seeting the variance of disturbance
size_power_gam_nsml  = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);

sml=1;
while sml<=rep
alpha=normrnd(0,sigmau,[N,1]);
y=zeros(N,TT);  % creat a space for saving data

y(:,1)=alpha+normrnd(0,1/(1-gamma^2), [N,1]);          % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);

% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('FESr_10.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;




rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=0;
TT= T0; % T* (because we want to discard first 50 units) 
size_power_gam_nsml   = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);
tao=1;     % seeting the variance of disturbance
sml=1;
while sml<=rep
varu=0.5+rand(N,1);
u=zeros(N,TT);
for ii=1:N
u(ii,:)=normrnd(0,varu(ii),[1,TT]);
end   
varum=mean(u,2);  
eta=((varum*tao)/(TT^(-1)*varum+1))^(0.5);     
alpha=eta*(varu+normrnd(0,1,[N,1]));         

y=zeros(N,TT);  % creat a space for saving data
y(:,1)=0;           % setting the initial value of data        % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);



% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('HFEm_0r_1.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;


rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=0;
TT= T0; % T* (because we want to discard first 50 units) 
size_power_gam_nsml   = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);
tao=10;     % seeting the variance of disturbance
sml=1;
while sml<=rep
varu=0.5+rand(N,1);
u=zeros(N,TT);
for ii=1:N
u(ii,:)=normrnd(0,varu(ii),[1,TT]);
end   
varum=mean(u,2);  
eta=((varum*tao)/(TT^(-1)*varum+1))^(0.5);     
alpha=eta*(varu+normrnd(0,1,[N,1]));         

y=zeros(N,TT);  % creat a space for saving data
y(:,1)=0;           % setting the initial value of data        % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);



% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('HFEm_0r_1.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;


rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma  
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=1;
TT= T0; % T* (because we want to discard first 50 units) 
size_power_gam_nsml   = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);
tao=1;     % seeting the variance of disturbance
sml=1;
while sml<=rep
varu=0.5+rand(N,1);
u=zeros(N,TT);
for ii=1:N
u(ii,:)=normrnd(0,varu(ii),[1,TT]);
end   
varum=mean(u,2);  
eta=((varum*tao)/(TT^(-1)*varum+1))^(0.5);     
alpha=eta*(varu+normrnd(0,1,[N,1]));         

y=zeros(N,TT);  % creat a space for saving data
y(:,1)=0;           % setting the initial value of data        % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);



% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('HFEm_1r_1.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;


rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=1;
TT= T0; % T* (because we want to discard first 50 units) 
size_power_gam_nsml   = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);
tao=10;     % seeting the variance of disturbance
sml=1;
while sml<=rep
varu=0.5+rand(N,1);
u=zeros(N,TT);
for ii=1:N
u(ii,:)=normrnd(0,varu(ii),[1,TT]);
end   
varum=mean(u,2);  
eta=((varum*tao)/(TT^(-1)*varum+1))^(0.5);     
alpha=eta*(varu+normrnd(0,1,[N,1]));         

y=zeros(N,TT);  % creat a space for saving data
y(:,1)=0;           % setting the initial value of data        % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);



% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('HFEm_1r_10.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;



rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=5;
TT= T0; % T* (because we want to discard first 50 units) 
size_power_gam_nsml   = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);
tao=1;     % seeting the variance of disturbance
sml=1;
while sml<=rep
varu=0.5+rand(N,1);
u=zeros(N,TT);
for ii=1:N
u(ii,:)=normrnd(0,varu(ii),[1,TT]);
end   
varum=mean(u,2);  
eta=((varum*tao)/(TT^(-1)*varum+1))^(0.5);     
alpha=eta*(varu+normrnd(0,1,[N,1]));         

y=zeros(N,TT);  % creat a space for saving data
y(:,1)=0;           % setting the initial value of data        % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);



% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('HFEm_5r_1.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;



rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=5;
TT= T0; % T* (because we want to discard first 50 units) 
size_power_gam_nsml   = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);
tao=10;     % seeting the variance of disturbance
sml=1;
while sml<=rep
varu=0.5+rand(N,1);
u=zeros(N,TT);
for ii=1:N
u(ii,:)=normrnd(0,varu(ii),[1,TT]);
end   
varum=mean(u,2);  
eta=((varum*tao)/(TT^(-1)*varum+1))^(0.5);     
alpha=eta*(varu+normrnd(0,1,[N,1]));         

y=zeros(N,TT);  % creat a space for saving data
y(:,1)=0;           % setting the initial value of data        % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);



% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('HFEm_5r_10.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;



rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma

for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T
       
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=0;
TT= T0; % T* (because we want to discard first 50 units) 
size_power_gam_nsml   = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);
tao=1;     % seeting the variance of disturbance
sml=1;
while sml<=rep
varu=0.5+rand(N,1);
u=zeros(N,TT);
for ii=1:N
u(ii,:)=normrnd(0,varu(ii),[1,TT]);
end   
varum=mean(u,2);  
eta=((varum*tao)/(TT^(-1)*varum+1))^(0.5);     
alpha=eta*(varu+normrnd(0,1,[N,1]));         

y=zeros(N,TT);  % creat a space for saving data
y(:,1)=alpha+normrnd(0,1/(1-gamma^2), [N,1]);           % setting the initial value of data        % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   

[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);


coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);



% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('HFESr_1.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;


rep = 2500;         % number of replications
list_T = [5 10];     % list of time, T
list_N = [50 100 500];   % list of N 
list_gam= [0.5 0.8];       % list of gamma 
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


for idx_gam=1:size(list_gam,2)
gamma = list_gam(idx_gam);  % for loop of gamma
for idx_T=1:size(list_T,2)
T0 = list_T(idx_T);        % for loop of T      
for idx_N=1:size(list_N,2)     
N = list_N(idx_N);        % for loop of N       
m=0;
TT= T0; % T* (because we want to discard first 50 units) 
size_power_gam_nsml   = zeros(size(list_power,2),rep); % creat a space for saving test result(power and size) of every replication
est_gam_nsml=zeros(rep,1);
se_gam_nsml=zeros(rep,1);
tao=10;     % seeting the variance of disturbance
sml=1;
while sml<=rep
varu=0.5+rand(N,1);
u=zeros(N,TT);
for ii=1:N
u(ii,:)=normrnd(0,varu(ii),[1,TT]);
end   
varum=mean(u,2);  
eta=((varum*tao)/(TT^(-1)*varum+1))^(0.5);     
alpha=eta*(varu+normrnd(0,1,[N,1]));         

y=zeros(N,TT);  % creat a space for saving data
y(:,1)=alpha+normrnd(0,1/(1-gamma^2), [N,1]);           % setting the initial value of data        % setting the initial value of data
for tt=2:TT+m+1
   y(:,tt)=gamma*y(:,tt-1)+(1-gamma)*alpha+normrnd(0,1,[N,1]);  % gen data
end
Y_NT=y(:,m+1:end)'; % discard first 50 observation 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Dy = Y_NT(2:TT,:) - ones(TT-1,1)*Y_NT(1,:);   % diference data
Dy1=Y_NT(1:TT-1,:)- ones(TT-1,1)*Y_NT(1,:); 

   theta0 = [rand; rand; rand];

    lq = [ -1;   0.0001;   0.0001 ];
    uq = [ 1;    inf;     inf  ];
   
[theta, fval, exitflag, output, lambda, hessian] = Kruiniger_opt(N,T0,Dy,theta0,lq,uq);

coef_ML  = theta(1,1);
S_ML = theta(2,1);
Sv_ML = theta(3,1);



% compute asymptotic variance
Q=eye(TT-1)-ones(TT-1)/(TT-1);
invphi=(S_ML^(-1))*Q+(Sv_ML ^(-1))*(1/(TT-1))*ones(TT-1);
phy=0;
for ii=1:TT-2
phy=phy+((TT-1)^(-1))*((TT-1-ii)/ii)*(coef_ML)^(ii);
end
P=zeros(TT-1);
for idx_P=1:TT-1
for jdx_P=1:TT-1
if idx_P>jdx_P
   P(idx_P,jdx_P)=coef_ML^( idx_P-jdx_P-1);
else
    P(idx_P,jdx_P)=0;
end
end
end

V11=S_ML*sum(diag(P'*invphi*P))+Sv_ML*ones(1,TT-1)*P'*invphi*P*ones(TT-1,1);
V12=-(1/S_ML)*phy+(1/Sv_ML)*phy;
V21=V12;
V13=(1/Sv_ML)*(TT-1)*phy;
V31=V13;
V22=(TT-2)/(2*S_ML^(2))+1/(2*Sv_ML^(2));
V33=(TT-1)^(2)/(2*Sv_ML^(2)); 
V23=(TT-1)/(2*Sv_ML^(2));
V32=V23;

V=[V11 V12 V13 ;
   V21 V22 V23;
   V31 V32 V33];
var_theta= pinv(V)/(N);

var_ML = var_theta(1,1);

est_gam_nsml(sml,1)  = coef_ML;  % saving gamma in est_gam_nsml
se_gam_nsml(sml,1)  =abs(sqrt(var_ML));   % saving standard error of gamma in se_gam_nsml

for idx_power=1:size(list_power,2)  % do test for power and size of test, power=P(reject H_0 | H_1 is true ), size=P(reject H_0 | H_0 is true) 
b = list_power(idx_power);

    if abs(est_gam_nsml(sml,1) - (gamma-b))/se_gam_nsml(sml,1)> 1.96; 
        size_power_gam_nsml(idx_power, sml)=1;  end
end    


sml=sml+1;
end
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

save('HFESr_10.mat','bias_mean_gam','se_gam','rmse_gam','power1_gamma','power2_gamma','size_gamma','power4_gamma','power5_gamma');

clear all;

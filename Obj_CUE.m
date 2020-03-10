function [ Obj_val ] = Obj_CUE( beta, y_N, X_N, Z_N )
% function [ Obj_val ] = Obj_CUE( beta, Zy_N, ZX_N )


% K = size(ZX_N,1);
% N = size(ZX_N,3);
[P, K, N] = size(Z_N);
Zu_N = zeros(N,K);
for i=1:N
    ui = y_N(:,:,i) - X_N(:,:,i)*beta ;        
    Zi = Z_N(:,:,i) ;
    Zu_N(i,:) = ( Zi'*ui)';
end
Zu = mean(Zu_N)';
Zu_N = Zu_N-repmat(mean(Zu_N),N,1);

% Obj_val = N*Zu'*((Zu_N'*Zu_N/N)\Zu);
Obj_val = N*Zu'*pinv(Zu_N'*Zu_N/N)*Zu;



end


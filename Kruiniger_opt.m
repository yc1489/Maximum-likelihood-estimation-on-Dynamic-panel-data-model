function [theta,fval,exitflag,output,lambda,hessian]=Kruiniger_opt(N,T0,Dy,Dy1,theta0,lq,uq)
 
% maximum likelihood routine. see the program transformed_ml.m
 

[theta,fval,exitflag,output,lambda,~,hessian] = fmincon(@likelihood,theta0,[],[],[],[],lq,uq,[]); 
 
    function [likelihood]=likelihood(theta)
 
    dim_theta = size(theta0,1);  
    phi   = theta(dim_theta-2,1);  
    S = theta(dim_theta-1,1);  
    Sv = theta(dim_theta,1);   

   inOmegastar=eye(T0-1)-ones(T0-1)/(T0-1); % Q
    lik1 = (N*(T0-1)/2)*log(2*pi)+(N*(T0-2)/2)*log(S) +(N/2)*log(S+(T0-1)*Sv);
    for i=1:N
    Dyi = Dy(:,i);
    DWi = Dy1(:,i);
    Dui =  Dyi - DWi*phi;
     lik1 = lik1 +0.5* Dui'*inOmegastar* Dui/S+(2*(S+(T0-1)*Sv))^(-1)*(T0-1)^(-1)*(ones(1,T0-1)*Dui)^(2);
    end
 
likelihood = lik1;

end
end
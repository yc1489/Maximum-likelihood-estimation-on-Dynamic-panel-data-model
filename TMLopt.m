function [theta,fval,exitflag,output,lambda,hessian]=TMLopt(N,T,Dy,DW,phi,Omegastar,theta0,lq,uq)
 
% maximum likelihood routine. see the program transformed_ml.m
 

[theta,fval,exitflag,output,lambda,~,hessian] = fmincon(@likelihood,theta0,[],[],[],[],lq,uq,[]); 
 
    function [likelihood]=likelihood(theta)
 
    dim_theta = size(theta0,1);
    phi   = theta(1:dim_theta-2,1);
    omega = theta(dim_theta-1,1);
    sigma = theta(dim_theta,1);

    Omegastar(1,1) = omega;
    inOmegastar = pinv(Omegastar);
 
    lik1 = (N*T/2)*log(sigma) + (N/2)*log(1+T*(omega-1));
 
    for i=1:N
    Dyi = Dy(:,i);
    DWi = squeeze(DW(:,i,:));

    Dui =  Dyi - DWi*phi;
     lik1 = lik1 + 0.5* Dui'*inOmegastar* Dui/sigma;
    end
 
likelihood = lik1;

end
end

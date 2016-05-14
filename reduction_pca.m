function [X,params] = reduction_pca( X, opts )

    p = opts.p;   
    M = zeros(1,size(X,2));
    S = ones(1,size(X,2));
    
    if( strcmp(opts.zero_mean,'true') ) 
        M = mean(X,1);        
        X     = X   - (ones(size(X))*diag(M));
    end
    
    if( strcmp(opts.unit_variance,'true') )       
        S = std(X)+eps;
        X     = X  ./ (ones(size(X))*diag(S));
        
    end
   
    % Co-variance   
    sigma = cov(X);

    % Perform SVD
    [~, E, V] = svd(sigma);    

    %k = find( (cumsum(diag(E)) / trace(E)) <= p,1,'last');
    tra = trace(E);
    sm = 0;
    for k=1:length(E)
        sm = sm + E(k,k);
        if( sm/tra > p )
            break
        end        
    end

   
    % PCA matrix, making this the return "features as W"
    W = V(:,1:k);
    
    % Compute the projections    
    X      = X*W;     
    
    params.M = M;
    params.S = S;
    params.W = W;

end


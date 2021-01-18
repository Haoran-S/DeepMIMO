function [p_opt, sr_opt, totIters] = WMMSE_sum_rate(H, K)
var_noise = 1e-11;
p_int = rand(K,1);
Pmax = ones(K,1);
vnew = 0;
b = sqrt(p_int);
f = zeros(K, 1);
w = f;
for i=1:K
    f(i) = H(i,i)*b(i)/((H(i,:).^2)*(b.^2)+var_noise);
    w(i) = 1/(1-f(i)*b(i)*H(i,i));
    vnew = vnew + log2(w(i));
end

VV = [vnew];
iter = 0;

while(1)
    iter = iter+1;
    vold = vnew;
    for i=1:K
        btmp = w(i)*f(i)*H(i,i)/sum(w.*(f.^2).*(H(:,i).^2));
        b(i) = min(btmp, sqrt(Pmax(i))) + max(btmp, 0) - btmp;
    end
    
    vnew = 0;
    for i=1:K
        f(i) = H(i,i)*b(i)/((H(i,:).^2)*(b.^2)+var_noise);
        w(i) = 1/(1-f(i)*b(i)*H(i,i));
        vnew = vnew + log2(w(i));
    end
    
    VV = [VV vnew];
    if vnew-vold <= 1e-3 | iter>100
        break;
    end
end

totIters = iter;
p_opt = b.^2;
sr_opt = vnew;
end


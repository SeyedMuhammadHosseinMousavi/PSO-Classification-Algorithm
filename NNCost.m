function Cost = NNCost(XX,Xtr,Ytr,Network)
Cost = zeros(size(XX,1),1);
for ii = 1:size(XX,1)
    X = XX(ii,:);
    Network = NetConstruct(Network,X);
    YtrNet = sim(Network,Xtr')';
    C = mse(YtrNet - Ytr);
    Cost(ii,1) = C;    
end
end

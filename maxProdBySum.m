function [I] = maxProdBySum(X)
    Y = [X(1:2,:); ((X(3,:).*X(4,:))./((X(3,:)+X(4,:))))];
    [M, I] = max(Y(3,:), [], 2);
end


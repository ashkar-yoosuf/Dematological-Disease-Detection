function [X_Norm_Prediction] = featureNormalizePrediction(X, mu, sigma)

    X_Norm_Prediction = bsxfun(@minus, X, mu);

    X_Norm_Prediction = bsxfun(@rdivide, X_Norm_Prediction, sigma);

end


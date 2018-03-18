function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
sum1=0;
sum2=0;
for j=1:m
    sum1=sum1+(theta'*X(j,:)'-y(j));
    sum2=sum2+(theta'*X(j,:)'-y(j))*X(j,2);
end
theta(1)=theta(1)-sum1*alpha/m;
theta(2)=theta(2)-sum2*alpha/m;
   
J_history(iter) = computeCost(X, y, theta);

end

end

function [J grad] = nnCostFunction(nn_params, ...
    input_layer_size, ...
    hidden_layer_size, ...
    num_labels, ...
    X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
    hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
    num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
size(X)

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


a1 =[ones(m,1), X];
layer1 =  sigmoid(Theta1*a1');
a2 =  layer1';
secondinput = size(a2)
a22 = [ones(m,1),a2];
outlayer = sigmoid(Theta2*a22');
size(outlayer)
h = outlayer;

for c=1:num_labels 
    ynew(c,:) = (y ==c);
end

yn = size(ynew)
J = (1/m) *sum(sum((-ynew .* log(h) - (1 - ynew) .* log(1 - h))))

Reg1 = Theta1;
Reg1(:,1)=0;
Reg2 = Theta2;
Reg2(:,1)=0;

Reg = (lambda/(2*m)) * (sum(sum( Reg1.^2 )) + sum( sum( Reg2.^2 ) ));
J= J+Reg
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
% %               first time.
% 
% for k=1:m
%     a1 = [1; X(k,:)']; %aded 1 column of 1 row containing 1 1x401
%     z2 = sigmoid(Theta1*a1);
%     a2 = [1; z2];
%     z3 = (Theta2 * a2);
%     a3 = sigmoid(z3)
%     
%     a1size = size(a1)
%     z2size = size(z2)
%     a2size = size(a2)
%     Theta2size = size(Theta2)
%     z3size = size(z3)
%     a3size = size(a3)
%     
%     error3 = a3 - ynew(:,k);
%     error3size = size(error3)
%     
%     error2 = (Theta2' * error3) .* [1;sigmoidGradient(z2)];
%     
%     error2 = error2(2:end,1);
%     
%     Error2 = (Error2 + error3 * a2');
%     Error1 = (Error1 + error2 * a1');
% end
% 
Delta1_2 = zeros(size(Theta2));
Delta1_1 = zeros(size(Theta1));
% 
for t=1:m

    %Step 1
	a1 = [1 ; X(t,:)']; %aded 1 column of 1 row containing 1 1x401
	z2 = Theta1 *a1 ; % matrix 25x1
	a2 = sigmoid(z2);
	a2 = [1;a2];       % mat 26x1
	z3 = Theta2 *a2;   % mat 10x1
	a3 = sigmoid(z3);
    
    a1size = size(a1)
    z2size = size(z2)
    a2size = size(a2)
    Theta2size = size(Theta2)%10x26
    z3size = size(z3)
    a3size = size(a3)

    %step 2
	error3 = a3 - ynew(:,t); %error calculated between the hyp and the output 
    
    %step3
	error2 = (Theta2' * error3 ) .*[1;sigmoidGradient(z2)]; % back propagation (error in the 2nd layer"hidden layer" is calculated by the 
    %activation weigts* by the output layer while respecting the error caused from the prev node through the gradient of the segmoid function)
    
    %step 4
	error2 = error2(2:end,1); %step 4 he mentined to ignore the error of the bais node
	Delta1_2 = Delta1_2 +   error3 * a2';   
	Delta1_1 = Delta1_1 +   error2 * a1';
    
      %step 5 ureg gradient
%     Theta1_grad = Delta1_1 /m;
%     Theta2_grad = Delta1_2 /m;
end

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad = Delta1_1 /m + ( lambda / m ) * [zeros(size(Theta1,1),1) , Theta1(:,2:end)];
Theta2_grad = Delta1_2 /m + ( lambda / m ) * [zeros(size(Theta2,1),1) , Theta2(:,2:end)];  

grad = [Theta1_grad(:) ; Theta2_grad(:)];


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

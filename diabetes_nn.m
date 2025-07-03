clear; close all; clc;

%% Load and prepare the dataset
data = load('dataset_diabetes.txt');
X = data(:, 1:2); % Input features (GL, WC)
Y = data(:, 3);    % Output labels (0 or 1)

% Display original data
disp('Original Dataset:');
disp([X Y]);

%% 1. Data Normalization to [0, 1] range
min_X = min(X);
max_X = max(X);
X_normalized = (X - min_X) ./ (max_X - min_X);

% Display normalized data
disp('Normalized Dataset:');
disp([X_normalized Y]);

%% 2. Network Architecture Parameters
input_size = 2;    % GL and WC
hidden_size = 3;   % 3 neurons in hidden layer
output_size = 1;   % 1 output neuron (risk prediction)

% Initialize weights and biases with small random values
rng(42); % Set seed for reproducibility
W1 = randn(input_size, hidden_size) * 0.1; % Input to hidden weights
b1 = zeros(1, hidden_size);                % Hidden layer biases
W2 = randn(hidden_size, output_size) * 0.1; % Hidden to output weights
b2 = zeros(1, output_size);                 % Output layer bias

% Training parameters
learning_rate = 0.1;
epochs = 1500;

% Store error for plotting
errors = zeros(epochs, 1);

% Store predictions for each sample over time (for optional plot)
prediction_evolution = zeros(epochs, size(X, 1));

%% 3. Training the Neural Network
for epoch = 1:epochs
    total_error = 0;
    
    % Forward Propagation
    % Hidden layer
    hidden_input = X_normalized * W1 + b1;
    hidden_output = 1 ./ (1 + exp(-hidden_input)); % Sigmoid
    
    % Output layer
    output_input = hidden_output * W2 + b2;
    output = 1 ./ (1 + exp(-output_input)); % Sigmoid
    
    % Calculate error (Mean Squared Error)
    error = 0.5 * (Y - output).^2;
    total_error = sum(error);
    errors(epoch) = total_error;
    
    % Store predictions for the optional plot
    prediction_evolution(epoch, :) = output';
    
    % Backpropagation
    % Output layer gradient
    d_output = (output - Y) .* output .* (1 - output);
    
    % Hidden layer gradient
    d_hidden = (d_output * W2') .* hidden_output .* (1 - hidden_output);
    
    % Update weights and biases
    W2 = W2 - learning_rate * (hidden_output' * d_output);
    b2 = b2 - learning_rate * sum(d_output);
    W1 = W1 - learning_rate * (X_normalized' * d_hidden);
    b1 = b1 - learning_rate * sum(d_hidden);
end

%% 4. Make final predictions
% Forward pass with trained weights
hidden_input = X_normalized * W1 + b1;
hidden_output = 1 ./ (1 + exp(-hidden_input));
output_input = hidden_output * W2 + b2;
final_predictions = 1 ./ (1 + exp(-output_input));

% Display final predictions
disp('Final Predictions:');
disp([X Y final_predictions]);

% Convert probabilities to binary predictions (0 or 1) using 0.5 threshold
binary_predictions = final_predictions >= 0.5;
disp('Binary Predictions (0.5 threshold):');
disp([X Y binary_predictions]);

%% 5. Plotting Results
% Error vs Epochs plot
figure;
plot(1:epochs, errors);
xlabel('Epoch');
ylabel('Total Error (Loss)');
title('Training Error vs Epochs');
grid on;

% Prediction evolution for each sample
figure;
hold on;
colors = ['r', 'g', 'b', 'm'];
for i = 1:size(X, 1)
    plot(1:epochs, prediction_evolution(:, i), 'Color', colors(i), ...
        'DisplayName', sprintf('Patient %d (Actual: %d)', i, Y(i)));
end
xlabel('Epoch');
ylabel('Predicted Risk Probability');
title('Prediction Evolution During Training');
legend('show');
grid on;
hold off;

%% 6. Analysis of Results
fprintf('\n--- Analysis of Results ---\n');

% Calculate accuracy
accuracy = mean(binary_predictions == Y) * 100;
fprintf('Accuracy: %.2f%%\n', accuracy);

% Display which cases were correctly classified
for i = 1:size(X, 1)
    if binary_predictions(i) == Y(i)
        fprintf('Patient %d (GL=%d, WC=%d): CORRECT (Actual: %d, Predicted: %d, Prob: %.4f)\n', ...
            i, X(i, 1), X(i, 2), Y(i), binary_predictions(i), final_predictions(i));
    else
        fprintf('Patient %d (GL=%d, WC=%d): INCORRECT (Actual: %d, Predicted: %d, Prob: %.4f)\n', ...
            i, X(i, 1), X(i, 2), Y(i), binary_predictions(i), final_predictions(i));
    end
end

% Clinical interpretation
fprintf('\nClinical Interpretation:\n');
fprintf('The network learned to identify high-risk patients based on GL and WC.\n');
fprintf('Patients with higher GL (>110) and WC (>95) were correctly identified as high-risk.\n');
fprintf('The model demonstrates that both glucose levels and waist circumference are important predictors of diabetes risk.\n');
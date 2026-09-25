%% This code evaluates the test set.

% ** Important.  This script requires that:
% 1)'centroid_labels' be established in the workspace
% AND
% 2)'centroids' be established in the workspace
% AND
% 3)'test' be established in the workspace

load('classifierdata.mat');  
load('testdata.mat');


% IMPORTANT!!:
% You should save 1) and 2) in a file named 'classifierdata.mat' as part of
% your submission.

testset=csvread('mnist_test_200.csv');

predictions = zeros(200,1);
outliers = zeros(200,1);

% loop through the test set, figure out the predicted number
for i = 1:200

testing_vector=test(i,1:784);

% Extract the centroid that is closest to the test image
[prediction_index, vec_distance] = assign_vector_to_centroid(testing_vector,centroids);

predictions(i) = centroid_labels(prediction_index);

end

%% DESIGN AND IMPLEMENT A STRATEGY TO SET THE outliers VECTOR
% outliers(i) should be set to 1 if the i^th entry is an outlier
% otherwise, outliers(i) should be 0
for i= 1:200
    [~, vecDistance] = assign_vector_to_centroid(test(i,1:784), centroids);
    outliers(i) = vecDistance > mean(vecDistance) + 2*std(vecDistance);
end



num_test = size(testset, 1);
predictedlabels = zeros(num_test, 1);

for i = 1:num_test
    test_vec = testset(i, 1:784);

    % Find nearest centroid using your Part 1 function
    [assigned_centroid, ~] = assign_vector_to_centroid(test_vec, centroids);

    % Store the centroid assignment in column 785 (as with training data)
    testset(i, 785) = assigned_centroid;

    % The predicted label is that centroid's numerical meaning
    predictedlabels(i) = centroid_labels(assigned_centroid);
end


%% MAKE A STEM PLOT OF THE OUTLIER FLAG
figure;
stem(outliers, 'filled');
xlabel('Test instance');
ylabel('Outlier flag');
title('Detected Outliers');
ylim([-0.1, 1.1]);



figure;
hold on;
plot(1:num_test, correctlabels, 'o', 'MarkerEdgeColor', 'b', ...
    'DisplayName', 'Correct Labels');
plot(1:num_test, predictedlabels, 'x', 'MarkerEdgeColor', [0.85 0.33 0.1], ...
    'DisplayName', 'Predicted Labels');
xlabel('Test Set Index');
ylabel('Label');
title('Predictions');
legend('show');
hold off;




%% The following plots the correct and incorrect predictions
% Make sure you understand how this plot is constructed
figure;
plot(correctlabels,'o');
hold on;
plot(predictions,'x');
title('Predictions');

%% The following line provides the number of instances where and entry in correctlabel is
% equatl to the corresponding entry in prediction
% However, remember that some of these are outliers
sum(correctlabels==predictions)

function [index, vec_distance] = assign_vector_to_centroid(data,centroids)
distances = vecnorm(centroids - data, 2, 2);
[vec_distance, index] = min(distances);

end

%% Compute accuracy
num_correct = sum(predictedlabels == correctlabels);
accuracy = num_correct / num_test;

fprintf('Accuracy: %d/%d = %.2f%%\n', num_correct, num_test, accuracy * 100);

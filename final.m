% Load and Preprocess Dataset
imageDir = 'C:\Users\SANJNA\OneDrive\Desktop\riceleaf';  % Replace this with your dataset path
categories = {'leaf_smut', 'brown_spot', 'bacterial_leaf_blight'};
imds = imageDatastore(fullfile(imageDir, categories), ...
                      'LabelSource', 'foldernames', ...
                      'IncludeSubfolders', true);

% Verify the dataset path
if ~isfolder(imageDir)
    error('The specified image directory does not exist: %s', imageDir);
end

% Check number of images and labels
disp(['Number of images: ', num2str(numel(imds.Files))]);
disp('Labels:');
disp(unique(imds.Labels));

% Initialize feature matrix and labels
features = [];
labels = [];

% Loop through each image in the dataset with a counter
labelIdx = 1;  % Counter to keep track of labels
while hasdata(imds)
    % Read image
    img = read(imds);
    
    % Access the label for the current image
    imgLabel = imds.Labels(labelIdx);
    labelIdx = labelIdx + 1;  % Increment the label counter

    % Display current image being processed
    disp(['Processing image: ', imds.Files{labelIdx - 1}]);

    % Step 1: Contrast Adjustment (Image Preprocessing)
    grayImg = rgb2gray(img);  % Convert to grayscale
    adjustedImg = imadjust(grayImg);  % Adjust contrast
    
    % Step 2: K-means Segmentation
    % Reshape image to a 2D array of pixels
    pixelValues = double(adjustedImg(:));
    % K-means clustering with 3 clusters
    [clusterIndices, ~] = kmeans(pixelValues, 3, 'MaxIter', 100, 'Replicates', 3);  % Increase replicates for stability
    segmentedImg = reshape(clusterIndices, size(adjustedImg));  % Reshape to the original image size
    
    % Step 3: Feature Extraction using GLCM (Gray-Level Co-occurrence Matrix)
    % Extract multiple GLCMs with different offsets to capture richer texture features
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcm = graycomatrix(segmentedImg, 'Offset', offsets);
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    
    % Concatenate GLCM features across offsets
    featureVector = [mean(stats.Contrast), mean(stats.Correlation), mean(stats.Energy), mean(stats.Homogeneity)];
    
    % Store features and labels
    features = [features; featureVector];
    labels = [labels; imgLabel];
end

% Check features
disp(['Number of feature vectors: ', num2str(size(features, 1))]);

% Convert labels to categorical
labels = categorical(labels);

% Split Data into Training and Test Sets
numObservations = numel(labels);
cv = cvpartition(numObservations, 'Holdout', 0.2);
trainInd = training(cv);
testInd = test(cv);

trainFeatures = features(trainInd, :);
trainLabels = labels(trainInd);
testFeatures = features(testInd, :);
testLabels = labels(testInd);

% Step 4: Train KNN Classifier with Optimized Parameters
Mdl = fitcknn(trainFeatures, trainLabels, 'NumNeighbors', 5, 'Distance', 'euclidean', 'Standardize', true);

% Evaluate the Model
predictions = predict(Mdl, testFeatures);
accuracy = sum(predictions == testLabels) / numel(testLabels);
disp(['Test Accuracy: ', num2str(accuracy * 100), '%']);

% Prediction Function for a New Image
function diseaseType = predictDisease(img, Mdl)
    % Step 1: Contrast Adjustment
    grayImg = rgb2gray(img);
    adjustedImg = imadjust(grayImg);

    % Step 2: K-means Segmentation
    pixelValues = double(adjustedImg(:));
    [clusterIndices, ~] = kmeans(pixelValues, 3, 'MaxIter', 100, 'Replicates', 3);
    segmentedImg = reshape(clusterIndices, size(adjustedImg));

    % Step 3: Feature Extraction (GLCM with Multiple Offsets)
    offsets = [0 1; -1 1; -1 0; -1 -1];
    glcm = graycomatrix(segmentedImg, 'Offset', offsets);
    stats = graycoprops(glcm, {'Contrast', 'Correlation', 'Energy', 'Homogeneity'});
    featureVector = [mean(stats.Contrast), mean(stats.Correlation), mean(stats.Energy), mean(stats.Homogeneity)];

    % Step 4: Classify using Trained KNN Model
    diseaseType = predict(Mdl, featureVector);
end

% Example: Predict Disease Type for a New Image
newImagePath = 'C:\Users\SANJNA\OneDrive\Desktop\riceleaf\bacterial_leaf_blight\DSC_0365.JPG';  % Replace with your test image path
if ~isfile(newImagePath)
    error('The specified test image does not exist: %s', newImagePath);
end

newImage = imread(newImagePath);
predictedDisease = predictDisease(newImage, Mdl);
disp(['Predicted Disease Type: ', char(predictedDisease)]);

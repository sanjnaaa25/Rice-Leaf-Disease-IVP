% Load Data
imageDir = 'C:\Users\SANJNA\OneDrive\Desktop\riceleaf';  % Replace with your dataset path
categories = {'leaf_smut', 'brown_spot', 'bacterial_leaf_blight'};
imds = imageDatastore(fullfile(imageDir, categories), ...
                      'LabelSource', 'foldernames', ...
                      'IncludeSubfolders', true);

% Split data into training and validation sets
[trainSet, valSet] = splitEachLabel(imds, 0.8, 'randomized');

% Image Size Required by ResNet-18
inputSize = [224 224 3];

% Set up Augmentation
augmenter = imageDataAugmenter('RandRotation', [-20, 20], ...
                               'RandXTranslation', [-5, 5], ...
                               'RandYTranslation', [-5, 5]);

augmentedTrainSet = augmentedImageDatastore(inputSize, trainSet, 'DataAugmentation', augmenter);
augmentedValSet = augmentedImageDatastore(inputSize, valSet);

% Load Pre-trained ResNet-18 Network
net = resnet18;

% Modify the Network for Transfer Learning
lgraph = layerGraph(net);

% Get the number of classes (disease types)
numClasses = numel(categories);

% Replace the last fully connected layer and classification layer
newFcLayer = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
                                 'WeightLearnRateFactor', 10, ...
                                 'BiasLearnRateFactor', 10);
newClassLayer = classificationLayer('Name', 'new_classoutput');

% Replace the layers in the network
lgraph = replaceLayer(lgraph, 'fc1000', newFcLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);

% Set Training Options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 16, ...
    'MaxEpochs', 15, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedValSet, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the Network
trainedNet = trainNetwork(augmentedTrainSet, lgraph, options);

% Evaluate the Network on Validation Set
predictions = classify(trainedNet, augmentedValSet);
valAccuracy = sum(predictions == valSet.Labels) / numel(valSet.Labels);
disp(['Validation Accuracy: ', num2str(valAccuracy * 100), '%']);

% Function to Predict Disease Type for a New Image
function diseaseType = predictDisease(imgPath, trainedNet)
    % Read and Resize the Image
    img = imread(imgPath);
    img = imresize(img, [224 224]);  % Resize to match input size of ResNet-18

    % Classify the Image Using the Trained Network
    diseaseType = classify(trainedNet, img);
end

% Example: Predict Disease Type for a New Image
newImagePath = 'C:\Users\SANJNA\OneDrive\Desktop\riceleaf\bacterial_leaf_blight\DSC_0365.JPG';  % Replace with the path to your test image
if ~isfile(newImagePath)
    error('The specified test image does not exist: %s', newImagePath);
end

predictedDisease = predictDisease(newImagePath, trainedNet);
disp(['Predicted Disease Type: ', char(predictedDisease)]);

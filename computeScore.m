% This code is to compute the final score of a saliency map.The goal is to
% have smaller region of the saliency map and smaller classification score.
% we normalize the area of affected pixels by the total number of pixels
% and multiply it by the classification score (hopefully normalized one)

function [sharp_score, smooth_score, sp_score] = computeScore(directory)
[sharpAcc, smoothAcc, spAcc, sharpValAcc, smoothValAcc, spValAcc] = deal([]);
%     imgName = 'flute1';
%     imgNameComplete = [imgName,'.jpg'];
%directory = 'results/alexnet_original';
csv_directory = fullfile(directory,'results.csv');
imgList = dir(directory);
imgList = imgList([imgList.isdir]);
imgList(1:2) = [];
dataTable = readtable(csv_directory);
imgNameList = dataTable{:,1};

sharpAcc = zeros(length(imgList), 1);
smoothAcc = zeros(length(imgList), 1);
spAcc = zeros(length(imgList), 1);

sharpValAcc = zeros(length(imgList), 1);
smoothValAcc = zeros(length(imgList), 1);
spValAcc = zeros(length(imgList), 1);

for j = 1:length(imgList)
    
    disp(['Compute ' num2str(j)]);
    
    imgName = imgList(j).name;
    imgNameComplete = [imgName,'.JPEG'];

    for i = 1:length(imgNameList)
        imgNameNow = imgNameList{i};
        imgNameCompleteLen = length(imgNameComplete);
        if ~strcmp (imgNameNow(end-imgNameCompleteLen+1:end),imgNameComplete)
            continue
        else
            n = i;
            break
        end
    end

    sharpPath = fullfile (directory,imgName,'sharp','mask.png');
    smoothPath = fullfile (directory,imgName,'smooth','mask.png');
    spPath = fullfile (directory,imgName,'superpixel','mask.png');

    sharpMask = imread(sharpPath);
    smoothMask = imread(smoothPath);
    spMask = imread(spPath);
    spMask = spMask(:,:,1);

    sharpMask (sharpMask <= 25) = 0;
    smoothMask (smoothMask <= 25) = 0;
    spMask (spMask <= 25) = 0;
    pixarea = 224*224;
    pixvalarea = 255*255;
    
    sharpProbArea = (numel(find(sharpMask))/pixarea);
    smoothProbArea = (numel(find(smoothMask))/pixarea);
    spProbArea = (numel(find(spMask))/pixarea);
    
    sharpProbValArea = (sum(sum(sharpMask))/pixvalarea);
    smoothProbValArea = (sum(sum(smoothMask))/pixvalarea);
    spProbValArea = (sum(sum(spMask))/pixvalarea);

    sharpCScore = dataTable{n,7};
    smoothCscore = dataTable{n,3};
    spCscore = dataTable{n,11};

    sharp_score = sharpCScore * sharpProbArea;
    smooth_score = smoothCscore * smoothProbArea;
    sp_score = spCscore * spProbArea;
    
    sharpval_score = sharpCScore * sharpProbValArea;
    smoothval_score = smoothCscore * smoothProbValArea;
    spval_score = spCscore * spProbValArea;
    
    sharpAcc(j) = sharp_score;
    smoothAcc(j) = smooth_score;
    spAcc(j) = sp_score;
    
    sharpValAcc(j) = sharpval_score;
    smoothValAcc(j) = smoothval_score;
    spValAcc(j) = spval_score;
end

figure
plot(sharpAcc)
hold on
plot(smoothAcc,'r')
hold on
plot(spAcc,'k')


figure
plot(sharpValAcc)
hold on
plot(smoothValAcc,'r')
hold on
plot(spValAcc,'k')

end

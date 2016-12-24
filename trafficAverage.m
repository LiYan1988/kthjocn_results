function [trafficAverage] = trafficAverage(trafficMatrixPath)

matrices = dir(trafficMatrixPath);
for i = 1:length(matrices)
    if ~matrices(i).isdir
        tmp = strcat(matrices(i).folder, '\', matrices(i).name);
        matrix = importfileTrafficMatrix(tmp);
        numFlows = sum(sum(matrix~=0));
        sumThroughputs = sum(matrix(:));
    end
end
trafficAverage = sumThroughputs/numFlows;
function outputPath = joinPath(pathList)
% join paths in pathList into one path, input is a cell
outputPath = pathList(1);
if length(pathList) >=2
    for i = 2:length(pathList)
        outputPath = strcat(outputPath{1}, '\', pathList{i});
    end
end
outputPath = outputPath{1};
end

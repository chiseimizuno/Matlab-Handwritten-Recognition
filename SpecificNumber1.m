function specific = SpecificNumber(index)

%1515 - example of a retarded bitch ass hoe
%Program shows what the average of the testing data looks
%clc;clear;

%data = NoiseData;

data = importdata('semeion.data');



graph = zeros(16);
specific = zeros(16);

for j = 1:16
    graph(j,:) = data(index,(16*j)-15:(16*j));
end

specific = graph;
imagesc(specific);




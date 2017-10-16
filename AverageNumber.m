function average = AverageNumber(numPattern)

%Program shows what the average of the testing data looks
%clc;clear;
data = importdata('semeion.data');
%data = inverseData();

graph = zeros(16);
average = zeros(16,16,10);
counter = zeros(10,1);
for z = 1:10
for i = 1:numPattern
    if (sum(data(i,256+z) ==  1))
        counter(z) = counter(z) + 1;
        for j = 1:16
            graph(j,:) = data(i,(16*j)-15:(16*j));
        end
        average(:,:,z) = average(:,:,z) + graph;
    end
end
end
average = average/20;
%average1 = round(average1);

% figure('Name','Average of Each Number');
% row = 2;
% column = 5;
% for i = 1:10
% subplot(row,column,i);imagesc(average(:,:,i)); title('0');
% end




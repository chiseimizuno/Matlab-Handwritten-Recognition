%_______________________Initial Initialization_____________________________
%Program by Chisei Mizuno + Jennifer Williams
 
clc; %Clears Command Window
clear; %Clears Workspace

%_______________________Creating Matrix for Each Images____________________
time1 = clock;

data = importdata('semeion.data');
%data = inverseData();

numPattern = 200;

desired_output = zeros(10,numPattern);
pattern = zeros(256,numPattern);

for i = 1:numPattern
    pattern(:,i) = data(i,1:256);
    desired_output(:,i) = data(i,257:266);
end


%%_______________________Intilization Variables____________________________

%Pattern & Desired Output created above
lrate = 1; 
inputUnit = 256;
hiddenUnit = 100;
outputUnit = 10;
%numPattern = 200; 
Epoch = 1500;
converge = 0.01; %When to converge


%Random network weights
w_fg = rand(hiddenUnit,inputUnit)-0.5;
w_gh = rand(outputUnit,hiddenUnit)-0.5;

output_activation = zeros(size(desired_output,1),numPattern);
errors = zeros(size(desired_output,1),numPattern);
sseMatrix = zeros(0);
SSE = 0;


for z = 1:Epoch %Epochs

    for i = 1:numPattern %Patterns
        
        %Calculating g and h values
        input_to_hidden = w_fg * pattern(:,i);
        hidden_activation = activation_fn(input_to_hidden);

        hidden_to_output = w_gh * hidden_activation;
        output_activation(:,i) = activation_fn(hidden_to_output);
        
        %Calculating error
        error = desired_output(:,i) - output_activation(:,i);
        errors(:,i) = error;
        
        
        %Calculating changes in weights
        dw_gh = lrate.*diag(activation_fn_derived(w_gh*hidden_activation))*(error)...
            *hidden_activation';

        dw_fg = lrate.*diag(activation_fn_derived(w_fg*pattern(:,i)))*w_gh'...
            *diag(error)*(activation_fn_derived(w_gh*hidden_activation))*pattern(:,i)';
        
        %Updating Weights
        w_fg = w_fg + dw_fg;
        w_gh = w_gh + dw_gh;

    end
        
    desired_number = zeros(numPattern,1);
    %Calculating Output Number
    for i = 1:numPattern
        for j = 1:10
            if (desired_output(j,i) == 1)
                desired_number(i) = j-1;
            end
        end
    end
    
    %Calculating yes or no
    new_output = zeros(numPattern,1);
    for i = 1:numPattern
        [maxNum, MaxIndex] = max(output_activation(:,i));
        new_output(i) = MaxIndex-1;
    end
    
    total = ((desired_number == new_output));
    total = sum(total);
        
    %Calculating SSE
    SSE = trace(errors'*errors);
    sseMatrix = [sseMatrix; SSE];
    
    %Break if SSE less than 0.01
    if (SSE < converge)
       break
    end
    
    %End Loop Conditionals
    if (z == Epoch)
       clc;clear;
       error('Warning did not converge. Please Run Experiment Again');
    end

end

%Figure plotting SSE/Error
figure('Name','SSE Matrix');
plot(sseMatrix);
xlabel('# of Epochs');
ylabel('SSE / Error');
%Graphing Image
figure('Name','Desired vs Actual Output');
subplot(1,2,1); imagesc(desired_output); title('Desired Output');
subplot(1,2,2); imagesc(output_activation); title('Actual Output');


%%_______________________Testing Novel Patterns____________________________

%data = inverseData;

desired_output1 = zeros(10,1);
pattern1 = zeros(256,1);

numTestPattern = 20;
p = round(rand(numTestPattern,1)*(1593-numPattern)+numPattern);


for i = 1:numTestPattern
    pattern1(:,i) = data(p(i),1:256);
    desired_output1(:,i) = data(p(i),257:266);
end

output_activation1 = zeros(size(desired_output1,1),numTestPattern);
errors1 = zeros(size(desired_output1,1),numTestPattern);


for i = 1:numTestPattern
        %Calculating g and h values
        input_to_hidden = w_fg * pattern1(:,i);
        hidden_activation = activation_fn(input_to_hidden);

        hidden_to_output = w_gh * hidden_activation;
        output_activation1(:,i) = activation_fn(hidden_to_output);
        
        error = desired_output(:,i) - output_activation(:,i);
        errors1(:,i) = error;
end

errors1 = sum(errors1,1)/10;
errors1 = abs(round(errors1*100000)/10);

%____________________GRAPHS________________________
figure('Name','NEW DATA: Desired vs Actual Output');
subplot(1,2,1); imagesc(desired_output1'); title('Desired Output');
subplot(1,2,2); imagesc(output_activation1'); title('Actual Output');



averages = AverageNumber(numPattern);

figure('Name','Average of Each Number');
row = 2;
column = numTestPattern;

desired_number1 = zeros(numTestPattern,1);
    %Calculating Output Number
    for i = 1:numTestPattern
        for j = 1:10
            if (desired_output1(j,i) == 1)
                desired_number1(i) = j-1;
            end
        end
    end
    
    %Calculating yes or no
    new_output1 = zeros(numTestPattern,1);
    for i = 1:numTestPattern
        [maxNum, MaxIndex] = max(output_activation1(:,i));
        new_output1(i) = MaxIndex-1;
    end
    
    total1 = ((desired_number1 == new_output1));
    total1 = sum(total1);
        


for i = 1:numTestPattern
    subplot(row,column,i);imagesc(averages(:,:,(desired_number1(i)+1))); title(desired_number1(i));
end
for i = 1:numTestPattern
    subplot(row,column,i+numTestPattern);imagesc(SpecificNumber1(p(i))); title(new_output1(i));
end

%________________________Print___________________________

numSSE = size(sseMatrix,1);
disp('Number of Epochs Until Convergence < 0.1: ');disp(numSSE)  
disp('Number of hidden units: '); disp(hiddenUnit);
disp('Number of trained pattern: '); disp(numPattern);
disp('Number of tested pattern: '); disp(numTestPattern);

time2 = clock;

time2-time1

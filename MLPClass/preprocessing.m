clearvars
dat = readtable('dat.csv','NumHeaderLines',4);  % skips the first three rows of data
dat = dat(1:2000,:); %trims the data to a more reasonable size
windowSize = 5; %defines the size of the window that is captured in each part of the table
dat = normalize(dat);
plot(dat.Var1, dat.Var2);
datArray = {};
width = length(dat.Var2)-windowSize;
for i=1:(length(dat.Var2)-windowSize)
    holder = dat(i:(i+windowSize),"Var2");
    datArray = [datArray; holder];
end
datArray = datArray{:,:};
datArray = reshape(datArray, windowSize+1,width);
datArray = transpose(datArray);
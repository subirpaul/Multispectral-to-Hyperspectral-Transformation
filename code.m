% Reading the Hyperspectral Data
[Hyp,R] = geotiffread('J:\IISc PhD\New Work\Hyperion\geo_ref_178_2014_144052.tif');
info1 = geotiffinfo('J:\IISc PhD\New Work\Hyperion\geo_ref_178_2014_144052.tif');
height1 = info1.Height; % Integer indicating the height of the image in pixels
width1 = info1.Width; % Integer indicating the width of the image in pixels
[rows1,cols1] = meshgrid(1:width1,1:height1);
[x1,y1] = pix2map(info1.RefMatrix, rows1, cols1);
[lat1,lon1] = projinv(info1,x1,y1); % Creating Latitude and Longitude Matrix

% Reading the Multispectral Data
[L7,R_] = geotiffread('J:\IISc PhD\New Work\LE071440522014062701T1-SC20171102080430\L7_Hyp_shp.tif');
info2 = geotiffinfo('J:\IISc PhD\New Work\LE071440522014062701T1-SC20171102080430\L7_Hyp_shp.tif');
height2 = info2.Height; % Integer indicating the height of the image in pixels
width2 = info2.Width; % Integer indicating the width of the image in pixels
[rows2,cols2] = meshgrid(1:width2,1:height2);
[x2,y2] = pix2map(info2.RefMatrix, rows2, cols2);
[lat2,lon2] = projinv(info2,x2,y2); % Creating Latitude and Longitude Matrix

% Transforming the data into Reflectance values
Ref_Hyp = double(Hyp);
Ref_Hyp = Ref_Hyp/10000; % 10000 is the scale factor
% Reflectances less than zero and more than one are identified as 'NaN'
f1 = find(Ref_Hyp<=0 | Ref_Hyp>1);
Ref_Hyp(f1) = NaN;
[x,y,z1] = size(Ref_Hyp); 
Ref_Hyp = reshape(Ref_Hyp,x*y,z1); % reshaping the 3D data into 2D, where Lat, Lon are vectorized

Ref_L7 = double(L7);
Ref_L7 = Ref_L7/10000;
f2 = find(Ref_L7<=0 | Ref_L7>1);
Ref_L7(f2) = NaN;
[x,y,z2] = size(Ref_L7); 
Ref_L7 = reshape(Ref_L7,x*y,z2);

% removing the locations or pixels having 'NaN' values
R1 = find(~any(isnan(Ref_Hyp),2));   % finding the rows which does not have NaN in all the columns
Ref_Hyp = Ref_Hyp(R1,:);
Ref_L7 = Ref_L7(R1,:);
R2 = find(~any(isnan(Ref_L7),2));   % finding the rows which does not have NaN in all the columns
Ref_Hyp = Ref_Hyp(R2,:);
Ref_L7 = Ref_L7(R2,:);

% Creation of Training (50% samples) and Testing (50% samples) data
[trainInd,valInd,testInd] = dividerand(size(Ref_L7,1),0.5,0.25,0.25);
val_test=[valInd testInd];
l7_train=Ref_L7;
l7_train(val_test,:)=[];
l7_test=Ref_L7;
l7_test(trainInd,:)=[];
hyp_train=Ref_Hyp;
hyp_train(val_test,:)=[];
hyp_test=Ref_Hyp;
hyp_test(trainInd,:)=[];

% Stepwise Linear Regression Model Fitting
% Four statistical measures (Correlation coefficient, Root mean square error, 
% Structural similarity measure, Peak Signal-to-Noise Ratio) are evaluated for comparison
Mdl1=cell(170,1);
r1=NaN(170,1);
p1=NaN(170,1);
RMSE1=NaN(170,1);
ssimval1=NaN(170,1);
peaksnr1=NaN(170,1);
for i=1:170
    Mdl1{i,1} = stepwiselm(l7_train,hyp_train(:,i),'Upper','linear');  % stepwise linear regression
    y=predict(Mdl1{i,1},l7_test);
    [r1(i),p1(i)]=corr(y,hyp_test(:,i)); % Correlation coefficient
    RMSE1(i)=rmse(hyp_test(:,i),y); % Root mean square error
    ssimval1(i) = ssim(y,hyp_test(:,i)); % Structural similarity measure
    peaksnr1(i) = psnr(y,hyp_test(:,i),max(hyp_test(:,i))); % Peak Signal-to-Noise Ratio
end
save('SLR_6Var.mat','Mdl1','r1','p1','RMSE1','ssimval1','peaksnr1','-v7.3');

% Support Vector Regression Model Fitting
Mdl2=cell(170,1);
r2=NaN(170,1);
p2=NaN(170,1);
elt2=NaN(170,1);
for i=1:170
    tic
    Mdl2{i,1} = fitrsvm(l7_train,hyp_train(:,i),'Standardize',true);  %support vector machine regression (linear)
    y=predict(Mdl2{i,1},l7_test);
    [r2(i),p2(i)]=corr(y,hyp_test(:,i));
    toc
    elt2(i)=toc;
end

Mdl2=cell(170,1);
r2=NaN(170,1);
p2=NaN(170,1);
RMSE2=NaN(170,1);
ssimval2=NaN(170,1);
peaksnr2=NaN(170,1);
elt2=NaN(170,1);
for i=121:170
    tic
    Mdl2{i,1} = fitrsvm(l7_train,hyp_train(:,i),'Standardize',true,'KernelFunction','gaussian');%...
        %,'OptimizeHyperparameters','auto'); %support vector machine regression (RBF)
    y=predict(Mdl2{i,1},l7_test);
    [r2(i),p2(i)]=corr(y,hyp_test(:,i));
    RMSE2(i)=rmse(hyp_test(:,i),y);
    ssimval2(i) = ssim(y,hyp_test(:,i));
    peaksnr2(i) = psnr(y,hyp_test(:,i),max(hyp_test(:,i)));
    toc
    elt2(i)=toc;
end
save('RBF_SVR_6Var_1.mat','Mdl2','r2','p2','RMSE2','ssimval2','peaksnr2','elt2','-v7.3');

r=NaN(170,6);
p=NaN(170,6);
for i=1:6
    [r(:,i),p(:,i)]=corr(Ref_L7(:,i),Ref_Hyp);
end

% Generation of Hyperspectral Data from new Multispectral data using the
% developed model
[L7_F,R] = geotiffread('J:\IISc PhD\New Work\LE071440522014062701T1-SC20171102080430\stacked_fa7.tif');
Ref_L7 = double(L7_F);
Ref_L7 = Ref_L7/10000;
f2 = find(Ref_L7<=0 | Ref_L7>1);
Ref_L7(f2) = NaN;
[x,y,z2] = size(Ref_L7); 
Ref_L7 = reshape(Ref_L7,x*y,z2);
R2 = find(~any(isnan(Ref_L7),2));   % finding the rows which does not have NaN in all the columns
Ref_L7 = Ref_L7(R2,:);
% hyp_pred = NaN((size(R2,1)-1)/2,170);
tic
for i=1:10
    hyp_pred(:,i) = predict(Mdl3_1{i,1}, Ref_L7);
end
toc
hp_10_1 = NaN(x*y,10);
for i=1:10
    hp_10_1(R2,i) = hyp_pred(:,i);
end
hp_10_1=reshape(hp_10_1,x,y,10);
save('Hyp_pred_l7_1.mat','hp_10_1','-v7.3');

% Evaluation of the generated CNNR-based Hyperspectral data, which was
% generated in python
hyp_pred = double(hyp_pred);
r3 = NaN(170,1);
p3 = NaN(170,1);
RMSE3 = NaN(170,1);
ssimval3 = NaN(170,1);
peaksnr3 = NaN(170,1);
for i = 1:170
    [r3(i),p3(i)] = corr(hyp_test(:,i),hyp_pred(:,i));
    RMSE3(i) = rmse(hyp_test(:,i),hyp_pred(:,i));
    ssimval3(i) = ssim(hyp_pred(:,i),hyp_test(:,i));
    peaksnr3(i) = psnr(hyp_pred(:,i),hyp_test(:,i),max(hyp_test(:,i)));
end

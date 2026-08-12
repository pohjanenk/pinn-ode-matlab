%PINN for solving pendulum equation of the form:s
% d2_theta/dt2 + 0.5*d_theta/dt + 9.80665*theta  (sin(theta) but small angle)
% IC : theta(0) = pi/4 , theta'(0) = 0

%this neural network is informed by a physics and data derived loss
%function
%% Generate input data and define network
t = linspace(0,4,200)';

tspan = linspace(0,4,20);
[tData, thetaExact] = ode45(@(t,theta) [theta(2); ...
    -0.5*theta(2) - 9.80665*theta(1)], ...
    linspace(0,4,20), [pi/4;0]);
thetaData = thetaExact(:,1); 

inputSize = 1;
layers = [
    featureInputLayer(1,Normalization="none")
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(1)
];

net = dlnetwork(layers);

%% Define loss function

function [loss,gradients] = modelLoss(...
    net,t,t0,tData,thetaData,w1,w2,wData)
%retrieve NN output
theta = forward(net,t);
d_theta = dlgradient(sum(theta,"all"),t,EnableHigherDerivatives=true);
d2_theta = dlgradient(sum(d_theta,"all"),t,EnableHigherDerivatives=true);

eq = d2_theta + 0.5*d_theta + 9.80665*theta;

theta0 = forward(net,t0); 
d_theta0 = dlgradient(sum(theta0,"all"),t0,EnableHigherDerivatives=true);

ic1 = theta0 - pi/4; 
ic2 = d_theta0;

physicsLoss = mean(eq.^2,"all") + w1*ic1.^2+ w2*ic2.^2;

%data loss
thetaPredData = forward(net,tData);
dataLoss = mean((thetaPredData - thetaData).^2,"all");

%Total loss
loss = physicsLoss+ wData*dataLoss;

gradients = dlgradient(loss, net.Learnables);
end
%% Specify training options
numEpochs = 10000;

initialLearnRate = 0.01; 

w1 = 100;
w2 = 100;
wData = 150;
targetLogLoss = -15;
targetReached = false;

%% Train Model
tTrain = dlarray(t',"CB");
tData = dlarray(tData',"CB");
thetaData = dlarray(thetaData',"CB");

averageGrad = [];
averageSqGrad = [];

%initialises the UI with relevant learning metrics
monitor = trainingProgressMonitor( ...
    Metrics="LogLoss", ...
    Info=["Epoch" "LearnRate"], ...
    XLabel="Iteration");
epoch = 0;
iteration = 0;
learnRate = initialLearnRate;
start = tic;

accFcn = dlaccelerate(@modelLoss);

while epoch < numEpochs  && ~monitor.Stop
    epoch = epoch + 1; 
    iteration = iteration + 1;

    % Evaluate the model gradients and loss using dlfeval and the modelLoss function
    t0 = dlarray(0,"CB");  
    [loss,gradients] = dlfeval(accFcn,...
          net,tTrain,t0,tData,thetaData,w1,w2,wData);

    [net,averageGrad,averageSqGrad] = adamupdate(...
        net,gradients,averageGrad,averageSqGrad,...
        iteration,learnRate);

    recordMetrics(monitor,iteration,LogLoss=log(loss));
    updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
    monitor.Progress = 100 * iteration/numEpochs;


    if mod(epoch, 2000) == 0
        learnRate = learnRate * 0.5;
    end
 
    if gather(extractdata(log(loss))) <= targetLogLoss
        disp("Target log loss reached at epoch "+ epoch)
        targetReached = true;
        break
    end
end


%% Test model
tTest = linspace(0,4,100)';

tTestDL = dlarray(tTest',"CB");
thetaModel = extractdata(forward(net,tTestDL))';

thetaAnalytic = exp(-0.25*tTest).*((pi/4)*cos(3.121562109*tTest) ...
    + (pi/16)*(1/3.121562109)*sin(3.121562109*tTest));

figure
plot(tTest,thetaAnalytic, "-")
hold on
plot(tTest,thetaModel, "--")
legend("Analytic","Model")
xlabel("t")
ylabel("\theta")
%% Evaluate error

rmse = sqrt(mean((thetaModel-thetaAnalytic).^2))

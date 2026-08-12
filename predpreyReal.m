%PINN model with real world data and lotka volterra differential equations, to solve inverse 
% problem for pred prey system
%reliable estimates : https://mc-stan.org/learn-stan/case-studies/lotka-volterra-predator-prey.html#exercises-and-extensions
%alpha = 0.55;
%beta = 0.028;
%delta = 0.026;
%gamma = 0.84;
%% Constants,data and network architecture 
alpha = 0.55;
delta = 0.026;

%read data 
data = readtable("hudson-bay-lynx-hare.csv");
tData = data.Year;
xData = data.Hare;
yData = data.Lynx;

% initial conditions
x0 = xData(1);
y0 = yData(1);
t0 = min(tData);

%Normalise time
T  = max(tData) - t0;
tauData = 2*(tData - t0)/T - 1;

%Collocation points
t = linspace(t0,t0+T,50)';
tau = 2*(t-t0)/T - 1;

%Scaling factors
xScale = max(xData);
yScale = max(yData);

inputSize = 1;
layers = [
    featureInputLayer(1,Normalization="none")
    fullyConnectedLayer(64)
    swishLayer
    fullyConnectedLayer(64)
    swishLayer
    fullyConnectedLayer(64)
    swishLayer
    fullyConnectedLayer(2)
    ];
net = dlnetwork(layers);
%% Define loss function
function [loss,physicsLoss,dataLoss,gradientB,...
    gradientG,gradientsNet] = modelLoss(...
    net,tau,tauData,xData,yData,wData,T,x0,y0,...
    alpha,beta,delta,gamma,xScale,yScale)
%Retrieve network output
N = forward(net,tau);
Nx = N(1,:);
Ny = N(2,:);

%Hard constraint enforcing initial conditions
x = x0 + (tau+1).*Nx;
y = y0 + (tau+1).*Ny;

%calculate derivatives 
dxTau = dlgradient(sum(x,"all"),tau,...
    EnableHigherDerivatives=true);
dyTau = dlgradient(sum(y,"all"),tau,...
    EnableHigherDerivatives=true);

dtau_dt = 2/T;
dxdt = dtau_dt*dxTau;
dydt = dtau_dt*dyTau;

%Lotka-Volterra equations
eq1 = dxdt - (alpha*x - beta*x.*y);
eq2 = dydt - (delta*x.*y - gamma*y);

%Scale outputs
eq1 = eq1/xScale;
eq2 = eq2/yScale;

physicsLoss = mean(eq1.^2,"all") + mean(eq2.^2,"all");

%Data loss
Ndata = forward(net,tauData);

NxData = Ndata(1,:);
NyData = Ndata(2,:);

xPred = x0 + (tauData+1).*NxData;
yPred = y0 + (tauData+1).*NyData;

%Scale outputs
xPredNorm = xPred./xScale;
yPredNorm = yPred./yScale;

xDataNorm = xData./xScale;
yDataNorm = yData./yScale;


dataLoss = mean((xPredNorm-xDataNorm).^2,"all") + ...
           mean((yPredNorm-yDataNorm).^2,"all");

%Overall loss
loss = physicsLoss + wData*dataLoss;

[gradientsNet,gradientB,gradientG] = ...
    dlgradient(loss,net.Learnables,beta,gamma);
end
%% Specify training options
numEpochs = 30000;
initialLearnRate = 0.005; 
targetLogLoss = -10;
targetReached = false;
wData = 5;
betaGuess = 0.02;
gammaGuess = 0.7;

%% Train Model
%Convert values to necessary format
tauTrain = dlarray(tau',"CB");
tauData = dlarray(tauData',"CB");
xData = dlarray(xData',"CB");
yData = dlarray(yData',"CB");
beta = dlarray(betaGuess);
gamma = dlarray(gammaGuess);

%initialise parameters necessary for adam optimizer
averageGrad = [];
averageSqGrad = [];
avgGradB = [];
avgSqGradB = [];
avgGradG = [];
avgSqGradG = [];

betaHistory = zeros(numEpochs,1);
gammaHistory = zeros(numEpochs,1);
lossHistory = zeros(numEpochs,1);

%initialises the UI with relevant learning metrics
monitor = trainingProgressMonitor( ...
    Metrics=["LogLoss","PhysicsLoss","DataLoss"], ...
    Info=["Epoch","LearnRate"], ...
    XLabel="Iteration");

epoch = 0;
iteration = 0;
learnRate = initialLearnRate;
start = tic;

accFcn = dlaccelerate(@modelLoss);

while epoch < numEpochs  && ~monitor.Stop
    epoch = epoch + 1; 

    iteration = iteration + 1;

    %if epoch < 15000
   %     wData = 5;
   % else
   %     wData = 1;
   % end

    % Evaluate the model gradients and loss using dlfeval and the modelLoss function
    [loss,physicsLoss,dataLoss,gradientB,gradientG,gradientsNet]...
        = dlfeval(accFcn,net, tauTrain, tauData,xData,yData,...
        wData,T,x0,y0,alpha,beta,delta,gamma,xScale,yScale);

    %Optimizer for each necessary changing variable  
    [net,averageGrad,averageSqGrad] = adamupdate(...
        net,gradientsNet,averageGrad,averageSqGrad,...
        iteration,learnRate);

    [beta,avgGradB,avgSqGradB] = adamupdate(...
        beta,gradientB,avgGradB,avgSqGradB,...
        iteration,learnRate);

    [gamma,avgGradG,avgSqGradG] = adamupdate(...
        gamma,gradientG,avgGradG,avgSqGradG,...
        iteration,learnRate);

    %Update training monitor
    recordMetrics(monitor,iteration,...
        LogLoss=log(double(gather(extractdata(loss)))), ...
        PhysicsLoss=log(double(gather(extractdata(physicsLoss)))), ...
        DataLoss=log(double(gather(extractdata(dataLoss)))));
    updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
    monitor.Progress = 100 * iteration/numEpochs;

    %Store how parameters have changed
    betaHistory(epoch) = extractdata(beta);
    gammaHistory(epoch) = extractdata(gamma);
    lossHistory(epoch) = extractdata(loss);

    %decay learning rate
    if mod(epoch, 5000) == 0 
        learnRate = learnRate * 0.5;
    end

    if gather(extractdata(log(loss))) <= targetLogLoss
        disp("Target log loss reached at epoch "+ epoch)
        targetReached = true;
        break
    end
end

%% Test model
figure
plot(betaHistory,'LineWidth',2)
hold on
yline(0.025,'r--','Target beta')
xlabel('Epoch')
ylabel('beta')
title('Estimated Predation rate coefficient')
grid on

figure
plot(gammaHistory,'LineWidth',2)
hold on
yline(0.81,'r--','Target gamma')
xlabel('Epoch')
ylabel('gamma')
title('Estimated predator death rate')
grid on

%plotting network output (predator and prey populations)
tTest = linspace(t0,t0+T,201)';
tauTest = 2*(tTest-t0)/T - 1;

tauTestDL = dlarray(tauTest',"CB");
Ntest = extractdata(forward(net,tauTestDL));
xTest = Ntest(1,:)';
yTest = Ntest(2,:)';

xPred = x0 + (tauTest+1).*xTest;
yPred = y0 + (tauTest+1).*yTest;

figure
plot(tTest,xPred,"b--")
hold on
plot(tTest,yPred,"r--")
plot(tData,xData,'b.')
plot(tData,yData,'r.')
legend("Prey","Predator")
xlabel("t")
ylabel("Population")

%% Evaluate error
rmseBeta = sqrt(mean((extractdata(beta)-0.028).^2))
rmseGamma = sqrt(mean((extractdata(gamma)-0.84).^2))



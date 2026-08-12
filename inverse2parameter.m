%PINN for solving inverse pendulum equation of the form:s
% d2_theta/dt2 + 0.5*d_theta/dt + 9.80665*theta  (sin(theta) but small angle)
% d2_theta/dt2 + c*d_theta/dt + k*theta
% IC : theta(0) = pi/4 , theta'(0) = 0

%% Generate input data and define network
t = linspace(0,4,200)';
tau = 2*(t/4) -1; %map time domain from [0,4] to [-1,1]
tSpan = linspace(0,4,20);

[tData, thetaExact] = ode45(@(t,theta) [theta(2); ...
    -0.5*theta(2) - 9.80665*theta(1)], ...
    tSpan, [pi/4;0]);
thetaData = thetaExact(:,1); 

tauData = 2*(tData/4) - 1;

layers = [
    featureInputLayer(1)
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(64)
    tanhLayer
    fullyConnectedLayer(1)];

net = dlnetwork(layers);

%% Define loss function
function [loss,gradientsNet,gradientC,gradientK] = modelLoss(...
    net,c,k,tau,tauData,thetaData,wData)

N = forward(net,tau);

% Hard-constrained trial solution
theta = pi/4 + ((tau+1).^2).*N;
% Derivatives wrt tau
dThetaTau = dlgradient(sum(theta,"all"),tau,...
    EnableHigherDerivatives=true);
d2ThetaTau = dlgradient(sum(dThetaTau,"all"),tau);

dtau_dt = 0.5;
theta_t  = dtau_dt*dThetaTau;
theta_tt = dtau_dt^2*d2ThetaTau;
 
eq = theta_tt + c*theta_t + k*theta;
physicsLoss = mean(eq.^2,"all");

%data loss
Ndata = forward(net,tauData);
thetaPredData = pi/4 + ((tauData+1).^2).*Ndata;
dataLoss = mean((thetaPredData - thetaData).^2,"all");

%Total loss
loss = physicsLoss+ wData*dataLoss;

[gradientsNet,gradientC,gradientK] = dlgradient(loss,net.Learnables,c,k);

end
%% Specify training options
numEpochs = 10000;
initialLearnRate = 0.01; 
wData = 10;
targetLogLoss = -15;
targetReached = false;
cGuess = 0.7;
kGuess = 9.0;

%% Train Model
tauTrain = dlarray(tau',"CB");
tauData = dlarray(tauData',"CB");
thetaData = dlarray(thetaData',"CB");
c = dlarray(cGuess);
k = dlarray(kGuess);

avgGradNet = [];
avgSqGradNet = [];
avgGradC = [];
avgSqGradC = [];
avgSqGradK = [];
avgGradK = [];

%want to store c,k at each loop to see it's convergence
cHistory = zeros(numEpochs,1);
kHistory = zeros(numEpochs,1);
lossHistory = zeros(numEpochs,1);

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

    [loss,gradientsNet,gradientC,gradientK] = dlfeval(accFcn,...
          net,c,k,tauTrain,tauData,thetaData,wData);

    [net,avgGradNet,avgSqGradNet] = adamupdate(...
        net,gradientsNet,avgGradNet,avgSqGradNet,...
        iteration,learnRate);

    [c,avgGradC,avgSqGradC] = adamupdate(...
        c,gradientC,avgGradC,avgSqGradC,...
        iteration,learnRate);

    [k,avgGradK,avgSqGradK] = adamupdate(...
        k,gradientK,avgGradK,avgSqGradK,...
        iteration,learnRate);

    recordMetrics(monitor,iteration,LogLoss=log(loss));
    updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
    monitor.Progress = 100 * iteration/numEpochs;
 
    cHistory(epoch) = extractdata(c);
    kHistory(epoch) = extractdata(k);
    lossHistory(epoch) = extractdata(loss);

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
figure
plot(cHistory,'LineWidth',2)
hold on
yline(0.5,'r--','True c')
xlabel('Epoch')
ylabel('c')
title('Estimated damping coefficient during training')
grid on

figure
plot(kHistory,'LineWidth',2)
hold on
yline(9.80655,'r--','True k')
xlabel('Epoch')
ylabel('k')
title('Estimated stiffness coefficient during training')
grid on

tTest = linspace(0,5,100)';
tauTest = 2*(tTest/4) - 1;
tauTestDL = dlarray(tauTest',"CB");
Ntest = extractdata(forward(net,tauTestDL))';
thetaPred = pi/4 + ((tauTest+1).^2).*Ntest;

thetaAnalytic = exp(-0.25*tTest).*((pi/4)*cos(3.121562109*tTest) ...
    + (pi/16)*(1/3.121562109)*sin(3.121562109*tTest));


figure
plot(tTest,thetaAnalytic, "-")
hold on
plot(tTest,thetaPred,'r--')
xlabel("t")
ylabel("\theta")
plot(tSpan,thetaData,'g*')
legend("Analytic","Model","Solution data points")
hold off
%% Evaluate error
errorC = abs(extractdata(c) - 0.5)
errorK = abs(extractdata(k) - 9.80655)

rmse = sqrt(mean((thetaPred-thetaAnalytic).^2))

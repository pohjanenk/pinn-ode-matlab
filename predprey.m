%PINN model with ode45 generated data and and Lotka-Volterra differential
%equation
%% Constants,data and network architecture 
x0 = 30;
y0 = 4;

alpha = 0.55;
beta = 0.028;
delta = 0.026;
gamma = 0.84;
T = 100;

%Collocation points
t = linspace(0,T,100)';
tau = t*2/T -1;

%Generate data
tspan = linspace(0,T,20);
[tData,Z] = ode45(@(t,z) [alpha*z(1) - beta*z(1)*z(2);
    delta*z(1)*z(2) - gamma*z(2)], tspan, [x0; y0]);
tauData = 2*tData/T - 1;
xData = Z(:,1);
yData = Z(:,2);

%plot(tspan,xData,'b')
%hold on
%plot(tspan,yData,'r')

% Scaling factors
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
function [loss,physicsLoss,dataLoss,gradients] = modelLoss(...
    net,tau,tauData,xData,yData,...
    wData,wPhysics,...
    T,x0,y0,alpha,beta,delta,gamma,...
    xScale,yScale)
%Retrieve network output
N = forward(net,tau);
Nx = N(1,:);
Ny = N(2,:);

x = x0 + (tau+1).*Nx;
y = y0 + (tau+1).*Ny;

dxTau = dlgradient(sum(x,"all"),tau,...
    EnableHigherDerivatives=true);
dyTau = dlgradient(sum(y,"all"),tau,...
    EnableHigherDerivatives=true);

dtau_dt = 2/T;
dxdt = dtau_dt*dxTau;
dydt = dtau_dt*dyTau;

eq1 = dxdt - (alpha*x - beta*x.*y);
eq2 = dydt - (delta*x.*y - gamma*y);

eq1 = eq1/xScale;
eq2 = eq2/yScale;

physicsLoss = mean(eq1.^2,"all") + mean(eq2.^2,"all");
%
Ndata = forward(net,tauData);

NxData = Ndata(1,:);
NyData = Ndata(2,:);

xPred = x0 + (tauData+1).*NxData;
yPred = y0 + (tauData+1).*NyData;

xPredNorm = xPred ./ xScale;
yPredNorm = yPred ./ yScale;

xDataNorm = xData ./ xScale;
yDataNorm = yData ./ yScale;


dataLoss = mean((xPredNorm-xDataNorm).^2,"all") + ...
           mean((yPredNorm-yDataNorm).^2,"all");

%
loss = wPhysics*physicsLoss + wData*dataLoss;

gradients = dlgradient(loss, net.Learnables);
end
%% Specify training options
numEpochs = 30000;
initialLearnRate = 0.01; 
targetLogLoss = -15;
targetReached = false;
wPhysics = 1;
wData = 3;

%% Train Model
tauTrain = dlarray(tau',"CB");
tauData = dlarray(tauData',"CB");
xData = dlarray(xData',"CB");
yData = dlarray(yData',"CB");

averageGrad = [];
averageSqGrad = [];
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

    %if epoch < 5000
   %     wData = 5;
   % else
   %     wData = 1;
   % end

    % Evaluate the model gradients and loss using dlfeval and the modelLoss function
    [loss,physicsLoss,dataLoss,gradients] = dlfeval(accFcn,...
        net, tauTrain, tauData,xData,yData,...
        wData,wPhysics,...
        T,x0,y0,alpha,beta,delta,gamma,...
        xScale,yScale);

    [net,averageGrad,averageSqGrad] = adamupdate(...
        net,gradients,averageGrad,averageSqGrad,...
        iteration,learnRate);

    recordMetrics(monitor,iteration,...
        LogLoss=log(double(gather(extractdata(loss)))), ...
        PhysicsLoss=log(double(gather(extractdata(physicsLoss)))), ...
        DataLoss=log(double(gather(extractdata(dataLoss)))));
    updateInfo(monitor,Epoch=epoch,LearnRate=learnRate);
    monitor.Progress = 100 * iteration/numEpochs;


    if mod(epoch, 5000) == 0 %&& epoch < 24000
        learnRate = learnRate * 0.5;
    end


    if gather(extractdata(log(loss))) <= targetLogLoss
        disp("Target log loss reached at epoch "+ epoch)
        targetReached = true;
        break
    end
end

%% Test model
tTest = (0:T)';
tauTest = 2*tTest/T - 1;

tauTestDL = dlarray(tauTest',"CB");
Ntest = extractdata(forward(net,tauTestDL));
xTest = Ntest(1,:)';
yTest = Ntest(2,:)';

xPred = x0 + (tauTest+1).*xTest;
yPred = y0 + (tauTest+1).*yTest;

[T,Z1] = ode45(@(t1,z1) [alpha*z1(1) - beta*z1(1)*z1(2);
    delta*z1(1)*z1(2) - gamma*z1(2)], tTest, [x0; y0]);
xExact = Z1(:,1);
yExact = Z1(:,2);

figure;
plot(tTest,xExact, "b-")
hold on
plot(tTest,yExact, "r-")
plot(tTest,xPred,"y--")
plot(tTest,yPred,"g--")
legend("Prey","Predator")
xlabel("t")
ylabel("Population")

%% Evaluate error
rmseX = sqrt(mean((xPred-xExact).^2))
rmseY = sqrt(mean((yPred-yExact).^2))



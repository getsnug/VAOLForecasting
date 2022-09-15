% Construct DNN architecture
network = MLPNet();
network.AddInputLayer(5,false);
network.AddHiddenLayer(10,'tanh',false);
network.AddHiddenLayer(10,'tanh',false);
network.AddHiddenLayer(10,'tanh',false);
network.AddOutputLayer(1,'tanh',false);
network.NetParams('rate',0.0005,'momentum','adam','lossfun','rmse',...
    'regularization','none');
network.trainable = true;
network.Summary();

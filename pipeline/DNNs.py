import torch
from pipeline import datasets
from pipeline import helpers
import numpy as np
import pytorch_lightning as pl
from pywick.optimizers import nadam
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    default_device = torch.device("cuda")
else:
    default_device = torch.device("cpu")


class Soup(pl.LightningModule):
    '''
    Superclass that sets up optimisers, training step, validation, dataloaders, ..., that are common to all models.
    '''

    def __init__(self,
                 data_dir="/home/mdt20/Code/hugo_data_processing/processed_data/250321/data.h5",
                 input_length=50,
                 input_offset=0,
                 channels = np.arange(64),
                 var_regulariser=0.00,
                 regularisation_type="l2",
                 weight_decay=0.01,
                 correlation_weight=1.0,
                 mse_weight=0.0,
                 batch_size=512,
                 val_batch_size=512,
                 learning_rate=1e-5,
                 train_participants=[0],
                 val_participants=None,
                 test_participants=[1],
                 train_parts=[0],
                 test_parts=[1],
                 val_parts=[2]):
        """
        config parameters:
          - input_length=50, number of input time samples.
          - input_offset=0, offset to predict speech envelope relative to start of input (controls amount of future/past information)
          - var_regulariser=0, controls penalisation of model prediction variance (want a large variance for bimodal distribution)
          - weight_decay=0.01, L2 regularisation
          - num_hidden=1, number of hidden fully-connected layers
          - batch_size=32, training batch size
          - val_batch_size=512, validation batch size. Don't load the entire dataset at once because this might deplete GPU memory.
          - learning_rate=1e-5, initial learning rate
          - correlation_weight=1, weight assigned to negative correlation in loss function
          - mse_weight=0, weight assigned to mean-squared-error term in loss function
          - dropout_rate=0.25, probability of dropping neurons during training
          - story parts=np.random.permutation, randomly shuffled sequence of trials to be used for training/validation/prediction.
        """
        super(Soup, self).__init__()

        self.data_dir=data_dir
        self.input_length = input_length
        self.input_offset = input_offset
        self.eeg_channels = channels
        self.num_input_channels = len(channels)
        self.var_regulariser = var_regulariser
        self.regularisation_type = regularisation_type
        self.weight_decay=weight_decay
        self.batch_size=batch_size
        self.val_batch_size = val_batch_size
        self.lr = learning_rate
        self.train_participants = train_participants
        self.test_participants = test_participants
        if val_participants == None:
            self.val_participants=self.train_participants
        else:
            self.val_participants=val_participants
        self.train_parts=train_parts
        self.val_parts=val_parts
        self.test_parts=test_parts

        self.correlation_weight = correlation_weight/(correlation_weight + mse_weight)
        self.mse_weight = mse_weight/(correlation_weight + mse_weight)

    def forward(self, x):
        '''
        define the forward pass
        '''
        raise NotImplementedError()

    def training_step(self, train_batch, batch_idx):
        '''
        input: training batch (x, y) and corresponding indices.
        output: loss function over the batch (for backpropogation)
        '''

        x, y = train_batch
        x = x.to(device=self.device, dtype=torch.float)
        y = y.to(device=self.device, dtype=torch.float)
        #y = helpers.linear_detrend(y)

        predictions = self.forward(x)
        mse = helpers.mse_loss(predictions, y)
        correlation = helpers.correlation(predictions, y)
        variance = helpers.variance(predictions)
        loss = self.mse_weight*mse - self.correlation_weight*correlation - self.var_regulariser*variance 

        self.log("ptl/train_loss", loss)
        self.log("ptl/train_correlation", correlation, prog_bar=True, on_step=True)
        self.log("ptl/train_mse", mse)
        self.log("ptl/train_variance", variance)

        if self.regularisation_type == "l1":

            l1 = 0
            for p in self.parameters():
                l1 = l1 + p.abs().sum()

            loss += l1
 
        return loss

    def validation_step(self, val_batch, batch_idx):
        '''
        input: validation batch (x, y) and corresonding indices.
        output: validation metrics (to be logged in validation_epoch_end)
        '''
        
        x, y = val_batch
        x = x.to(device=self.device, dtype=torch.float)
        y = y.to(device=self.device, dtype=torch.float)


        predictions = self.forward(x)
        mse = helpers.mse_loss(predictions, y)
        correlation = helpers.correlation(predictions, y)
        variance = helpers.variance(predictions)
        loss = self.mse_weight*mse - self.correlation_weight*correlation - self.var_regulariser*variance

        return {"val_loss": loss, "val_correlation": correlation, "val_mse": mse, "val_variance": variance}

    def validation_epoch_end(self, outputs):
        '''
        log the validation results.
        It would be better to measure these quantities over the entire validation set, but this is likely to deplete CUDA memory.
        Notes:
        -   The mean (over batches) correlation coefficient is not an unbiased estimator of the overall validation correlation coefficient. It is asymptotically unbiased under some conditions... https://stats.stackexchange.com/questions/220961.
        -   The variance measurement is unbiased (default mode in pytorch.)        
        '''

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_corr = torch.stack([x["val_correlation"] for x in outputs]).mean()
        avg_mse = torch.stack([x["val_mse"] for x in outputs]).mean()
        avg_var = torch.stack([x["val_variance"] for x in outputs]).mean()

        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_correlation", avg_corr)
        self.log("ptl/val_mse", avg_mse)
        self.log("ptl/val_variance", avg_var)


    def train_dataloader(self):
        '''
        Create the train DataLoader. 
        '''
        dataset = datasets.HugoMapped(self.train_parts, self.data_dir, participant=self.train_participants, num_input=self.input_length,channels=self.eeg_channels)
        return DataLoader(dataset, batch_size=int(self.batch_size), sampler = torch.randperm(len(dataset)), num_workers=16, pin_memory=True)

    def val_dataloader(self):
        '''
        Create the validation DataLoader
        '''
        dataset = datasets.HugoMapped(self.val_parts, self.data_dir, participant=self.val_participants, num_input=self.input_length,channels=self.eeg_channels)
        return DataLoader(dataset, batch_size=int(self.val_batch_size), num_workers=16, pin_memory=True)

    def test_dataloader(self, test_batch_size=512):
        '''
        Create the test DataLoader
        '''
        dataset = datasets.HugoMapped(self.test_parts, self.data_dir, participant=self.test_participants, num_input=self.input_length,channels=self.eeg_channels)
        return DataLoader(dataset, batch_size=test_batch_size, pin_memory=True)

    def configure_optimizers(self):
        '''
        set up optimizer prior to training
        '''
        if self.regularisation_type == "l2":
            optimizer = nadam.Nadam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.regularisation_type == "l1":
            optimizer = nadam.Nadam(self.parameters(), lr=self.lr, weight_decay=0)
        return optimizer

    def test_results(self):
        '''
        Assess performance on test dataset
        '''
        loader = self.test_dataloader()
        batch_outputs = []

        for test_batch in loader:
            x, y = test_batch
            x = x.to(device=self.device, dtype=torch.float)
            y = y.to(device=self.device, dtype=torch.float)
            y_hat = self.eval()(x).detach()
            batch_outputs.append({"test_predictions": y_hat, "test_targets": y})

        test_predictions = torch.cat([x['test_predictions'] for x in batch_outputs])
        test_targets = torch.cat([x['test_targets'] for x in batch_outputs])
        correlation = helpers.correlation(test_predictions, test_targets)

        return {"predictions": test_predictions, "targets": test_targets, "correlation": correlation}

class DeTaillez(Soup):

    def __init__(self,
                 num_hidden=4,
                 dropout_rate=0.25,
                 **kwargs):
        """
        config parameters:
          - num_input_channels=64, number of electrodes used as model input
          - num_hidden=1, number of hidden fully-connected layers
          - dropout_rate=0.25, probability of dropping neurons during training
        """
        super(DeTaillez, self).__init__(**kwargs)

        self.num_hidden = num_hidden
        units = np.round(np.linspace(1, self.input_length*self.num_input_channels, self.num_hidden+2)[::-1]).astype(int) #hidden units in each fully-connected layer
        self.fully_connected = torch.nn.ModuleList([torch.nn.Linear(units[i], units[i+1]) for i in range(len(units)-1)])
        self.activations = torch.nn.ModuleList([torch.nn.Tanh() for i in range(len(units)-2)])
        self.dropouts = torch.nn.ModuleList([torch.nn.Dropout(p=dropout_rate) for i in range(len(units)-2)])

    def forward(self, x):
        '''
        define the forward pass
        '''
        x = x.flatten(start_dim=1)
        for i, layer in enumerate(self.fully_connected[:-1]):
          x = layer(x)
          x = self.activations[i](x)
          x = self.dropouts[i](x)
        
        x = self.fully_connected[-1](x)
        return x.flatten()

class EEGNet(Soup):

    def __init__(self,
                 temporalFilters=8,
                 spatialFilters=8,
                 TemporalPool1=1,
                 TemporalPool2=1,
                 temporalConvKernelSize=3,
                 dropout_rate=0.25,
                 activation="ELU",
                 **kwargs):

        super(EEGNet, self).__init__(**kwargs)
 
        self.F1=temporalFilters
        self.F2=temporalFilters
        self.D=spatialFilters
        self.TemporalPool1=TemporalPool1
        self.TemporalPool2=TemporalPool2
        
        self.conv1_kernel = temporalConvKernelSize
        self.conv3_kernel = temporalConvKernelSize
        
        self.conv1 = torch.nn.Conv2d(1, self.F1, (1, self.conv1_kernel), padding=(0, self.conv1_kernel//2))
        self.conv2 = torch.nn.Conv2d(self.F1, self.F1*self.D, (self.num_input_channels, 1))
        self.pool1 = torch.nn.AvgPool2d((1, self.TemporalPool1))
        self.conv3 = torch.nn.Conv2d(1, self.F2, (1, self.conv3_kernel), padding = (0, self.conv3_kernel//2))
        self.conv4 = torch.nn.Conv2d(self.F1*self.D, 1, (1,1))
        self.pool2 = torch.nn.AvgPool2d((1, self.TemporalPool2))
                
        self.linear = torch.nn.Linear(self.F2*self.input_length//TemporalPool1*TemporalPool2, 1)

        self.bnorm1 = torch.nn.BatchNorm2d(self.F1)
        self.bnorm2 = torch.nn.BatchNorm2d(self.F1*self.D)
        self.bnorm3 = torch.nn.BatchNorm2d(1)

        self.dropout1 = torch.nn.Dropout2d(dropout_rate)
        self.dropout2 = torch.nn.Dropout2d(dropout_rate)

        if activation == "ELU":
            self.activation1 = torch.nn.ELU()
            self.activation2 = torch.nn.ELU()
            self.activation3 = torch.nn.ELU()
        if activation == "LeakyRELU":
            self.activation1 = torch.nn.LeakyReLU(0.4)
            self.activation2 = torch.nn.LeakyReLU(0.4)
            self.activation3 = torch.nn.LeakyReLU(0.4)
        if activation == "Tanh":
            self.activation1 = torch.nn.Tanh()
            self.activation2 = torch.nn.Tanh()
            self.activation3 = torch.nn.Tanh()

    def forward(self, x):
        #x shape = [batch, C, T]
        x = x.unsqueeze(1)
        out = self.conv1(x)
        out = self.bnorm1(out)

        out = self.conv2(out)
        out = self.bnorm2(out)

        out = self.activation1(out)
        out = self.pool1(out)
        out = self.dropout1(out)

        #shape is [batch, DxF1, 1, T//TPool1]
        #reshape to [batch, 1, DxF1, T//TPool1]
        #out = out[:, None, :, 0, :]
        out = out.squeeze().unsqueeze(1)


        out = self.conv3(out) #shape is now [batch, F2, DxF1, T//TPool1]
        out = self.activation2(out)
        out = out.swapdims(1, 2)
        out = self.conv4(out)
        out = self.bnorm3(out)

        out = self.pool2(out) #shape is now [batch, 1, F2, T//(TPool1*TPool2)]
        out = self.dropout2(out)
        
        out = torch.flatten(out, start_dim = 1) # shape is now [batch, F2*T//(TPool1*TPool2)]
        out = self.linear(out)
        return out.flatten() #need to change for multidimensional outputs
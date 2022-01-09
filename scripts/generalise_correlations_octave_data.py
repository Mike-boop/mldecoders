from pipeline.DNNs import DeTaillez, EEGNet
from pipeline import helpers
from pipeline.datasets import OctaveMapped
import json
from torch.utils.data import DataLoader
import torch.multiprocessing
import pickle
import os

data_dir = "/media/mdt20/Storage/data/Octave/280921F"
resultsdir = "/home/mdt20/Code/mldecoders/results/0.5-32Hz/"

def study(participant, test_batch_size, condition='fM', parts=range(1,5), data_dir=""):
    data = OctaveMapped(50,participant, condition, parts, data_dir=data_dir, return_null=True)
    loader = DataLoader(data, batch_size=test_batch_size, num_workers=16, drop_last=True)
    ridge_eeg, ridge_stim = data.raw_data()
    ridge_stim_null = data.stim_u_null, data.stim_a_null

    eegnet_config = json.load(open(
        os.path.join(resultsdir, "eegnet/best_config.json")
    ))

    detaillez_config = json.load(open(
        os.path.join(resultsdir, "detaillez/best_config.json")
    ))

    eegnet_checkpoint_path = os.path.join(resultsdir, "eegnet/population.ckpt")
    detaillez_checkpoint_path = os.path.join(resultsdir, "detaillez/population.ckpt")

    eegnet = EEGNet.load_from_checkpoint(checkpoint_path=eegnet_checkpoint_path,**eegnet_config, data_dir="/home/mdt20/Code/hugo_data_processing/processed_data/230621/data.h5")
    eegnet.eval()
    eegnet.to(device='cuda')
    detaillez = DeTaillez.load_from_checkpoint(checkpoint_path=detaillez_checkpoint_path,**detaillez_config, data_dir="/home/mdt20/Code/hugo_data_processing/processed_data/230621/data.h5")
    detaillez.eval()
    detaillez.to(device='cuda')

    ridge_path = os.path.join(resultsdir, "ridge/population_mdl.pickle")
    ridge = pickle.load(open(ridge_path, "rb"))

    corrs = {key: [] for key in ["detaillez_attended", "detaillez_attended_null",
                                "eegnet_attended", "eegnet_attended_null",
                                "ridge_attended", "ridge_attended_null",
                                "detaillez_unattended", "detaillez_unattended_null",
                                "eegnet_unattended", "eegnet_unattended_null",
                                "ridge_unattended", "ridge_unattended_null"]}


    corrs["ridge_attended"] = ridge.score_in_batches(ridge_eeg.T, ridge_stim[1][:, None], batch_size=test_batch_size).tolist()
    corrs["ridge_attended_null"] = ridge.score_in_batches(ridge_eeg.T, ridge_stim_null[1][:, None], batch_size=test_batch_size).tolist()
    corrs["ridge_unattended"] = ridge.score_in_batches(ridge_eeg.T, ridge_stim[0][:, None], batch_size=test_batch_size).tolist()
    corrs["ridge_unattended_null"] = ridge.score_in_batches(ridge_eeg.T, ridge_stim_null[0][:, None], batch_size=test_batch_size).tolist()


    for x, y, y_null in loader:

        x = x.to(dtype=torch.float, device=eegnet.device)
        y[1] = y[1].to(dtype=torch.float, device=eegnet.device)
        y[0] = y[0].to(dtype=torch.float, device=eegnet.device)
        y_null[1] = y_null[1].to(dtype=torch.float, device=eegnet.device)
        y_null[0] = y_null[0].to(dtype=torch.float, device=eegnet.device)

        eegnet_predictions = eegnet(x)
        detaillez_predictions = detaillez(x)

        dt_corr_a = helpers.correlation(detaillez_predictions, y[1]).item()
        en_corr_a = helpers.correlation(eegnet_predictions, y[1]).item()
        dt_corr_null_a = helpers.correlation(detaillez_predictions, y_null[1]).item()
        en_corr_null_a = helpers.correlation(eegnet_predictions, y_null[1]).item()

        dt_corr_u = helpers.correlation(detaillez_predictions, y[0]).item()
        en_corr_u = helpers.correlation(eegnet_predictions, y[0]).item()
        dt_corr_null_u = helpers.correlation(detaillez_predictions, y_null[0]).item()
        en_corr_null_u = helpers.correlation(eegnet_predictions, y_null[0]).item()

        corrs["detaillez_attended"].append(dt_corr_a)
        corrs["detaillez_attended_null"].append(dt_corr_null_a)
        corrs["eegnet_attended"].append(en_corr_a)
        corrs["eegnet_attended_null"].append(en_corr_null_a)

        corrs["detaillez_unattended"].append(dt_corr_u)
        corrs["detaillez_unattended_null"].append(dt_corr_null_u)
        corrs["eegnet_unattended"].append(en_corr_u)
        corrs["eegnet_unattended_null"].append(en_corr_null_u)

    return corrs

if __name__ == '__main__':

    participants = ["YH00", "YH01", "YH02", "YH03", "YH06", "YH07", "YH08", "YH09", "YH10", "YH11", "YH14", "YH15", "YH16", "YH17", "YH18", "YH19", "YH20"]

    for batch_size in [250]:

        results = {}

        for condition in ['clean', 'lb', 'mb', 'hb', 'fM', 'fW']:
            results[condition] = {}
            for each_participant in participants:

                results[condition][each_participant] = study(each_participant, batch_size, condition, [1,2,3,4], data_dir=data_dir)

                print("Done: ", each_participant, condition, batch_size)

        json.dump(results, open(os.path.join(resultsdir,"generalisation/{}_octave_correlations.json".format(batch_size)), "w"))

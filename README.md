# correncoder_ppg_respiration
Rapid extraction of respiratory waveforms from PPG with a deep convolutional correncoder 

PyTorch code for the following paper:

Harry J. Davies, Danilo P. Mandic, "Rapid extraction of respiratory waveforms from photoplethysmography: A deep corr-encoder approach",
Biomedical Signal Processing and Control, Volume 85, 2023,
ISSN 1746-8094, https://doi.org/10.1016/j.bspc.2023.104992.
(https://www.sciencedirect.com/science/article/pii/S1746809423004251)

Please cite this work when using this code.

Abstract:

Much of the information related to breathing is contained within the photoplethysmography (PPG) signal, through changes in venous blood flow, heart rate and stroke volume. We aim to leverage this fact, by employing a novel deep learning framework which is a based on a repurposed convolutional autoencoder. Our corr-encoder model aims to encode all of the relevant respiratory information contained within photoplethysmography waveform, and decode it into a waveform that is similar to a gold standard respiratory reference — the Capnogram. The model is employed on two photoplethysmography data sets, namely Capnobase and BIDMC. We show that the model is capable of producing respiratory waveforms that approach the gold standard, while in turn producing state of the art respiratory rate estimates. We also show that when it comes to capturing more advanced respiratory waveform characteristics such as duty cycle, our model is for the most part unsuccessful. A suggested reason for this, in light of a previous study on in-ear PPG, is that the respiratory variations in finger-PPG are far weaker compared with other recording locations. Importantly, our model can perform these waveform estimates in a fraction of a millisecond, giving it the capacity to produce over 6 hours of respiratory waveforms in a single second. Moreover, we attempt to interpret the behaviour of the kernel weights within the model, showing that in part our model intuitively selects different breathing frequencies. The model proposed in this work could help to improve the usefulness of consumer PPG-based wearables for medical applications, where detailed respiratory information is required.

Important information:

CPU vs GPU:
These models are small and trainable on CPU. It is straightforward to adapt them to a GPU (if one is present and set up) by defining torch tensors with ".cuda()" and the model with ".cuda()".

Data:
This model was initially trained with publicly available datasets Capnobase and BIDMC.
Capnobase: Karlen, Walter, et al. "Multiparameter respiratory rate estimation from the photoplethysmogram." IEEE Transactions on Biomedical Engineering 60.7 (2013): 1946-1953.
BIDMC: Pimentel, Marco AF, et al. "Toward a robust estimation of respiratory rate from pulse oximeters." IEEE Transactions on Biomedical Engineering 64.8 (2016): 1914-1923
The Capnobase dataset consists of simultaneously recorded capnography and finger PPG. The BIDMC dataset consists of simultaneously recorded impedance pnuemography and finger PPG.
The same model has more recently been implemented for in-ear PPG, with a dataset consisting of simultaneously recorded airflow from the lungs and in-ear PPG. These findings were presented at the 45th Engineering in Medicine and Biology conference (EMBC) in Sydney 2023, with the paper: Davies, Harry J, et al. "Feasibility of Transfer Learning from Finger PPG to In-Ear PPG"
This model is effective at extracting respiratory waveforms from PPG as long as it is trained with a simultaneously recorded PPG as an input and respiration as a training reference.

Code:
Included in this repo is the code used to train the model "correncoder_capnobase_training.py" and save it.
To load a trained model, simply define the model function and use the following code: model.load_state_dict(torch.load(model_path)), where the path is the file location and name of the saved model.


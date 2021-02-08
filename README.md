# ChopinBotApp
Web application for piano music generation in the style of Chopin. Powered by Streamlit.io

Currently deployed at https://share.streamlit.io/msgilmer/chopinbotapp/main/ChopinBotApp.py

Following Flowchart created with [LucidChart](https://lucidchart.com):
![](./images/ChopinBotApp_flowchart.jpg)

</br></br>
<b>Directory Structure</b>

    .
    ├── ChopinBotApp.py               # The script that is deployed.
    ├── ChopinBotApp_local.py         # The local version (run as streamlit run ChopinBotApp_local.py).
    ├── X_val                       
    │   ├── X_val_ext_1.npy           # The X part of the validation set broken into pieces to avoid git-lfs (not currently compatible with Streamlit). Copied from msgilmer/Springboard_Capstone/train_and_val.
    │   ├── booleanized_sequences.npy # The same validation set broken up into the binary classification part (saved with dtype = np.bool) and the duration part. Generated from the above two via ../separate_X_val.py.
    │   └── durations.npy
    ├── images                        # Contents copied from msgilmer/Springboard_Capstone/images. Images posted to application.
    │   ├── chopin.jpg    
    │   ├── precision_and_recall.jpg
    ├── logfiles                      # Logfiles for each unique session. Used for monitoring errors.
    │   └── logzero*log
    ├── midi_output                   # Directory to store generated MIDI files for download for the local version only.
    │   ├── generated.mid
    │   └── seed.mid
    ├── models                        # Directory to store pre-trained keras model weights, copied from msgilmer/Springboard_Capstone/models.
    │   └── best_maestro_model_weights_ext20_2_1_1024_0pt4_mnv_2.h5
    ├── requirements.txt              # Required to deploy the app. Generated with pipreqs.
    ├── rndm_seed_index_files         # Text files that uniquely store the last randomly-generated seed index (for the X_val set) for each unique session.
    │   └── rndm*txt
    ├── separate_X_val.py             # Script used to save the X_val set into a more energy-efficient format (X_val/booleanized_sequences.npy and X_val/durations.npy) for consuption by ChopinBotApp*py.
    └── ...
    
    │   ├── X_val_ext_2.npy

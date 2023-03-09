Generates multi-instrument symbolic music (MIDI), based on user-provided emotions from valence-arousal plane. In simpler words, it can generate happy (positive valence, positive arousal), calm (positive valence, negative arousal), angry (negative valence, positive arousal) or sad (negative valence, negative arousal) music.

Source code for our paper "Symbolic music generation conditioned on continuous-valued emotions", 
Serkan Sulun, Matthew E. P. Davies, Paula Viana, 2022. 
https://ieeexplore.ieee.org/document/9762257

To cite:
```S. Sulun, M. E. P. Davies and P. Viana, "Symbolic music generation conditioned on continuous-valued emotions," in IEEE Access, doi: 10.1109/ACCESS.2022.3169744.```

Required Python libraries: Numpy, Pytorch, Pandas, pretty_midi, Pypianoroll, tqdm, Spotipy, Pytables. Or run: ```pip install -r requirements.txt```

To create the Lakh-Spotify dataset:

- Go to the ```src/create_dataset``` folder

- Download the datasets:

Lakh pianoroll 5 full dataset
https://drive.google.com/uc?id=19FJwmNxWUmR3UutCDhZtbKVnEBVHU-2q

MSD summary file
http://labrosa.ee.columbia.edu/millionsong/sites/default/files/AdditionalFiles/msd_summary_file.h5

Echonest mapping dataset
```ftp://ftp.acousticbrainz.org/pub/acousticbrainz/acousticbrainz-labs/download/msdrosetta/millionsongdataset_echonest.tar.bz2```
Alternatively: https://drive.google.com/file/d/1AZctGV7WysvsAaDCPWM1GVBvgaFz2Dys/view?usp=sharing

Lakh-MSD matching scores file
http://hog.ee.columbia.edu/craffel/lmd/match_scores.json

- Extract when necessary, and place all inside folder ```./data_files```

- Get Spotify client ID and client secret:
https://developer.spotify.com/dashboard/applications
Then, fill in the variables "client_id" and "client_secret" in ```src/create_dataset/utils.py```

- Run ```run.py```. 

To preprocess and create the training dataset:

- Go to the ```src/data``` folder and run ```preprocess_pianorolls.py```


To generate MIDI using pretrained models:

- Download model(s) from the following link:
https://drive.google.com/drive/folders/1qFgomRDXmx8ewRz_E0X2T10CY-w_kUFt?usp=sharing

- Extract into the folder ```output```

- Go to ```src``` folder and run ```generate.py``` with appropriate arguments. e.g:
```python generate.py --model_dir continuous_concat --conditioning continuous_concat --valence -0.8, -0.8 0.8 0.8 --arousal -0.8 -0.8 0.8 0.8```


To train:

- Go to ```src``` folder and run ```train.py``` with appropriate arguments. e.g:
```python train.py --conditioning continuous_concat```

There are 4 different conditioning modes:
```none```: No conditioning, vanilla model.
```discrete_token```: Conditioning using discrete tokens, i.e. control tokens.
```continuous_token```: Conditioning using continuous values embedded as vectors, then prepended to the other embedded tokens in sequence dimension.
```continuous_concat```: Conditioning using continuous values embedded as vectors, then concatenated to all other embedded tokens in channel dimension.

See ```config.py``` for all options.

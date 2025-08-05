
# imports
import librosa

from tqdm import tqdm

from random import sample
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import manifold
from scipy.linalg import svd, toeplitz
from scipy.stats import zscore
import numpy as np
from scipy.stats import entropy
from statsmodels.tsa.stattools import acf

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import soundfile as sf
import tempfile

import lap

from pyha_analyzer import extractors
from pyha_analyzer.models import EfficentNet
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors

import numpy as np
import torch
import os


lag = 512

# juan colonna entropy

def Entropy(p1):
    p1 = p1/np.sum(p1)
    return entropy(p1)/np.log(len(p1))
# EGCI calculation from https://github.com/juancolonna/EGCI/blob/master/Example_of_EGCI_calculation.ipynb

def JSD(p):
    n = len(p)
    q = np.ones(n)/n # Uniform reference
    p = np.asarray(p)
    q = np.asarray(q)
    p = p/p.sum() # normalize
    m = (p + q) / 2
    jensen0 = -2*((((n+1)/n)*np.log(n+1)-2*np.log(2*n) + np.log(n))**(-1))
    return jensen0*(entropy(p, m) + entropy(q, m)) / 2

def EGCI(x):
    x = zscore(x)
    
    # Algorithm steps 
    rxx = acf(x, nlags=lag, adjusted=True, fft=True) #https://github.com/blue-yonder/tsfresh/issues/902
    Sxx = toeplitz(rxx)
    U, s, Vt = np.linalg.svd(Sxx) #svd(Sxx)
    
    print(JSD(s))
    
    return Entropy(s), Entropy(s)*JSD(s) 

def process_data(path):
    print(path)
    
    if (".WAV" not in path) and (".wav" not in path):
        print("directory found")
        return None
    
    i = 0
    audio, sr = librosa.load(path, duration=5, offset=i)

    
    if librosa.get_duration(y = audio, sr = sr) == 0:
        return None
    
    h, c = EGCI(audio)

    
    
    output_data = {
            "path": path,
            "offset_s": 0,
            "sr": sr,
            # "gt": data['labels'],
            # "site": data['site_name'],
            "entropy": h,
            "complexity": c
        }
    return output_data

import bioacoustics_model_zoo as bmz
# Perch can only generate embeddings for audio files greater than 5 seconds. Therefore, loop any short audio files to make it atleast 5 seconds
def pad_short_clip(audio_path):
    target_duration_sec = 5
    samplerate=sf.info(audio_path).samplerate
    target_len = samplerate * target_duration_sec
    y, sr = librosa.load(audio_path, sr=samplerate)
    #pad if less than 5 seconds
    if len(y) < target_len:
        reps = int(np.ceil(target_len / len(y)))
        y = np.tile(y, reps)[:target_len]
    return np.asarray(y, dtype=np.float32), sr
def generate_embedding(audio_path):
    info = sf.info(audio_path)
    duration = info.frames / info.samplerate #faster than using librosa to load length
    if (duration < 5):
        formatted_wav, sample_rate = pad_short_clip(audio_path)
        #creates a new wav file of 5 seconds long to generate embedding and then immediately deletes it
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            # Write the array to the temp .wav file
            sf.write(tmp.name, formatted_wav, sample_rate)
            # Use the file path for embedding
            embedding = model.embed(tmp.name)
    # if >=5 seconds then embed directly
    else:
        embedding = model.embed(audio_path)
    return embedding, duration
# # Generate vector embeddings of all audios using perch embeddings and insert into lancedb

model=bmz.Perch()
# model = EfficentNet(num_classes=2)




# getting all the wavs

music_extractor = extractors.Music()
music_ads = music_extractor("audio")
preprocessor = MelSpectrogramPreprocessors(duration=5, class_list=music_ads["train"].features["labels"].feature.names)

music_ads["train"].set_transform(preprocessor)
music_ads["test"].set_transform(preprocessor)
music_ads["valid"].set_transform(preprocessor)

embeddings = []
images = []

for filepath in music_ads['train'][0:]['filepath']:
    processed = process_data(filepath)
    if processed is not None:
        embeddings.append([processed['entropy'], processed['complexity']])
        images.append(filepath)
        

embeddings = np.array(embeddings[0:100])

# for split in ['train','test', 'valid']:
#     data = music_ads[split][0:]['filepath']


#     embeddings_split = model.embed(data)
#     embeddings.append(embeddings_split)



# embeddings = pd.concat(embeddings, axis=0)[0:100]


# get 2d embeddings
tsne = manifold.TSNE(random_state = 1, n_components=2, learning_rate=10)
data2d = tsne.fit_transform(embeddings)

data2d -= data2d.min(axis=0)
data2d /= data2d.max(axis=0)

data2d = np.array(data2d)



plt.figure(figsize=(4, 4))
plt.scatter(data2d[:,0], data2d[:,1], edgecolors='none', marker='o', s=12)
plt.show()

side = 10
xv, yv = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
grid = np.dstack((xv, yv)).reshape(-1, 2)


from scipy.spatial.distance import cdist
cost = cdist(grid, data2d, 'sqeuclidean')
cost = cost * (10000000. / cost.max())




cmap = cm.viridis
norm = Normalize(vmin=0, vmax=3) # Adjust vmin/vmax based on your data range

min_cost, row_assigns, col_assigns = lap.lapjv(np.copy(cost))
grid_jv = grid[col_assigns]
print(col_assigns.shape)
plt.figure(figsize=(4, 4))
for index, (start, end) in enumerate(zip(data2d, grid_jv)):
    arrow_color = cmap(1)
    
    plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1],
              head_length=0.01, head_width=0.01, fc=arrow_color, ec=arrow_color)
plt.show()


from mpl_toolkits.axes_grid1 import ImageGrid


fig = plt.figure(figsize=(16., 16.))



coords = (grid_jv*9).astype(int)


indexes = np.lexsort((coords[:, 0], -coords[:, 1]))

images = np.array(images)[indexes]

df_list = []

for i in range(len(images)):
    item = images[i]
    y, sr = librosa.load(item) # Replace with your audio file path
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    
    fig, ax = plt.subplots()
    
    S_db = librosa.power_to_db(S, ref=np.max)


    # Display the Mel spectrogram without any markers
    # The 'x_axis' and 'y_axis' arguments only control the labels and scaling, not markers.
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)

    
    ax.set_axis_off()
    
    filename = os.path.splitext(os.path.basename(item))[0]
    
    
    df_list.append({
        'filename': filename,
        'path': 'images/' + filename + '.png',
        'x': coords[i, 0],
        'y': coords[i, 1]
    })
    
    
    plt.savefig('images/' + filename + '.png', bbox_inches='tight')
    
    plt.close()


df = pd.DataFrame(df_list)

df.to_csv(os.getcwd() + '/images/muha_data.csv', index=False)









# imports
import librosa, os
from scipy.linalg import svd, toeplitz
from scipy.stats import zscore
import numpy as np
from scipy.stats import entropy
from statsmodels.tsa.stattools import acf

from tqdm.notebook import tqdm
from random import sample
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn import svm, neighbors, model_selection, manifold
from statistics import mean
from sklearn.metrics import classification_report
from skimage.measure import block_reduce

import tensorflow_hub as hub
import tensorflow as tf

import lap

from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors
from pyha_analyzer.models.demo_CNN import ResnetConfig, ResnetModel
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors
from pyha_analyzer.models import EfficentNet


import torch






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
    
    return Entropy(s), Entropy(s)*JSD(s) 

def process_data(data):
    path = data["filepath"]
    
    if (".WAV" not in path) and (".wav" not in path) and (path.contains('part')):
        print("directory found")
        return None
    
    i = 0
    audio, sr = librosa.load(path, sr=data['sample_rate'], duration=5, offset=i)

    
    if librosa.get_duration(y = audio, sr = sr) == 0:
        return None
    
    h, c = EGCI(audio)

    
    
    output_data = {
            "path": path,
            "offset_s": 0,
            "sr": sr,
            "gt": data['labels'],
            "site": data['labels'],
            "entropy": h,
            "complexity": c
        }
    return output_data


# getting all the wavs

music_extractor = extractors.Music()
music_ads = music_extractor("/home/a.jajodia.229/acoustic/local_data/muha/Liked Sounds/Location A Sand Forrest")


df = music_ads['train'].to_pandas()


sum(df['filepath'].str.contains('SERRA'))


def process_data_multiprocess(paths, process_data_func=process_data,  processes=8, sample=250):
    sample = paths.sample(sample)
    # sample = paths
    
    samples = list(sample.T.to_dict().values())
    
    
    with Pool(processes) as p:
        data = list(tqdm(p.imap(process_data_func, samples)))
    return data


data = [process_data(df.iloc[i]) for i in tqdm(range(200))]


metadata = pd.read_csv('/home/a.jajodia.229/acoustic/local_data/muha/Liked Sounds/Location A Sand Forrest/Metadat -  Sandforest.csv',delimiter=';')


metadata.rename(columns={'FileName': 'filepath'}, inplace=True)


data = pd.DataFrame(data)


data['time'] = data['path'].str.split('_').apply(lambda x: ('Night' if ((int(x[5]) > 180000) or (int(x[5]) < 60000)) else 'Day') if (len(x) == 8) else pd.NA)


sns.scatterplot(data, x = 'entropy', y = 'complexity', hue = 'time')





sr = 96000
max_length = sr*4 # ignore samples longer than 4 seconds
fixed_length = sr/4 # trim all samples to 250 milliseconds
limit = None # set this to 100 to only load the first 100 samples


def load_sample(fn, sr=None,
                max_length=None, fixed_length=None, normalize=True):
    if fn == '': # ignore empty filenames
        return None
    audio, _ = librosa.load(fn)
    duration = len(audio)
    if duration == 0: # ignore zero-length samples
        return None
    if max_length and duration >= max_length: # ignore long samples
        return None
    if fixed_length:
        np.resize(audio, fixed_length)
    max_val = np.abs(audio).max()
    if max_val == 0: # ignore completely silent sounds
        return None
    if normalize:
        audio /= max_val
    return (fn, audio, duration)


max_length = sr*4 # ignore samples longer than 4 seconds
fixed_length = sr//4 # trim all samples to 250 milliseconds
limit = None # set this to 100 to only load the first 100 samples


sample = [load_sample(filepath, max_length=max_length, fixed_length=fixed_length) for filepath in df['filepath']]

samples = [i[1] for i in sample if i is not None]


n_fft = 128
hop_length = int(n_fft/4)
use_logamp = False # boost the brightness of quiet sounds
reduce_rows = 10 # how many frequency bands to average into one
reduce_cols = 1 # how many time steps to average into one
crop_rows = 32 # limit how many frequency bands to use
crop_cols = 32 # limit how many time steps to use
limit = 100 # set this to 100 to only process 100 samples

window = np.hanning(n_fft)
def job(y):
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    amp = np.abs(S)
    # if reduce_rows > 1 or reduce_cols > 1:
    #     amp = block_reduce(amp, (reduce_rows, reduce_cols), func=np.mean)
    if amp.shape[1] < crop_cols:
        amp = np.pad(amp, ((0, 0), (0, crop_cols-amp.shape[1])), 'constant')
    amp = amp[:crop_rows, :crop_cols]
    if use_logamp:
        amp = librosa.logamplitude(amp**2)
    amp -= amp.min()
    if amp.max() > 0:
        amp /= amp.max()
    amp = np.flipud(amp) # for visualization, put low frequencies on bottom
    return amp
pool = Pool()
fingerprints = pool.map(job, samples[:limit])
fingerprints = np.asarray(fingerprints).astype(np.float32)


# get 2d embeddings
tsne = manifold.TSNE(random_state = 1, n_components=2, learning_rate=50)
data2d = tsne.fit_transform(fingerprints.reshape((100,1024)))

data2d -= data2d.min(axis=0)
data2d /= data2d.max(axis=0)


plt.figure(figsize=(4, 4))
plt.scatter(data2d[:,0], data2d[:,1], edgecolors='none', marker='o', s=12)
plt.show()

side = 10
xv, yv = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
grid = np.dstack((xv, yv)).reshape(-1, 2)


from scipy.spatial.distance import cdist
cost = cdist(grid, data2d, 'sqeuclidean')
cost = cost * (10000000. / cost.max())


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

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


data_trimmed = [i[:,:,0:1876] for i in data['audio']]



import matplotlib.pyplot as plt
import numpy as np
import torch

from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm_notebook


fig = plt.figure(figsize=(16., 16.))



coords = (grid_jv*9).astype(int)

grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(10, 10),
                 axes_pad=0,  # pad between Axes in inch.
                 )

indexes = np.lexsort((coords[:, 0], -coords[:, 1]))

images = np.array(data['path'])[indexes]

for ax, item in tqdm_notebook(zip(grid, images)):
    
    y, sr = librosa.load(item, sr = None)
    im = librosa.feature.melspectrogram(y=y, sr=sr)

    im = librosa.power_to_db(im, ref=np.max)
    
    im = im[:,0:256]
    
    ax.imshow(im)


data['x'] = [coord[0] for coord in coords]
data['y'] = [coord[1] for coord in coords]

data.to_csv('/Users/anu/Documents/e4e/ecoacoustic-discovery/images/muha_data.csv', index=False)








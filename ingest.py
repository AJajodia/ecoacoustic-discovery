
# imports
import librosa

from tqdm import tqdm

from random import sample
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn import manifold

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

import lap

from pyha_analyzer import extractors

import numpy as np

import os



def process_data(data):
    path = data["filepath"]
    
    if (".WAV" not in path) and (".wav" not in path) and (path.contains('part')):
        print("directory found")
        return None
    
    i = 0
    audio, sr = librosa.load(path, sr=data['sample_rate'], duration=5, offset=i)

    
    if librosa.get_duration(y = audio, sr = sr) == 0:
        return None
    
    # h, c = EGCI(audio)

    
    
    output_data = {
            "path": path,
            "offset_s": 0,
            "sr": sr,
            "gt": data['labels'],
            "site": data['labels'],
            # "entropy": h,
            # "complexity": c
        }
    return output_data


# getting all the wavs

music_extractor = extractors.Music()
music_ads = music_extractor("/home/a.jajodia.229/acoustic/local_data/muha/Liked Sounds/Location A Sand Forrest")


df = music_ads['train'].to_pandas()


print(df)

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

sr = 96000
max_length = sr*4 # ignore samples longer than 4 seconds
fixed_length = sr/4 # trim all samples to 250 milliseconds
limit = None # set this to 100 to only load the first 100 samples


def load_sample(fn, 
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

images = np.array(data['path'])[indexes]

df_list = []

for item, coord in zip(images, coords):
    
    plt.figure(2,2)
    
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
        'x': coord[0],
        'y': coord[1]
    })
    
    
    plt.savefig('images/' + filename + '.png', bbox_inches='tight')
    
    plt.close()


df = pd.DataFrame(df_list)

df.to_csv(os.getcwd() + '/images/muha_data.csv', index=False)








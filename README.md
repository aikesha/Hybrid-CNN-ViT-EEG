# Hybrid-CNN-ViT-EEG
PyTorch implementation of "Subject-Independent Brain-Computer Interfaces: A Comparative Study of Attention Mechanism-Driven Deep Learning Models" (IHCI 2023)

> Link to [BCI Competition IV 2a and 2b datasets](https://drive.google.com/drive/folders/1RJi3uOjL9P6g9zxMAYkEZ9SxHHx4mUND?usp=sharing)


## BCI Competition IV 2a data shape format:
```
EEG data is 3D numpy array (trials x channels x time samples) ---->
Number of subjects in BNCI2014001: 9
Subject 0 : (288, 22, 321)
Subject 1 : (288, 22, 321)
Subject 2 : (288, 22, 321)
Subject 3 : (288, 22, 321)
Subject 4 : (288, 22, 321)
Subject 5 : (288, 22, 321)
Subject 6 : (288, 22, 321)
Subject 7 : (288, 22, 321)
Subject 8 : (288, 22, 321)
Total : ( 2592 , 22 , 321 )
```


## BCI Competition IV 2b data shape format:
```
EEG data is 3D numpy array (trials x channels x time samples) ---->
Number of subjects in BNCI2014004: 9
Subject 0 : (720, 3, 321)
Subject 1 : (680, 3, 321)
Subject 2 : (720, 3, 321)
Subject 3 : (740, 3, 321)
Subject 4 : (740, 3, 321)
Subject 5 : (720, 3, 321)
Subject 6 : (720, 3, 321)
Subject 7 : (760, 3, 321)
Subject 8 : (720, 3, 321)
Total : ( 6520 , 3 , 321 )
```

## How to load the data:

1. Define the data list
```
datasets = ['aBNCI2014001R.pickle', 'aBNCI2014004R.pickle']
```
2. Define the data loading function
```
import pickle

def load_data(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

data = load_data(datasets[1])    # datasets[1] = 'aBNCI2014004R.pickle'
```
3. Learn about the data

```
subject = 0
s1 = data[subject]
s1.get_data().shape       # for bci IV 2b data subject 0 --> (720, 3, 321)
```
```
len(data)     # gives the number of subjects
print(s1)     # will show data about number of events, targets, time range, and baseline for a subject 0 as below:
'''
<Epochs |  45 events (all good), 1 - 3.5 sec, baseline off, ~4.5 MB, data loaded,
 'left_hand': 23
 'right_hand': 22>
'''
```
```
# Plot the event for a certain subject with the following command:
s1['right_hand'].plot()

# Plot the psd too, if needed:
s1['right_hand'].plot_psd(fmin=0, fmax=40)  # the freq values can be changed

# Plot the sensors location by:
s1.plot_sensors(title = 'EEG sensor locations and labels', show_names = True)
```
4. To analyze further you can take a look at [MNE library](https://mne.tools/dev/auto_tutorials/raw/index.html), they have a lot of useful info.

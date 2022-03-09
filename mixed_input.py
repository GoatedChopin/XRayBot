import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import Model
from keras import layers
from keras.applications import MobileNetV2
from NPZDataGenerator import NPZDataGenerator

def get_X_y(df, wd = "Categories\\npz"):
    # os.chdir(wd)
    ims = []
    stats = []
    conds = []
    for path in df["npz_path"]:
        entry = np.load(path)
        ims.append(convert_bw_rgb(entry["im"]))
        stats.append(entry["stats"])
        conds.append(entry["conds"])
    ims = np.asarray(ims)
    stats = np.asarray(stats)
    conds = np.asarray(conds)
    return (ims, stats), conds

def tt_split(df, train = 0.7):
    shuffled_df = df.sample(frac=1)
    nrow = round(train * len(df))
    train_df = shuffled_df.iloc[:nrow]
    test_df = shuffled_df.iloc[nrow:]
    return train_df, test_df

def stat_scaler(tensor):
    means = tf.constant([46.90146271851587, 0.4350695683196575, 0.39966107741705315])
    std_devs = tf.constant([16.839922533380808, 0.4957683241322746, 0.4898308285442431])
    return (tensor - means) / std_devs

def convert_bw_rgb(pic_array):
    return np.stack((pic_array, pic_array, pic_array), axis=2)

def build_mi_nn(pic_shape=(1024,1024,3), num_stats=3):
    # CNN
    input_pic = layers.Input(shape=pic_shape)
    # x = layers.Lambda(tf.image.grayscale_to_rgb)(input_pic)
    x = MobileNetV2(input_shape=(pic_shape[0], pic_shape[1], 3), include_top=False)(input_pic)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation='relu')(x)
    x = Model(inputs=input_pic, outputs=x)
    
    # stats
    input_stats = layers.Input(shape=(num_stats,))
    w = layers.Lambda(stat_scaler)(input_stats)
    w = layers.Dense(32, activation='relu')(w)
    w = layers.Dense(10, activation='relu')(w)
    w = Model(inputs=input_stats, outputs=w)
    
    # concat both layers
    combined = layers.concatenate([x.output, w.output])
    z = layers.Dense(14, activation='relu')(combined)
    model = Model(inputs=[x.input, w.input], outputs=z)
    
    model.compile(optimizer = "Adam", loss = tf.keras.losses.CategoricalCrossentropy(), metrics=["acc"])
    return model

def breakdown_conds(gen1, gen2):
    def get_conds(filepath):
        d = np.load(filepath)
        return d["conds"]
    def get_percs(generator):
        percs = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for d in generator.data:
            c = get_conds(d)
            index = 0
            for i in np.nditer(c):
                percs[index] += i
                index += 1
        percs = [i / len(generator.data) for i in percs]
        return percs
    return get_percs(gen1), get_percs(gen2)

if __name__ == '__main__':
    subset = pd.read_pickle("midf.pkl")
    blacklist = []
    for (dirpath, dirnames, filenames) in os.walk("Categories\\Defects"):
        for filename in filenames:
            if filename.endswith('.npz'): 
                blacklist.append(os.sep.join(["Categories\\npz", filename]))

    train, test = tt_split(subset)
    train_paths = [i for i in train["npz_path"].to_list() if i not in blacklist]
    test_paths = [i for i in test["npz_path"].to_list() if i not in blacklist]
    del subset, blacklist
    
    tiny = False
    if tiny:
        train_paths = train_paths[0:100]
        test_paths = test_paths[0:100]
    del tiny
    
    train_gen = NPZDataGenerator(train_paths, 10, (1024, 1024), 14)
    test_gen = NPZDataGenerator(test_paths, 10, (1024, 1024), 14)
    del train, test, train_paths, test_paths
    model = build_mi_nn()
    # Does the filepath require a full absolute filepath, or is a relative filepath sufficient?
    cp = tf.keras.callbacks.ModelCheckpoint(filepath = os.path.join(os.getcwd(), "Models\\ckpt_{epoch}"), save_best_only = True)
    """
    Note, there are 4 channels on a certain subset of the images, maybe 50 or so out of 6500. it might make sense just to move the troublemakers to a new folder where they can't hurt us anymore.
    import os
    import numpy as np
    for (dirpath, dirnames, filenames) in os.walk("Categories\\npz"):
        for filename in filenames:
            if filename.endswith('.npz'): 
                entry = np.load(os.sep.join([dirpath, filename]))
                im = entry["im"]
                shape = im.shape
                if len(shape) != 2:
                    # get that shit out of here
    """
    model.fit(train_gen, validation_data = test_gen, epochs = 15, callbacks = [cp])
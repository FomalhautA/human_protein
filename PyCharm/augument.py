import random
import time
import copy
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import pandas as pd

from os import path
from stats import origin_stats
from util import rand_sample, dict_substract, df_to_dict, COLORS

PERCENTAGE = 0.0357
PERCENT_TH = 0.038


def isaugment(name, full_set, perts):
    code = full_set[name]
    for i in range(len(code)):
        if i and perts[i] < PERCENTAGE:
            return True

    return False


def img_generator(name, new_name, source_folder, target_folder):
    names = [name + '_' + c + '.png' for c in COLORS]
    new_names = [new_name + '_' + c + '.png' for c in COLORS]
    images = []
    for item in names:
        im = Image.open(path.join(source_folder, item), 'r')
        images.append(im)

    flip_lr = random.random()
    flip_tb = random.random()
    rotate = random.randint(1, 360)

    rand = random.random()
    brightness = random.uniform(1., 2.5) if rand > 0.5 else random.uniform(0.5, 1.)

    rand = random.random()
    color = random.uniform(1., 2.5) if rand > 0.5 else random.uniform(0.5, 1.)

    rand = random.random()
    contrast = random.uniform(1., 2.) if rand > 0.5 else random.uniform(0.5, 1.)

    rand = random.random()
    sharpness = random.uniform(1., 5.) if rand > 0.5 else random.uniform(0.5, 1.)

    for i, im in enumerate(images):
        # Flip
        if flip_lr > 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        if flip_tb > 0.5:
            im = im.transpose(Image.FLIP_TOP_BOTTOM)

        # Rotate
        im = im.rotate(rotate)

        # Brightness
        enh_bri = ImageEnhance.Brightness(im)
        im = enh_bri.enhance(brightness)

        # Color
        enh_col = ImageEnhance.Color(im)
        im = enh_col.enhance(color)

        # Contrast
        enh_con = ImageEnhance.Contrast(im)
        im = enh_con.enhance(contrast)

        # Sharpness
        enh_sha = ImageEnhance.Sharpness(im)
        im = enh_sha.enhance(sharpness)

        im.save(path.join(target_folder, new_names[i]))


def update(namelst, cnts, name, full_set, cnt, source_folder, target_folder):
    new_name = name+'-aug_'+str(cnt)
    img_generator(name, new_name, source_folder, target_folder)
    namelst.append(new_name)
    code = full_set[name]
    for i in range(len(code)):
        if code[i]:
            cnts[i] += 1

    return


def isterminate(perts):
    for item in perts:
        if item < PERCENTAGE:
            return False

    return True


def dict_aug(dict1, dict_full):
    ans = copy.deepcopy(dict1)
    for key in dict1.keys():
        for item in dict_full.keys():
            if item.__contains__('-aug'):
                origin = item.split('-aug')[0]
                if origin == key:
                    ans[item] = dict_full[item]

    return ans


def val_aug(fname, fname_val, target_fname='../Data/val_aug.csv'):
    """

    :param fname:
    :param fname_val:
    :param target_fname:
    :return:
    """
    df = pd.read_csv(fname)
    dict_full_aug = df_to_dict(df)

    df_val = pd.read_csv(fname_val)
    f_dict_val = df_to_dict(df_val)

    f_dict_val = dict_aug(f_dict_val, dict_full_aug)

    df_val = pd.DataFrame()
    df_val = df_val.from_dict(f_dict_val)
    df_val.to_csv(target_fname, index=False)

    return


def data_partition(fname_aug, fname_ori, fname_tra, fname_val, tv_ratio=0.2, aug=True, local_aug=True):
    """

    :param fname_tra:
    :param fname_val:
    :param fname:
    :param tv_ratio: train validation ratio
    :return: dict of train set and dict of validation set
    """

    df = pd.read_csv(fname_ori)
    dict_full = df_to_dict(df)

    df = pd.read_csv(fname_aug)
    dict_full_aug = df_to_dict(df)

    dict_temp = dict()
    if aug:
        dict_temp = dict_full_aug
    else:
        dict_temp = dict_full

    total = len(dict_temp.keys())

    f_dict_full = copy.deepcopy(dict_temp)
    f_dict_val = rand_sample(dict_temp, int(total * tv_ratio))
    dict_substract(dict_temp, f_dict_val)
    f_dict_train = dict_temp

    if local_aug:
        f_dict_train = dict_aug(f_dict_train, dict_full_aug)
        f_dict_val = dict_aug(f_dict_val, dict_full_aug)

    df_val = pd.DataFrame()
    df_val = df_val.from_dict(f_dict_val)
    df_val.to_csv(fname_val, index=False)

    df_train = pd.DataFrame()
    df_train = df_train.from_dict(f_dict_train)
    df_train.to_csv(fname_tra, index=False)

    return f_dict_train, f_dict_val, f_dict_full


def main_proc(fname, source_folder='../Data/train_s', target_folder='../Data/train_s'):
    namelst_grouped, namelst, full_set = origin_stats()
    total = len(namelst)
    cnts = np.array([len(item) for item in namelst_grouped])
    print(cnts)
    cnts_ori = copy.deepcopy(cnts)

    perts = cnts/total
    muls = PERCENT_TH / perts

    idx_g = np.argsort(cnts_ori)[::-1]
    cnt = 1
    while not isterminate(perts):
        idx = np.argmax(muls)
        nameset = namelst_grouped[idx]
        aug = int((muls[idx] - 1) * 0.01 * cnts[idx])
        # print('----------------------------------')
        # print('Class {} Augmenting, mult is {}, scale is {}, aug is {}'.format(idx, round(muls[idx], 2), cnts[idx], aug))
        for i in range(aug):
            name = random.sample(nameset, 1)[0]
            update(namelst, cnts, name, full_set, cnt, source_folder, target_folder)
            cnt += 1

        total = len(namelst)
        perts = cnts/total
        muls = PERCENT_TH / perts

        # print('Sample Scale: ', len(namelst))

        time.sleep(0)

    for i, item in enumerate(perts):
        print(i, round(item*100, 2))

    print((cnts-cnts_ori)[idx_g])
    print([round(item1/item2, 2) for item1, item2 in zip(cnts[idx_g], cnts_ori[idx_g])])

    df_auged = pd.DataFrame()
    for name in namelst:
        temp = name.split('-aug')[0] if name.__contains__('-aug') else name
        df_auged[name] = full_set[temp]

    df_auged.to_csv(fname, index=False)


if __name__ == '__main__':
    # main_proc(fname='../Data/full_aug.csv', source_folder='../Data/train_s', target_folder='../Data/aug_test3')
    data_partition(fname_aug='../Data/full_aug.csv',
                   fname_ori='../Data/full.csv',
                   fname_tra='../Data/tra_aug.csv',
                   fname_val='../Data/val_aug.csv',
                   tv_ratio=0.2,
                   aug=True,
                   local_aug=False)
    # val_aug(fname='../Data/full_true_aug.csv',
    #         fname_val='../Data/val.csv')
    # img_generator(name='000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0',
    #               new_name='000a6c98-bb9b-11e8-b2b9-ac1f6b6435d0-aug2',
    #               source_folder='../Data/aug_test',
    #               target_folder='../Data/aug_test')

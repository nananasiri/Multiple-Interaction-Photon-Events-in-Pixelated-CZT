from collections import defaultdict
import vg
import math as m
import numpy as np
import random
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')


# -------------------------------------------------------------------------------------------------------------
# Target Sequence: [non_com, compton, photo] OR [non_com, photo, compton] sorted based on time (har kudum zudtar khorde)
# Features: After blurring Energy we gain our features such as theta_p, theta_e, energies ...
# -------------------------------------------------------------------------------------------------------------

def icos(a):
    if a > 1.0:
        a = 1.0
    elif a < -1.0:
        a = -1.
    inv_ = m.acos(a)
    return m.degrees(inv_)

    # -------------------------------------------------------------------------------------------------------------
    # ----------------------------- Filter families based on Energy Window ----------------------------------------
    # -------------------------------------------------------------------------------------------------------------


def process_family(family):
    dict_ = defaultdict(list)
    for i, item in enumerate(family):
        dict_[item[-8]].append(i)  # ID of particle [-8]

    for key in dict_:
        items = dict_[key]  # items = [0, 1] ke 0 ye satre kamele
        energy = 0.0
        for item in items:
            energy += float(family[item][11])
        if energy < 421 or energy > 621:  # if energy < 421 or energy > 601:
            return False
    return True

    # -------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Calculate Theta_P and Theta_E and Features for ML-------------------
    # -------------------------------------------------------------------------------------------------------------


def calculate(family):  # col10: time stamp nana
    # global non_comp, compton, pe
    # random.shuffle(family)
    return_pack = {'target_seq': [], 'pred_seq': [], 'valid_family': False}
    # -------------------------------------------------------------------------------------------------------------
    # ---------------------------------Check for family validity whether we have 1C2P combination -----------------
    # -------------------------------------------------------------------------------------------------------------

    if len(family) != 3:
        return return_pack

    counter_2 = 0
    counter_3 = 0
    for i in range(len(family)):  # count the number of rows for column -8 to be 2 or 3
        if family[i][-8] == '2':
            counter_2 += 1
        elif family[i][-8] == '3':
            counter_3 += 1
        """return empty pack if the row is neither Compton nor Photon"""
        if family[i][-3] not in ['Compton', 'PhotoElectric']:
            return return_pack

    """ Check if family has 2P1C photon IDs """
    # if not ((counter_2 == 1 and counter_3 == 2) or (counter_2 == 2 and counter_3 == 1)):
    if not (counter_2 == 1 and counter_3 == 2):
        return return_pack

    dict_ = defaultdict(list)
    for i, item in enumerate(family):
        dict_[item[-8]].append(i)

    comp_recov = None
    for k in dict_:  # key='2' ya '3'
        items = dict_[k]  # [0] ya [1,2] ya [0,1,2] ya [0] ya [2]maslan
        if len(items) == 2:  # age tu items 2 ta item darim pas comp_recov = value(hala ya 2 ya 3)
            # print(items)
            comp_recov = k  # comp_recov = '2' OR '3'
        else:  # age tu items 2 ta nadarim va faghat yek item darim pas in item awal non_comp has!:)
            non_comp = items[0]
        # print('dict_.keys():', dict_.keys())
        # print('dict_.values():', dict_.values())

    # -------------------------------------------------------------------------------------------------------------
    # We label data-set: [Non_Comp index, {compton photo OR photo compton} sorted by time! Smaller time comes first]
    # -------------------------------------------------------------------------------------------------------------
    dict_label = {'time': [], 'panel': [], 'row': []}
    for i, row in enumerate(family):
        dict_label['time'].append(row[10])
        dict_label['panel'].append(row[-8])
        dict_label['row'].append(i)

    target_seq = [non_comp]
    df = pd.DataFrame(dict_label)
    df = df.sort_values(by=['time'], ignore_index=False)

    first_panel = df['panel'][non_comp]  # bar asase time khordan. uni ke awal khorde ro var midarim bad dustesho:)
    # print(df[df['panel'] != first_panel])
    target_seq.extend(list(df[df['panel'] != first_panel]['row']))
    return_pack['target_seq'] = target_seq

    """ Blur Energy to find theta_p and theta_e after blurring not from Ground Truth"""
    # mu1, sigma1 = 0.0, 17.35881104
    # for i in range(len(family)):
    #     val = float(family[i][11])
    #     val += np.random.normal(mu1, sigma1)
    #     family[i][11] = str(val)
    """ Blur Time """

    mu, sigma = 0.0, 0.000000018
    for i in range(len(family)):
        val = float(family[i][10])
        val += np.random.normal(mu, sigma)
        family[i][10] = str(val)

    dict_pred = {'time': [], 'panel': [], 'row': []}
    for i, row in enumerate(family):
        dict_pred['time'].append(row[10])
        dict_pred['panel'].append(row[-8])
        dict_pred['row'].append(i)

    pred_seq = [non_comp]
    df = pd.DataFrame(dict_pred)
    df = df.sort_values(by=['time'], ignore_index=False)

    first_panel_pred = df['panel'][non_comp]  # bar asase time khordan. uni ke awal khorde ro var midarim bad dustesho:)
    # print(df[df['panel'] != first_panel])
    pred_seq.extend(list(df[df['panel'] != first_panel_pred]['row']))
    return_pack['pred_seq'] = pred_seq

    return_pack['valid_family'] = True
    return return_pack


def main():
    with open("50BigBinnedNoBlurred3.csv", 'r') as f:  # 3 radif data 50BigBinnedNoBlurred3
        family = []
        invalid_family_counter = 0
        correct_pred_counter = 0
        wrong_pred_counter = 0
        for line in f:
            out = line.rstrip("\r\n")
            if out == "":
                process = process_family(family)
                if process:
                    return_pack = calculate(family)
                    if return_pack['valid_family']:
                        if return_pack['target_seq'] == return_pack['pred_seq']:
                            correct_pred_counter += 1
                        else:
                            wrong_pred_counter += 1
                        print('correct: ', correct_pred_counter, 'wrong: ', wrong_pred_counter)
                    else:
                        invalid_family_counter += 1
                family = []
                continue
            else:
                out = out.strip()
                items = out.split("\t")
                if items[22] != "RayleighScattering":
                    family.append(items)
    accuracy = correct_pred_counter/(correct_pred_counter + wrong_pred_counter) * 100
    print('Accuracy: ', accuracy)


if __name__ == "__main__":
    main()

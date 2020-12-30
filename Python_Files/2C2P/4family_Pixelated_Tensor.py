from collections import defaultdict
import vg
import math as m
import numpy as np
import random
import pandas as pd

np.seterr(divide='ignore', invalid='ignore')


# -------------------------------------------------------------------------------------------------------------
# Target Sequence: [C1P1, C2P2] OR [C1P1, P2C2] OR [P1C1, C2P2] OR [P1C1, P2C2] sorted based on time
# Features: After blurring Energy, we gain our features such as theta_p, theta_e, energies ...
# # Don't forget to blur energy and shuffle family:)
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
        items = dict_[key]  # items = [0, 1] [2, 3] ke 0 ye satre kamele
        energy = 0.0
        for item in items:
            energy += float(family[item][11])
        if energy < 421 or energy > 621:  # if energy < 421 or energy > 601:
            return False
        # print(items)
    return True

    # -------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Calculate Theta_P and Theta_E and Features for ML-------------------
    # -------------------------------------------------------------------------------------------------------------


def calculate(family):  # col10: time stamp nana
    # global non_comp, compton, pe
    # random.shuffle(family)
    return_pack = {'ID_Flag': [], 'X': [], 'Y': [], 'Z': [],
                   'theta_p_1': [], 'theta_e_1': [], 'theta_p_2': [], 'theta_e_2': [],
                   'theta_p_3': [], 'theta_e_3': [], 'theta_p_4': [], 'theta_e_4': [],
                   'energy_c1': [], 'energy_p1': [], 'energy_c2': [], 'energy_p2': [],
                   'event1x': [], 'event1y': [], 'event1z': [],
                   'event2x': [], 'event2y': [], 'event2z': [],
                   'event3x': [], 'event3y': [], 'event3z': [],
                   'event4x': [], 'event4y': [], 'event4z': [],
                   'time1': [], 'time2': [], 'time3': [], 'time4': [],
                   'DDA1': [], 'DDA2': [], 'DDA3': [], 'DDA4': [],
                   'target_seq': [],
                   'rf_counter': 0, 'tf_counter': 0, 'valid_family': False}
    # -------------------------------------------------------------------------------------------------------------
    # ---------------------------------Check for family validity whether we have 1C2P combination -----------------
    # -------------------------------------------------------------------------------------------------------------
    if len(family) != 4:
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

    if counter_2 != 2 or counter_3 != 2:  # Check if family has 2P2C photon IDs
        return return_pack

    # -------------------------------------------------------------------------------------------------------------
    # ----------------------------------Check if all event ids are identical to recognize Random Coincidences -
    # -------------------------------------------------------------------------------------------------------------
    event_id = []
    for row in family:
        event_id.append(int(row[1]))

    if event_id[1:] == event_id[:-1]:
        return_pack['ID_Flag'].append(1)
        # return_pack['tf_counter'] += 1
    else:
        return_pack['ID_Flag'].append(0)
        # return_pack['rf_counter'] += 1

    """ Blur Energy to find theta_p and theta_e after blurring not from Ground Truth"""
    mu1, sigma1 = 0.0, 17.35881104
    for i in range(len(family)):
        val = float(family[i][11])
        val += np.random.normal(mu1, sigma1)
        family[i][11] = str(val)
    #     # -------------------------------------------------------------------------------------------------------------
    #     # Prepare target sequence: [C1P1, C2P2] OR [C1P1, P2C2] OR [P1C1, C2P2] OR [P1C1, P2C2] sorted by time! Smaller time comes first]
    #     # -------------------------------------------------------------------------------------------------------------

    dict_label = {'time': [], 'panel': [], 'row': []}
    for i, row in enumerate(family):
        dict_label['time'].append(row[10])
        dict_label['panel'].append(row[-8])
        dict_label['row'].append(i)

    df = pd.DataFrame(dict_label)
    df = df.sort_values(by=['time'], ignore_index=False)

    first_panel = df['panel'][0]
    target_seq = []
    target_seq.extend(list(df[df['panel'] == first_panel]['row']))
    target_seq.extend(list(df[df['panel'] != first_panel]['row']))
    return_pack['target_seq'] = target_seq
    # print(target_seq)

    # -------------------------------------------------------------------------------------------------------------
    # ---------------------------------------- Calculate Theta_P and Theta_E --------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    perms = [[target_seq[0], target_seq[1], target_seq[2], target_seq[3]],
             [target_seq[1], target_seq[0], target_seq[2], target_seq[3]],
             [target_seq[0], target_seq[1], target_seq[3], target_seq[2]],
             [target_seq[1], target_seq[0], target_seq[3], target_seq[2]]]
    # print(perms)

    i = target_seq[0]  # 0
    j = target_seq[1]  # 2
    k = target_seq[2]  # 1
    l = target_seq[3]  # 3
    a_vector_1 = [float(family[i][13]) - float(family[k][13]),
                  float(family[i][14]) - float(family[k][14]),
                  float(family[i][15]) - float(family[k][15])]
    b_vector_1 = [float(family[j][13]) - float(family[i][13]),
                  float(family[j][14]) - float(family[i][14]),
                  float(family[j][15]) - float(family[i][15])]
    a_vector_1 = np.array(a_vector_1)
    b_vector_1 = np.array(b_vector_1)
    theta_p_1 = vg.angle(a_vector_1, b_vector_1)
    theta_e_1 = icos(1. - 511. * (1 / (float(family[j][11])) - 1 / (float(family[i][11]) + float(family[j][11]))))

    a_vector_2 = [float(family[j][13]) - float(family[k][13]),
                  float(family[j][14]) - float(family[k][14]),
                  float(family[j][15]) - float(family[k][15])]
    b_vector_2 = [float(family[i][13]) - float(family[j][13]),
                  float(family[i][14]) - float(family[j][14]),
                  float(family[i][15]) - float(family[j][15])]
    a_vector_2 = np.array(a_vector_2)
    b_vector_2 = np.array(b_vector_2)
    theta_p_2 = vg.angle(a_vector_2, b_vector_2)
    theta_e_2 = icos(1. - 511. * (1 / (float(family[i][11])) - 1 / (float(family[i][11]) + float(family[j][11]))))

    a_vector_3 = [float(family[k][13]) - float(family[j][13]),
                  float(family[k][14]) - float(family[j][14]),
                  float(family[k][15]) - float(family[j][15])]
    b_vector_3 = [float(family[l][13]) - float(family[k][13]),
                  float(family[l][14]) - float(family[k][14]),
                  float(family[l][15]) - float(family[k][15])]

    a_vector_3 = np.array(a_vector_3)
    b_vector_3 = np.array(b_vector_3)
    theta_p_3 = vg.angle(a_vector_3, b_vector_3)
    theta_e_3 = icos(1. - 511. * (1 / (float(family[l][11])) - 1 / (float(family[k][11]) + float(family[l][11]))))
    a_vector_4 = [float(family[l][13]) - float(family[j][13]),
                  float(family[l][14]) - float(family[j][14]),
                  float(family[l][15]) - float(family[j][15])]
    b_vector_4 = [float(family[k][13]) - float(family[l][13]),
                  float(family[k][14]) - float(family[l][14]),
                  float(family[k][15]) - float(family[l][15])]

    a_vector_4 = np.array(a_vector_4)
    b_vector_4 = np.array(b_vector_4)
    theta_p_4 = vg.angle(a_vector_4, b_vector_4)
    theta_e_4 = icos(1. - 511. * (1 / (float(family[k][11])) - 1 / (float(family[k][11]) + float(family[l][11]))))


    return_pack['theta_p_1'].append(theta_p_1)
    return_pack['theta_e_1'].append(theta_e_1)
    return_pack['theta_p_2'].append(theta_p_2)
    return_pack['theta_e_2'].append(theta_e_2)
    return_pack['theta_p_3'].append(theta_p_3)
    return_pack['theta_e_3'].append(theta_e_3)
    return_pack['theta_p_4'].append(theta_p_4)
    return_pack['theta_e_4'].append(theta_e_4)

    if np.isnan(theta_p_1) or np.isnan(theta_p_2) or np.isnan(theta_p_3) or np.isnan(theta_p_4):
        return return_pack
    return_pack['valid_family'] = True

    return return_pack


def main():
    with open("test.csv", 'r') as f:  # 3 radif data 50BigBinnedNoBlurred4
        # with open("test1.csv", 'w') as g:  # az 3 radif + kudum Correct kudum Wrong!
        # lines = f.readlines()
        family = []
        invalid_family_counter = 0
        tf_counter = 0
        rf_counter = 0
        tensor_X = np.empty((0, 1, 8))  # I wanted to implement encoder decoder, that's why I store features in a tensor
        tensor_y = np.empty((0, 4), dtype=int)
        for line in f:
            out = line.rstrip("\r\n")
            if out == "":
                process = process_family(family)
                # print('process', process)
                if process:
                    return_pack = calculate(family)
                    # print('return_pack:', return_pack)
                    if return_pack['valid_family']:
                        for key in return_pack:
                            # these are String or integer and not convertible to floats
                            if key not in ['valid_family', 'Process_name', 'rf_counter', 'tf_counter', 'target_seq']:
                                return_pack[key] = list(map(float, return_pack[key]))  # Maps items to get their floats!
                        y = np.array(return_pack['target_seq'])
                        y = y.astype(int)
                        """
                            matrix = [return_pack['ID_Flag'], 
                                      return_pack['event1x'], return_pack['event1y'], return_pack['event1z'],
                                      return_pack['event2x'], return_pack['event2y'], return_pack['event2z'],
                                      return_pack['event3x'], return_pack['event3y'], return_pack['event3z'],
                                      return_pack['event4x'], return_pack['event4y'], return_pack['event4z'],
                                      return_pack['time1'], return_pack['time2'], 
                                      return_pack['time3'], return_pack['time4'],
                                      return_pack['energy_c1'], return_pack['energy_p1'],
                                      return_pack['energy_c2'], return_pack['energy_p2'],
                                      return_pack['theta_p_1'], return_pack['theta_e_1'],
                                      return_pack['theta_p_2'], return_pack['theta_e_2'],
                                      return_pack['theta_p_3'], return_pack['theta_e_3'],
                                      return_pack['theta_p_4'], return_pack['theta_e_4']]
                            """
                        matrix = [return_pack['theta_p_1'], return_pack['theta_e_1'],
                                  return_pack['theta_p_2'], return_pack['theta_e_2'],
                                  return_pack['theta_p_3'], return_pack['theta_e_3'],
                                  return_pack['theta_p_4'], return_pack['theta_e_4']]
                        matrix = np.array(matrix).T
                        # print(tensor_np.shape, matrix.shape)
                        tensor_X = np.append(tensor_X, [matrix], axis=0)
                        tensor_y = np.append(tensor_y, [y], axis=0)
                        # print(tensor_y)
                    else:
                        invalid_family_counter += 1
                family = []
                continue
            else:
                out = out.strip()
                items = out.split("\t")
                if items[22] != "RayleighScattering":
                    family.append(items)
    print('tensor_X:', tensor_X.shape, 'tensor_y:', tensor_y.shape)  # folder
    np.save('test_X', tensor_X)
    np.save('test_y', tensor_y)

    # print(tensor_np.shape[0])
    # out_file = open("LSTM/seq_tensor.csv", 'ab')
    # for i in range(tensor_np.shape[0]):
    #     np.savetxt(out_file, np.reshape(tensor_np[i,:], [tensor_np.shape[1], tensor_np.shape[2]]), delimiter="\t")

    with open('test_X.csv', 'w') as f:
        for matrix_m in tensor_X:
            for row in matrix_m:
                for char in row:
                    f.write(str(char))
                    f.write('\t')
                f.write('\n')
            # f.write('\n')

    with open('test_y.csv', 'w') as k:
        for row in tensor_y:
            k.write(str(row))
            k.write('\n')
        # k.write('\n')

    # out_file.close()

    # np.savetxt("LSTM/seq_tensor.csv", np.reshape(tensor_np, [tensor_np.shape[0]*tensor_np.shape[1], tensor_np.shape[2]]), delimiter="\t")


if __name__ == "__main__":
    main()

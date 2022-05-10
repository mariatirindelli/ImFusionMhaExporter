import numpy as np


def print_imfusion(input_mat):

    print_str = "["
    for row in range(4):
        print_str += "["
        for col in range(4):
            if col < 3:
                print_str += str(input_mat[row, col]) + ","
            else:
                print_str += str(input_mat[row, col]) + "]"

        if row < 3:
            print_str += ","
        else:
            print_str += "]"

    print(print_str)


T_centerToImage = np.eye(4)
T_centerToImage[0, -1] = 161.19/2
T_centerToImage[1, -1] = 107.83/2

print(T_centerToImage)

T_ptoImage = np.array([
[-0.290886,  1.002729, -0.057588, 135.027420],
 [0.578499, -0.019398, -1.631128, -64.091045] ,
[-2.269415, -0.063918, -0.408411, 94.402864 ],
[ 0.000000,  0.000000,  0.000000,  1.000000 ],
])

res = np.matmul(T_ptoImage, T_centerToImage)

print_imfusion(res)
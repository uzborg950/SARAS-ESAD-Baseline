''' Finds output sizes of different resnet architectures'''
import sys
import numpy as np
import math

#Output: channels,height,width
def resnet18(input_size):
    sizes = np.array([64,64,64,128,256,512]).reshape(6,1) #channels

    dims = input_size.split(",")
    for dim, dim_size in enumerate(dims):
        conv1 = calc_output_size(int(dim_size),7,2)
        maxpool = calc_output_size(conv1,3,2)
        conv2 = calc_output_size(maxpool,3,1)
        conv3 = calc_output_size(conv2,3,2)
        conv4 = calc_output_size(conv3, 3, 2)
        conv5 = calc_output_size(conv4, 3, 2)
        dim_output_sizes = np.array([conv1,maxpool,conv2,conv3,conv4,conv5])
        sizes = np.append(sizes, dim_output_sizes.reshape(6,1), axis=1)
    print("c","h","w")
    print(sizes)
    return sizes


def calc_output_size(input_size, kernel, stride, padding='same'):
    if padding == 'same':
        padding_size = (kernel-1)/2
    return math.floor(((input_size - kernel + (2*padding_size)) / (stride)) + 1)


def main(argv):
    #The difference is only the channels in resnet18, 34, 50 etc.
    if argv[1] == "18":
        return resnet18(argv[2]) #(h,w)



if __name__ == '__main__':
    main(sys.argv)

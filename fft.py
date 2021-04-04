import sys, re
import numpy as np
# Constant values
# naive_size = 16 # can be acheived using (1<<naive_size_pow)
naive_size_pow = 4

# default values
mode = 1
image_filename = "moonlanding.png"

#input: -m Mode -i image
arg_line = " ".join(sys.argv[1:])
mode_match = re.search("(?<=-m )[0-9]+", arg_line)
image_filename_match = re.search("(?<=-i )\S+", arg_line)

if mode_match is not None:
    mode = int(mode_match.group())

if image_filename_match is not None:
    image_filename = image_filename_match.group()

print("selected mode: "+str(mode))
print("selected file: "+image_filename)


if mode==1:
    fft_image()
elif mode==2:
    denoise()
elif mode==3:
    compress()
elif mode==4:
    plot_runtime()
else:
    print("invalid mode, please choose a number from 1 to 4.")
    exit()
exit()


# used to find padding length
def next_pow2(init_len):
    count = 0;

    # find if power of 2
    if (n and not(n & (n - 1))):
        return n

    # find number of bits set
    while( n != 0):
        n >>= 1
        count += 1

    return 1 << count;

def log2(num):
    count = 0
    while num!=0:
        num >>= 1
        count += 1
    return count

# naive
def naive_ft(vector):
    N = len(vector)
    ft_vector = np.zeros(N, dtype=np.complex64)
    for k in range(N):
        exponential = np.exp(-2j*np.pi*k/N*(np.arange(N, dtype=np.complex64)))
        for n in range(N):
            ft_vector[k]+=vector[n]*exponential[n]
    return ft_vector

def naive_ft_k(vector, ex):
    sum = 0+0j
    for n in range(N):
        sum+=vector[n]*ex[n]


# fast
def fast_ft(vector):
    N = next_pow2(len(vector)) #makes the length of the input a power of 2
    pow = log2(N) #power of the length
    if pow<=naive_size_pow:
        return naive_ft(vector)
    else:
        ft_vector = np.zeros(N, dtype=np.complex64) #output array of the fft
        pre_naive_exp = -2j*np.pi/(1<<naive_size_pow)*(np.arange((1<<naive_size_pow), dtype=np.complex64)) # e^(pre_naive_exp*k) are the exponentials used in the naive FT
        base_join_exp = np.zeros((pow-naive_size_pow), dtype=np.complex64)
        for i in range(pow-naive_size_pow):
            base_join_exp[i]=1<<i
        base_join_exp = -2j*np.pi/N*exp_vector # e^(pre_naive_exp*k) are the exponentials multiplied by the sum of the odd n values

        for k in range(N):
            join_exp = np.exp(k*base_join_exp)
            naive_exp = np.exp(k*pre_naive_exp)
            ft_vector[k] = fast_join(vector, 0, exp_vector, naive_exp)

        return ft_vector

# sum back together
def fast_join(vector, depth, join_exp, naive_exp):
    if len(vector)<=(1<<naive_size_pow):
        return naive_ft_k(vector, naive_exp)
    else:
        join_even = fast_join(vector[::2], depth+1, join_exp, naive_exp)
        join_odd = fast_join(vector[1::2], depth+1, join_exp, naive_exp)
        return join_even+join_exp[depth]*join_odd


# TODO: fft inverse

# TODO: 2d-fft

# TODO: 2d-fft inverse

# TODO: 2d log scale plot


# TODO: save dft to .txt or .csv

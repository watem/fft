import sys, re, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Constant values
# naive_size = 16 # can be acheived using (1<<naive_size_pow)
naive_size_pow = 4
timing_min_size = 5
timing_max_size = 10
timing_number_experiments = 10

# default values
mode = 1
image_filename = "moonlanding.png"


# used to find padding length
def next_pow2(init_len):
    count = 0
    n = init_len

    # find if power of 2
    if (n and not(n & (n - 1))):
        return n

    # find number of bits set
    while( n != 0):
        n >>= 1
        count += 1

    return 1 << count

def log2(num):
    if num==0:
        return np.NINF
    count = 0
    while num!=0:
        num >>= 1
        count += 1
    return count-1

# naive
def naive_ft(vector):
    N = len(vector)
    ft_vector = np.zeros(N, dtype=np.complex64)
    for k in range(N):
        exponential = np.exp(-2j*np.pi*k/N*(np.arange(N, dtype=np.complex64)))
        for n in range(N):
            ft_vector[k]+=vector[n]*exponential[n]
    return ft_vector

# sum parts back together
def fast_join(vector, depth, join_exp, naive_exp):
    if len(vector)<=(1<<naive_size_pow):
        return np.dot(vector, naive_exp[:len(vector)])
    else:
        join_even = fast_join(vector[::2], depth+1, join_exp, naive_exp)
        join_odd = fast_join(vector[1::2], depth+1, join_exp, naive_exp)
        return join_even+join_exp[depth]*join_odd

# fast
def fast_ft(vector):
    N = next_pow2(len(vector)) #makes the length of the input a power of 2
    padding = np.zeros(N-len(vector), dtype=np.complex64)
    vector = np.concatenate((vector,padding), axis=None)
    pow = log2(N) #power of the length
    if pow<=naive_size_pow:
        return naive_ft(vector)
    else:
        ft_vector = np.zeros(N, dtype=np.complex64) #output array of the fft
        base_naive_exp = -2j*np.pi/(1<<naive_size_pow)*(np.arange((1<<naive_size_pow), dtype=np.complex64)) # e^(base_naive_exp*k) are the exponentials used in the naive FT
        base_join_exp = np.zeros((pow-naive_size_pow), dtype=np.complex64)
        for i in range(pow-naive_size_pow):
            base_join_exp[i]=1<<i
        base_join_exp = -2j*np.pi/N*base_join_exp # e^(base_join_exp*k) are the exponentials multiplied by the sum of the odd n values

        for k in range(N):
            join_exp = np.exp(k*base_join_exp)
            naive_exp = np.exp(k*base_naive_exp)
            ft_vector[k] = fast_join(vector, 0, join_exp, naive_exp)

        return ft_vector

# fft inverse
def inverse_fast_ft(vector):
    inverse = fast_ft(vector)/len(vector)
    reverse = np.concatenate((inverse[0],inverse[:0:-1]),axis=None)
    return reverse

# 2d-naive-ft
def naive_ft_2d(a):
    n = a.shape[0] # rows
    m = a.shape[1] # columns

    ft_clmns = np.zeros((m,n), dtype=np.complex64)
    ft_rows = np.zeros((n,m), dtype=np.complex64)

    # Take transpose to compute fft on columns
    T = np.transpose(a)

    for i in range(m):
        ft_clmns[i] = naive_ft(T[i])

    for j in range(n):
        ft_rows[j] = naive_ft(np.transpose(ft_clmns)[j])

    return ft_rows

# 2d-fft
def fft_2d(a):
    n = a.shape[0] # rows
    m = a.shape[1] # columns

    ft_clmns = np.zeros((m,n), dtype=np.complex64)
    ft_rows = np.zeros((n,m), dtype=np.complex64)

    # Take transpose to compute fft on columns
    T = np.transpose(a)

    for i in range(m):
        ft_clmns[i] = fast_ft(T[i])

    for j in range(n):
        ft_rows[j] = fast_ft(np.transpose(ft_clmns)[j])

    return ft_rows

# 2d-fft inverse
def ifft_2d(a):
    n = a.shape[0] # rows
    m = a.shape[1] # columns

    ft_clmns = np.zeros((m,n), dtype=np.complex64)
    ft_rows = np.zeros((n,m), dtype=np.complex64)

    # Take transpose to compute fft on columns
    T = np.transpose(a)

    for i in range(m):
        ft_clmns[i] = inverse_fast_ft(T[i])

    for j in range(n):
        ft_rows[j] = inverse_fast_ft(np.transpose(ft_clmns)[j])

    return ft_rows

# TODO: 2d log scale plot
def plot(fft_image):
    plt.imshow(np.abs(fft_image), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.figure()
    plot_spectrum(fft_image)
    plt.title('Fourier transform')

# TODO: save dft to .txt or .csv

# TODO: fft of image
def fft_image(im):
    fft_2d(im)

# TODO: denoise image
def denoise(im): 
    return
# TODO: compress image
def compress(im):
    return
#TODO: plot runtime
def timing_fast(image):
    t1 = time.time()
    fft_2d(image)
    t2 = time.time()
    return t2-t1

def timing_naive(image):
    t1 = time.time()
    naive_ft_2d(image)
    t2 = time.time()
    return t2-t1

def single_timing_experiment(size):
    dimensions = 2<<size
    image = np.random.random((dimensions,dimensions)) #create a random nXn 2d matrix
    time_fast = timing_fast(image)
    time_naive = timing_naive(image)
    return np.array([time_fast, time_naive])


def single_size_timing(size):
    timings = np.zeros((timing_number_experiments,2))
    for i in range(timing_number_experiments):
        timings[i] = single_timing_experiment(size)

    timing_details = np.zeros((2,2))
    timing_details[0] = timings.mean(axis=0)
    timing_details[1] = timings.std(axis=0)
    return timing_details

def print_timing_details(algorithm, details):
    for size in range(timing_max_size+1-timing_min_size):
        side_size = 1<<(size+timing_min_size)
        txt = "length of square: {size:=} \tmean time taken(s): {mean:.3e} \tstd deviation(s): {std:.3e}"
        print(txt.format(size=side_size, mean=details[algorithm][0][size], std=details[algorithm][1][size]))

def plot_runtime():
    all_timings = np.zeros(((timing_max_size+1-timing_min_size),2,2))
    problem_sizes = np.arange(timing_min_size,timing_max_size+1)
    for size in range(timing_max_size+1-timing_min_size):
        all_timings[size] = single_size_timing(size+timing_min_size)
        problem_sizes[size] = ((1<<(size+timing_min_size))*(1<<(size+timing_min_size)))

    details = all_timings.transpose() # details[algorithm][info][size] where info=0 is mean, and info=1 is std
    print("naive Fourier times")
    print_timing_details(1, details)
    print("fast Fourier times")
    print_timing_details(0, details)

    algo = details[0]
    plt.errorbar(problem_sizes, algo[0],yerr=2*algo[1], barsabove=True, label="fast")
    algo = details[1]
    plt.errorbar(problem_sizes, algo[0],yerr=2*algo[1], barsabove=True, label="naive")

    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel("problem size")
    plt.ylabel("time taken (s) (95% confidence)")
    plt.title("Runtime of algorithms vs Time taken")
    plt.show()


#input: -m Mode -i image
arg_line = " ".join(sys.argv[1:])
mode_match = re.search("(?<=-m )[0-9]+", arg_line)
image_filename_match = re.search("(?<=-i )\S+", arg_line)

if mode_match is not None:
    mode = int(mode_match.group())

if image_filename_match is not None:
    image_filename = image_filename_match.group()

img = mpimg.imread(image_filename)

print("selected mode: " + str(mode))
print("selected file: " + image_filename)

if mode==1:
    fft_image(img)
elif mode==2:
    denoise(img)
elif mode==3:
    compress(img)
elif mode==4:
    plot_runtime()
else:
    print("invalid mode, please choose a number from 1 to 4.")
    exit()
exit()
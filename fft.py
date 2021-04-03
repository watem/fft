import sys, re
import numpy as np
# Constant values
naive_size = 10

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

# TODO: naive
def naive_ft(vector):
    N = len(vector)
    ft_vector = np.zeros(N, dtype=np.complex64)
    for k in range(N):
        exponential = np.exp(-2j*np.pi*k/N*(np.arange(N, dtype=np.complex64)))
        for n in range(N):
            ft_vector[k]+=vector[n]*exponential[n]
    return ft_vector

# TODO: fast


# TODO: fft inverse

# TODO: 2d-fft

# TODO: 2d-fft inverse

# TODO: 2d log scale plot


# TODO: save dft to .txt or .csv

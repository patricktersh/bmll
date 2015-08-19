import pylab as pl

def ff_pl(x, a, b):
    return a * pow(x, b)


def ff_lg(x, a, b):
    return a * pl.log10(1.+b*x)

def is_in(list_1, list_2):
    return map(lambda x: x in list_2, list_1)


def extract_ym(str_in):
    # returns from a string in format YYYY-MM-DD hh:mm:ss the corresponding year and month, i.e. YYYY-MM
    return str_in[0:7]


def extract_min(str_in):
    # returns from a string in format YYYY-MM-DD hh:mm:ss the corresponding minute of the day,
    # where "2007-01-01 09:30:00" corresponds to 1
    h = int(str_in[11:13])
    m = int(str_in[14:16])
    return h*60+m-569


def extract_min_short(str_in):
    # returns from a string in format hh:mm:ss the corresponding minute of the day,
    # where "09:30:00" corresponds to 1
    h = int(str_in[0:2])
    m = int(str_in[3:5])
    return h*60+m-569


def is_greater(list_1, comp):
    map()


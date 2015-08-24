import pylab as pl
import scipy.optimize as optimize

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

def generate_starting_point(i_in, mat_start_point, n_par, n_step):
    tmp = []
    cnt = n_par
    while cnt > 0:
        n_col = n_par - cnt
        n_row = i_in%n_step
        i_in = i_in/n_step
        cnt = cnt -1
        tmp  = tmp + [mat_start_point[n_row,n_col]]
    return tmp




def fit_nonlin_1d_2p(ff_in, df_in, a_r, b_r):
    par = []
    vv = []
    chi = []
    for a_s in pl.linspace(a_r[0], a_r[1], 10):
        for b_s in pl.linspace(b_r[0], b_r[1], 10):
            p0 = [a_s, b_s] # Initial guess for the parameters
            fit_in = optimize.curve_fit(ff_in, df_in.pi, df_in.imp, p0, pl.array(df_in.stdd)/pl.array(pl.sqrt(df_in.nn)))
            par.append(fit_in[0])
            vv.append(fit_in[1])
            imp_pred = ff_in(df_in.pi,fit_in[0][0],fit_in[0][1])
            chi.append(sum(pow(imp_pred-df_in.imp, 2)))

    ind_min = chi.index(min(chi))
    chi_opt = chi[ind_min]
    par_opt = par[ind_min]
    vv_opt = vv[ind_min]
    return par_opt, vv_opt, chi_opt


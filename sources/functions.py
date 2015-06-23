
def extract_day( str_in ):
    return str_in[0:10]

def extract_min( str_in ):
    h = int(str_in[11:13])
    m = int(str_in[14:16])
    return h*60+m-569

def find_pos(  list_1, list_2 ):
    return map(lambda x:list_2.index(x), list_1)



def fit_nonlin_1d_2p(ff_in, df_in, a_r, b_r):
    par = []
    vv = []
    chi = []
    for a_s in py.linspace(a_r[0],a_r[1],10):
        for b_s in py.linspace(b_r[0],b_r[1],10):
            p0 = [a_s, b_s] # Initial guess for the parameters
            fit_in = optimize.curve_fit(ff_in,df_in.pi,df_in.imp,p0,py.array(df_in.stdd)/py.array(py.sqrt(df_in.nn)))
            par.append(fit_in[0])
            vv.append(fit_in[1])
            imp_pred = ff_in(df_in.pi,fit_in[0][0],fit_in[0][1])
            chi.append( sum(pow(imp_pred-df_in.imp,2)) )

    ind_min = chi.index(min(chi))
    chi_opt = chi[ind_min]
    par_opt = par[ind_min]
    vv_opt = vv[ind_min]
    return par_opt, vv_opt, chi_opt

parse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
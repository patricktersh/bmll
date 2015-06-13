def fit_nonlin_1d_2p(ff, df, a_r, b_r):
    par = []
    vv = []
    chi = []
    for a_s in py.linspace(a_r[0],a_r[1],10):
        for b_s in py.linspace(b_r[0],b_r[1],10):
            p0 = [a_s, b_s] # Initial guess for the parameters
            fit_in = optimize.curve_fit(ff,df.pi,df.imp,p0,py.array(df.stdd)/py.array(py.sqrt(df.nn)))
            par.append(fit_in[0])
            vv.append(fit_in[1])
            imp_pred = ff(df.pi,fit_in[0][0],fit_in[0][1])
            chi.append( sum(pow(imp_pred-df.imp,2)) )
            
    ind_min = chi.index(min(chi))
    chi_opt = chi[ind_min]
    par_opt = par[ind_min]
    vv_opt = vv[ind_min]
    return par_opt, vv_opt, chi_opt

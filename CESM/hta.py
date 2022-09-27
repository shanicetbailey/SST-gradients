def horizontal_turner_angle(ds, da_SST, da_SSS):
    """Compute horizontal turner angle given SST and SSS, 
        and user-defined basin selection"""
    import pop_tools
    import fastjmd95
    import xarray as xr
    import numpy as np
    
    grid, ds_ren = pop_tools.to_xgcm_grid_dataset(ds)

    
    def gradient(da):
        """Compute the gradients of T, S and rho"""
        
        da_diffx = grid.diff(da, 'X')
        da_diffy = grid.diff(da, 'Y', boundary='fill')
        da_diffx_interp = grid.interp(da_diffx, 'X')
        da_diffy_interp = grid.interp(da_diffy, 'Y', boundary='fill')
        dadx = da_diffx_interp/ds_ren.DXT
        dady = da_diffy_interp / ds_ren.DYT
        return dadx, dady
    
    #create grad-rho and |grad-rho| terms
    rho = fastjmd95.rho(ds_ren[f'{da_SSS}'], ds_ren[f'{da_SST}'], 0)
    gradrho = gradient(rho)
    modgradrho = 1 / np.sqrt(gradrho[0]**2 + gradrho[-1]**2)
    
    #create gradT and gradS
    gradT = gradient(ds_ren[f'{da_SST}'])
    gradS = gradient(ds_ren[f'{da_SSS}'])
    
    #define alpha and beta
    runit2mass = 1.035e3 #rho_0
    drhodt = fastjmd95.drhodt(ds_ren[f'{da_SSS}'], ds_ren[f'{da_SST}'], 0)
    drhods = fastjmd95.drhods(ds_ren[f'{da_SSS}'], ds_ren[f'{da_SST}'], 0)
    alpha = - drhodt / runit2mass
    beta = drhods / runit2mass
    
    #define eq
    #gradrho[0/1] is selecting the dadx/dady output from `gradient` func above
    term1 = modgradrho*(gradrho[0]*(alpha*gradT[0] + beta*gradS[0]) + gradrho[1]*(alpha*gradT[1] + beta*gradS[1]))
    term2 = modgradrho*(gradrho[0]*(alpha*gradT[0] - beta*gradS[0]) + gradrho[1]*(alpha*gradT[1] - beta*gradS[1]))
    turner_radians = np.arctan2(term1, term2)
    turner_angle = np.rad2deg(turner_radians)
    
    return (turner_angle)
from pygifsicle import optimize
import imageio
import numpy as np
import torch
from torch.utils import tensorboard
import wandb


from sample_factory.utils.utils import AttrDict
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
import numpy as np
import matplotlib.pyplot as plt

from openTSNE import TSNE


def save_simulation_gif(cfg, all_img):
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = f'{cfg.train_dir}/{cfg.experiment}/sim_{t_stamp}.gif'

    wrt = imageio.get_writer(pth, mode='I',fps=35)

    with wrt as writer:
        for i in range(all_img.shape[3]):
            writer.append_data(all_img[:,:,:,i])

    optimize(pth)

    # displaying in WandB
    if cfg.with_wandb:
        wandb.init(name=cfg.experiment, project='sample_factory', group='rfs')
        wandb.log({"video": wandb.Video(pth, fps=35, format="gif")})

def save_receptive_fields_plot(cfg,device,enc,lay,env):
    
    obs = env.reset() # this is the first observation when agent is spawned
    obs_torch = AttrDict(transform_dict_observations(obs))
    for key, x in obs_torch.items():
        obs_torch[key] = torch.from_numpy(x).to(device).float()
    
    if cfg.encoder_custom=='greyscale_lindsey':
        obs = obs_torch['obs'][:,0,None,:,:] # setting shape of observation to be only single channel
    else:
        obs = obs_torch['obs']

    isz = list(obs.size())[1:]
    outmtx = enc.conv_head[0:(lay*2)](obs)
    osz = list(outmtx.size())[1:]

    nchns = isz[0]
    flts = osz[0]

    if cfg.retinal_stride == 2 and cfg.kernel_size == 9: # lindsay with stride and kernel 9
        diams = [1, 9, 17, 50]
        rds = diams[lay]//2 + 2*lay# padding a bit
    else: # classic lindsay case
        rds = 2 + (8*(lay-1) + enc.kernel_size) // 2

    rwsmlt = 2 if flts > 8 else 1 # rows in rf subplot
    fltsdv = flts//rwsmlt

    fig, axs = plt.subplots(nchns*rwsmlt,fltsdv,dpi = 100,figsize=(6*fltsdv, 4*rwsmlt*nchns))

    for i in range(fltsdv):

        for j in range(rwsmlt):

            flt = i + j*fltsdv
            avg = spike_triggered_average(device,enc,lay,flt,rds,isz)

            vmx = abs(avg).max()

            for k in range(nchns):

                # Plotting statistics
                rw = k + j*nchns

                if flts > 1 and nchns*rwsmlt > 1:
                    ax = axs[rw,i]
                elif flts <= 1 and nchns*rwsmlt > 1:
                    ax = axs[rw]
                elif flts > 1 and nchns*rwsmlt <= 1:
                    ax = axs[i]

                ax.set_axis_off()
                pnl = ax.imshow(avg[k,:,:],vmin=-vmx,vmax=vmx)
                cbar = fig.colorbar(pnl, ax=ax)
                cbar.ax.tick_params(labelsize=12*rwsmlt)

                if k == 0:
                    ax.set_title("Filter: " + str(flt), { 'weight' : 'bold' }, fontsize=12*rwsmlt)


    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')

    # displaying in tensorboard
    pth_tb = cfg.train_dir +  "/" + cfg.experiment + '/.summary/0/'
    writer = tensorboard.SummaryWriter(pth_tb)
    writer.add_figure(f"rf-conv{lay}", fig, close=False) # only show the latest rfs in tb (for comparison across models)
    writer.close()

    # displaying in WandB
    if cfg.with_wandb:
        wandb.init(name=cfg.experiment, project='sample_factory', group='rfs')
        wandb.log({f"rf-conv{lay}": fig})

    # saving
    pth = cfg.train_dir +  "/" + cfg.experiment + f"/rf-conv{lay}_" + t_stamp + ".png"
    plt.savefig(pth)

def plot_all_rf(cfg, actor_critic, env):
    enc = actor_critic.encoder.base_encoder
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda') # this line should go above the get_acts_actor function

    # always plot receptive fields
    n_conv_lay = len(enc.conv_head)//2
    for lay in range(1, n_conv_lay+1):
        save_receptive_fields_plot(cfg,device,enc,lay,env)

def spike_triggered_average(dev,enc,lay,flt,rds,isz):

    with torch.no_grad():

        btchsz = [10000] + isz
        cnty = (1+btchsz[2])//2
        cntx = (1+btchsz[3])//2
        mny = cnty - rds
        mxy = cnty + rds
        mnx = cntx - rds
        mxx = cntx + rds
        obsns = torch.randn(size=btchsz,device=dev)
        outmtx = enc.conv_head[0:(lay*2)](obsns) #forward pass
        outsz = outmtx.size()
        outs = outmtx[:,flt,outsz[2]//2,outsz[3]//2].cpu()
        obsns1 = obsns[:,:,mny:mxy,mnx:mxx].cpu()
        avg = np.average(obsns1,axis=0,weights=outs)

    return avg

def fit_tsne_1d(data):
    print('fitting 1d-tSNE...')
    # default openTSNE params
    tsne = TSNE(
        n_components=1,
        perplexity=30,
        initialization="pca",
        metric="euclidean",
        n_jobs=8,
        random_state=3,
    )

    tsne_emb = tsne.fit(data)
    return tsne_emb

def fit_tsne(data):
    print('fitting tSNE...')
    # default openTSNE params
    tsne = TSNE(
        perplexity=30,
        initialization="pca",
        metric="euclidean",
        n_jobs=8,
        random_state=3,
    )

    tsne_emb = tsne.fit(data.T)
    return tsne_emb

def plot_acts_tsne_stim(cfg, acts, health, title):
       
    # zscore
    data=row_zscore(acts)
    # get tSNE sorting and resort data
    embedding = fit_tsne_1d(data)
    temp = np.argsort(embedding[:,0])
    data = data[temp,:]
    
    # get stimulus collection times
    pos_col = np.where(np.sign(get_stim_coll(health)) == 1)
    neg_col = np.where(np.sign(get_stim_coll(health)) == -1)
    
    # plot
    fig = plt.figure(figsize=(10,3), dpi = 400)
    plt.imshow(data, cmap='bwr', interpolation='nearest', aspect='auto', vmin=-4, vmax=4)
    plt.vlines(pos_col, 0, data.shape[0], color='grey', linewidth=0.3, linestyle='--')
    plt.vlines(neg_col, 0, data.shape[0], color='black', linewidth=0.3, linestyle=':')
    plt.xlabel('Time (stamps)')
    plt.ylabel(f'{title} unit id.')
    plt.title(f'Activations of {title} neurons')
    
    # displaying in WandB
    if cfg.with_wandb:
        wandb.init(name=cfg.experiment, project='sample_factory', group='rfs')
        wandb.log({f"acts_tsne_{title}_sim": fig})

    # saving
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = cfg.train_dir +  "/" + cfg.experiment + f"/acts_tsne_{title}_sim_" + t_stamp + ".png"
    plt.savefig(pth)

def get_stim_coll(all_health, health_dep=-8, death_dep=30):
    
    stim_coll = np.diff(all_health)
    stim_coll[stim_coll == health_dep] = 0 # excluding 'hunger' decrease
    stim_coll[stim_coll > death_dep] = 0 # excluding decrease due to death
    stim_coll[stim_coll < -death_dep] = 0
    return stim_coll

# to plot library
def row_zscore(mat):
    return (mat - np.mean(mat,1)[:,np.newaxis])/(np.std(mat,1)[:,np.newaxis]+1e-8)

def plot_dimred(cfg, embedding, c, title='embedding'):
    fig = plt.figure(figsize=(20,20))
    plt.scatter(embedding[:,0],embedding[:,1], s=5, c=c, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    
    # saving
    t_stamp =  str(np.datetime64('now')).replace('-','').replace('T','_').replace(':', '')
    pth = cfg.train_dir +  "/" + cfg.experiment + f"/acts_tsne_FC_ds_" + t_stamp + ".png"
    plt.savefig(pth)

def plot_dimred_ds_acts(cfg, all_fc_act, all_lab):
    tsne_emb = fit_tsne(all_fc_act)
    plot_dimred(cfg, tsne_emb, all_lab)
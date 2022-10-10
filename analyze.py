import sys

from retinal_rl.environment import register_retinal_environment
from retinal_rl.encoders import register_encoders
from retinal_rl.parameters import custom_parse_args

from retinal_rl.activations import get_env_ac, simulate, load_sim_out, get_acts_dataset
from retinal_rl.visualization import save_simulation_gif, plot_all_rf, plot_acts_tsne_stim, plot_dimred_ds_acts

def analyze(cfg):
    env, actor_critic = get_env_ac(cfg)
    simulate(cfg, env, actor_critic) # this saves

    # load simulated data
    sim_out = load_sim_out(cfg)

    # visualise
    plot_all_rf(cfg, actor_critic, env) # receptive fields
    save_simulation_gif(cfg, sim_out['all_img'])

    plot_acts_tsne_stim(cfg, sim_out['all_fc_act'], sim_out['all_health'], title='FC')
    plot_acts_tsne_stim(cfg, sim_out['all_rnn_act'], sim_out['all_health'], title='RNN')

    if cfg.analyze_acts == 'dataset':
        ds_out = get_acts_dataset(cfg, actor_critic)
        plot_dimred_ds_acts(cfg, ds_out['all_fc_act'], ds_out['all_lab'])

def main():
    """Script entry point."""
    register_retinal_environment()
    register_encoders()
    cfg = custom_parse_args(evaluation=True)
    analyze(cfg)


if __name__ == '__main__':
    sys.exit(main())

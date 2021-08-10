import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), 'icenet'))
import config
import utils
import re
import warnings
from utils import compute_heatmap, arr_to_ice_edge_rgba_arr
from sklearn import calibration
import numpy as np
import xarray as xr
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

### User input
####################################################################

dataloader_ID = '2021_06_15_1854_icenet_nature_communications'
architecture_ID = 'unet_tempscale'

plot_tempscaled_ensemble = True

dataloader_ID_pretrain_ablation = '2021_06_30_0954_icenet_pretrain_ablation'
icenet_ID_pretrain_ablation = 'unet_tempscale'

### Matplotlib default params
################################################################################

# Global figure properties
fontsize = 7
legend_fontsize = 6
linewidth = 0.3
markersize = 3
markeredgewidth = 0.5
dpi = 300

params = {
    'axes.labelsize': fontsize,
    'axes.titlesize': fontsize,
    'font.size': fontsize,
    'figure.titlesize': fontsize,
    'legend.fontsize': legend_fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'axes.linewidth': linewidth,
    'savefig.dpi': dpi,
    'font.family': 'sans-serif',
    'figure.facecolor': 'w',
}

mpl.rcParams.update(params)

getattr(mpl.cm, 'Blues_r').set_bad(color='gray')

fig_folder = os.path.join(config.figure_folder, 'paper_figures')
fig_folder_png = os.path.join(fig_folder, 'png')
fig_folder_pdf = os.path.join(fig_folder, 'pdf')
fig_folder_table = os.path.join(fig_folder, 'tables')
for folder in [fig_folder_png, fig_folder_pdf, fig_folder_table]:
    if not os.path.exists(folder):
        os.makedirs(folder)

### Load results dataset
####################################################################

dataloader_config_fpath = os.path.join(config.dataloader_config_folder, dataloader_ID+'.json')
dataloader = utils.IceNetDataLoader(dataloader_config_fpath)

icenet_folder = os.path.join(config.networks_folder, dataloader_ID, architecture_ID)

### Forecast results
results_df_fnames = sorted([f for f in os.listdir(config.forecast_results_folder)
                            if re.compile('.*.csv').match(f)])
if len(results_df_fnames) >= 1:
    results_df_fname = results_df_fnames[-1]
    results_df_fpath = os.path.join(config.forecast_results_folder, results_df_fname)
    print('\n\nLoading most recent results dataset from {}'.format(results_df_fpath))

# Do not interpret 'NA' as NaN
results_df = pd.read_csv(results_df_fpath, keep_default_na=False, comment='#')
print('Done.\n')

### Probability analysis results
ice_edge_region_df_fpath = os.path.join(config.uncertainty_results_folder, 'ice_edge_region_results.csv')
uncertainty_df_fpath = os.path.join(config.uncertainty_results_folder, 'uncertainty_results.csv')
sip_bounding_df_fpath = os.path.join(config.uncertainty_results_folder, 'sip_bounding_results.csv')
sip_prime_90_fpath = os.path.join(icenet_folder, 'sip_prime_90.npy')

print('Loading uncertainty analysis results... ', end='', flush=True)
ice_edge_region_df = pd.read_csv(ice_edge_region_df_fpath, comment='#')
uncertainty_df = pd.read_csv(uncertainty_df_fpath, comment='#')
sip_bounding_df = pd.read_csv(sip_bounding_df_fpath, comment='#')
sip_prime_90 = np.load(sip_prime_90_fpath)
print('Done.\n')

sip_bounding_df = sip_bounding_df.set_index('Model')

uncertainty_df = uncertainty_df.drop('Unnamed: 0', axis=1, errors='ignore')
uncertainty_df['Forecast date'] = pd.to_datetime(uncertainty_df['Forecast date'])
uncertainty_df = uncertainty_df.set_index(['Model', 'Forecast date']).sort_index()

ice_edge_region_df = ice_edge_region_df.drop('Unnamed: 0', axis=1, errors='ignore')
ice_edge_region_df['Forecast date'] = pd.to_datetime(ice_edge_region_df['Forecast date'])
ice_edge_region_df = ice_edge_region_df.set_index(['Leadtime', 'Forecast date'])

### Permute-and-predict results
pap_results_df_fpath = os.path.join(config.permute_and_predict_results_folder,
                                    'permute_and_predict_results.csv')
pap_results_df = pd.read_csv(pap_results_df_fpath, comment='#')
pap_results_df['Forecast date'] = pd.to_datetime(pap_results_df['Forecast date'])

### SIE errors from the SIO
sie_errors_df = pd.read_csv(os.path.join(config.data_folder, 'sea_ice_outlook_errors.csv'), comment='#')
sie_errors_df['Model'] = 'SIO'
sie_errors_df = sie_errors_df.set_index(['year', 'init_month', 'Model'])

### Preproc results dataset
####################################################################

# Drop spurious index column if present
results_df = results_df.drop('Unnamed: 0', axis=1, errors='ignore')

results_df['Forecast date'] = pd.to_datetime(results_df['Forecast date'])

month_names = np.array(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'])
forecast_month_names = month_names[results_df['Forecast date'].dt.month.values - 1]
results_df['Calendar month'] = forecast_month_names

# Format for storing different IceNet results in one dataframe
icenet_ID = 'IceNet__{}__{}'.format(dataloader_ID, architecture_ID)
icenet_ID_pretrain_ablation = 'IceNet__{}__{}'.format(
    dataloader_ID_pretrain_ablation, icenet_ID_pretrain_ablation)

results_df.loc[results_df.Model == icenet_ID, 'Model'] = 'IceNet'
results_df.loc[results_df.Model == icenet_ID_pretrain_ablation, 'Model'] = 'IceNet-noCMIP'

results_df = results_df.set_index(['Model', 'Ensemble member', 'Leadtime', 'Forecast date'])

heldout_start = '2012-01-01'
heldout_end = '2020-09-01'

val_start = '2012-01-01'
val_end = '2017-12-01'

test_start = '2018-01-01'
test_end = '2020-09-01'

results_df = results_df.loc(axis=0)[pd.IndexSlice[:, :, :, slice(heldout_start, heldout_end)]]

results_df = results_df.sort_index()

### Load maps
####################################################################

### IceNet forecasts
if plot_tempscaled_ensemble:
    fname = 'icenet_sip_forecasts_tempscaled.nc'
    icenet_ensemble = 'ensemble_tempscaled'
else:
    fname = 'icenet_sip_forecasts.nc'
    icenet_ensemble = 'ensemble'
fpath = os.path.join(
    config.forecast_data_folder, 'icenet', dataloader_ID, architecture_ID, fname
)
icenet_ensemble_mean_sip_da = xr.open_dataarray(fpath)

### Ground truth SIC
true_sic_fpath = os.path.join(config.obs_data_folder, 'siconca_EASE.nc')
true_sic_da = xr.open_dataarray(true_sic_fpath)

### Masks
land_mask = np.load(os.path.join(config.mask_data_folder, 'land_mask.npy'))
region_mask = np.load(os.path.join(config.mask_data_folder, 'region_mask.npy'))

### Figure 2
####################################################################

if True:

    print('Plotting Figure 2... ', end='', flush=True)

    mask = np.load(os.path.join(config.mask_data_folder,
                                config.active_grid_cell_file_format.format('09')))

    min_0 = np.min(np.argwhere(mask)[:, 0])
    max_0 = np.max(np.argwhere(mask)[:, 0])
    mid_0 = np.mean((min_0, max_0)).astype(int)

    min_1 = np.min(np.argwhere(mask)[:, 1])
    max_1 = np.max(np.argwhere(mask)[:, 1])
    mid_1 = np.mean((min_1, max_1)).astype(int)

    max_diff = np.max([mid_0-min_0, mid_1-min_1])
    max_diff *= .85  # Zoom in
    max_diff = int(max_diff)
    top = mid_0 - max_diff + 10
    bot = mid_0 + max_diff + 10
    left = mid_1 - max_diff
    right = mid_1 + max_diff

    extreme_leadtimes = [4, 3, 2, 1]

    sns.color_palette('tab10')

    years = [2012, 2013, 2020]

    pred_ice_edge_rgb = sns.color_palette('tab10')[2]
    true_ice_edge_rgb = [0, 0, 0]

    fig, axes = plt.subplots(nrows=len(years), ncols=len(extreme_leadtimes),
                             figsize=(6, len(years)/len(extreme_leadtimes)*6))

    for year_i, year in enumerate(years):

        groundtruth_sic = true_sic_da.sel(time=f'{year}-09-01')

        for leadtime_i, leadtime in enumerate(extreme_leadtimes):
            icenet_sip = icenet_ensemble_mean_sip_da.sel(time=pd.Timestamp(year,9,1), leadtime=leadtime).data
            ax = axes[year_i, leadtime_i]

            # ax.imshow(icenet_sip[top:bot, left:right], cmap='Blues_r')
            ax.contourf(land_mask[top:bot, left:right], levels=[0.5, 1], colors=[mpl.cm.gray(123)])

            pred_ice_edge_rgba_arr = arr_to_ice_edge_rgba_arr(
                icenet_sip, .5, land_mask, region_mask, pred_ice_edge_rgb)
            ax.imshow(pred_ice_edge_rgba_arr[top:bot, left:right, :])
            # pred_ice_edge_rgba_arr[top:bot, left:right, :]

            ice_edge_rgba_arr = arr_to_ice_edge_rgba_arr(
                groundtruth_sic, .15, land_mask, region_mask, true_ice_edge_rgb)
            ax.imshow(ice_edge_rgba_arr[top:bot, left:right, :])

            sie_err = results_df.loc['IceNet', icenet_ensemble, leadtime, pd.Timestamp(year,9,1)]['SIE error']
            acc = results_df.loc['IceNet', icenet_ensemble, leadtime, pd.Timestamp(year,9,1)]['Binary accuracy']
            t = ax.text(s='Binary acc: {:.1f}%'.format(acc),
                        x=.99, y=.09, fontsize=5, transform=ax.transAxes, horizontalalignment='right')
            t.set_bbox(dict(facecolor='white', alpha=.8, edgecolor='white', pad=0.))
            t = ax.text(s='SIE error: {:+.3f} mil km$^2$'.format(sie_err/1e6),
                        x=0.99, y=0.02, fontsize=5, transform=ax.transAxes, horizontalalignment='right')
            t.set_bbox(dict(facecolor='white', alpha=.8, edgecolor='white', pad=0.))

            if year_i == 0:
                if leadtime == 1:
                    ax.set_title('Leadtime = {} month'.format(leadtime))
                else:
                    ax.set_title('Leadtime = {} months'.format(leadtime))
            if leadtime_i == 0:
                ax.set_ylabel('Sept {} forecasts'.format(year))

            if year_i == 0 and leadtime_i == 0:
                proxy = [plt.Line2D([0], [1], color=true_ice_edge_rgb),
                         plt.Line2D([0], [1], color=pred_ice_edge_rgb)]
                ax.legend(proxy, ['Observed ice edge', 'Predicted ice edge'],
                          loc='upper left', fontsize=5)

    for i, ax in enumerate(axes.ravel()):
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.tight_layout()

    axis_labels = 'abcdefghijkl'
    for i, ax in enumerate(axes.ravel()):
        ax.text(s=axis_labels[i], fontweight='bold', fontsize=6, x=.02, y=.02,
                transform=ax.transAxes)

    plt.savefig(os.path.join(fig_folder_png, 'fig2.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'fig2.pdf'))
    plt.close()

    print('Done.')

### Figure 3
####################################################################

if True:
    print('Plotting Figure 3... ', end='', flush=True)

    metric = 'Binary accuracy'

    df_1 = compute_heatmap(results_df, 'IceNet', icenet_ensemble, metric)
    df_2 = compute_heatmap(results_df, 'SEAS5', 'NA', metric)
    df_3 = compute_heatmap(results_df, 'Linear trend', 'NA', metric)

    min = np.min((df_1 - df_2).values)
    max = np.max((df_1 - df_2).values)
    max = np.max(np.abs((min, max)))
    max1 = np.max(np.abs((min, max)))

    min = np.min((df_1 - df_3).values)
    max = np.max((df_1 - df_3).values)
    max2 = np.max(np.abs((min, max)))

    max = np.max([max1, max2])

    cbar_max = max

    fig = plt.figure(figsize=(3.5, 7.5))

    gs = fig.add_gridspec(nrows=4, ncols=1, hspace=0.3, wspace=0.5)

    axes = []

    cbar_kws = dict(extend='both', fraction=0.05, aspect=8)

    title_str = 'binary accuracy (%)'
    df1_cbar_label = 'Binary accuracy (%)'

    cmap1 = 'inferno'
    cmap2 = 'PRGn'
    fmt = '+.1f'
    df1_fmt = '.1f'
    # cbar_kws['label'] = 'Difference (%)'
    cbar_kws['label'] = 'Binary accuracy difference (%)'
    cbar_kws['format'] = FuncFormatter(lambda x, p: format(x, '+.0f'))

    ax = fig.add_subplot(gs[0])
    axes.append(ax)

    line_kws = dict(
        palette={
            'IceNet': sns.color_palette('tab10')[0],
            'SEAS5': sns.color_palette('tab10')[3],
            'Linear trend': 'gray'},
    )
    model_order = ['IceNet', 'SEAS5', 'Linear trend']
    dashes = ['', '', (4, 4)]

    # Select the benchmarks and chosen IceNet ensemble by slicing
    #   'Ensemble member' index
    sns.lineplot(
        data=results_df.loc[pd.IndexSlice[:, [icenet_ensemble, 'NA']], :],
        x='Leadtime',
        y=metric,
        hue='Model',
        hue_order=['IceNet', 'SEAS5'],
        style='Model',
        style_order=['IceNet', 'SEAS5'],
        markers=True,
        ci=None,
        ax=ax,
        legend=False,
        **line_kws)

    sns.lineplot(
        data=results_df.loc[pd.IndexSlice[:, [icenet_ensemble, 'NA']], :],
        x='Leadtime',
        y=metric,
        hue='Model',
        hue_order=['Linear trend'],
        style='Model',
        style_order=['Linear trend'],
        ci=None,
        legend=False,
        ax=ax,
        **line_kws)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cax.patch.set_visible(False)
    cax.spines['top'].set_visible(False)
    cax.spines['bottom'].set_visible(False)
    cax.spines['right'].set_visible(False)
    cax.spines['left'].set_visible(False)
    cax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.legend(model_order, loc='best')

    ax.set_xlabel('')
    ax.set_xlim([.5, 6.5])
    ax.set_title('a', x=-.1, loc='left', fontweight='bold')
    ax.set_title('Model performance comparison')

    ax = fig.add_subplot(gs[1])
    axes.append(ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    sns.heatmap(data=df_1,
                annot=True,
                annot_kws=dict(alpha=.7),
                fmt=df1_fmt,
                cbar_kws=dict(label=df1_cbar_label, extend='both', fraction=0.05, aspect=8),
                cbar_ax=cax,
                cmap=cmap1,
                ax=ax)
    ax.set_title('b', x=-.1, loc='left', fontweight='bold')
    ax.set_title('IceNet binary accuracy')
    ax.set_xlabel('')
    ax.set_yticks(np.arange(.5, 12))
    ax.set_yticklabels(month_names)
    ax.tick_params(axis='y', rotation=0)

    ax = fig.add_subplot(gs[2])
    axes.append(ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    sns.heatmap(data=df_1 - df_2,
                cmap=cmap2,
                center=0.,
                annot=True,
                annot_kws=dict(alpha=.7),
                fmt=fmt,
                cbar_kws=cbar_kws,
                cbar_ax=cax,
                ax=ax,
                vmin=-cbar_max,
                vmax=cbar_max)
    ax.set_title('c', x=-.1, loc='left', fontweight='bold')
    ax.set_title('Improvement over SEAS5')
    ax.set_xlabel('')
    ax.set_yticks(np.arange(.5, 12))
    ax.set_yticklabels(month_names)
    ax.tick_params(axis='y', rotation=0)

    ax = fig.add_subplot(gs[3])
    axes.append(ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    sns.heatmap(data=df_1 - df_3,
                cmap=cmap2,
                center=0.,
                annot=True,
                annot_kws=dict(alpha=.7),
                fmt=fmt,
                cbar_kws=cbar_kws,
                cbar_ax=cax,
                ax=ax,
                vmin=-cbar_max,
                vmax=cbar_max)
    ax.set_title('d', x=-.1, loc='left', fontweight='bold')
    ax.set_title('Improvement over linear trend')
    ax.set_yticks(np.arange(.5, 12))
    ax.set_yticklabels(month_names)
    ax.set_xlabel('Lead time (months)')
    ax.tick_params(axis='y', rotation=0)

    plt.yticks(rotation=0)

    plt.subplots_adjust(left=.15, top=1-.05, bottom=0.05, right=1-.15)

    plt.savefig(os.path.join(fig_folder_png, 'fig3.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'fig3.pdf'))
    plt.close()

    print('Done.')

### Figure 4
####################################################################

if True:
    print('Plotting Figure 4... ', end='', flush=True)

    extreme_improvement_df = pd.DataFrame(columns=['year', 'Lead time (months)', 'Model', 'improvement'])

    for year in np.arange(2012, 2021):
        icenet_acc = results_df.loc[pd.IndexSlice['IceNet', icenet_ensemble, :, '{}-09-01'.format(year)], 'Binary accuracy']
        seas5_acc = results_df.loc[pd.IndexSlice['SEAS5', 'NA', :, '{}-09-01'.format(year)], 'Binary accuracy']
        lintrend_acc = results_df.loc[pd.IndexSlice['Linear trend', 'NA', :, '{}-09-01'.format(year)], 'Binary accuracy']
        icenet_acc.values - lintrend_acc.values
        for i, leadtime in enumerate(np.arange(1, 6+1)):
            if leadtime in [2, 3, 4]:
                extreme_improvement_df.loc[len(extreme_improvement_df)] = year, leadtime, 'SEAS5', (icenet_acc.values - seas5_acc.values)[i]
                extreme_improvement_df.loc[len(extreme_improvement_df)] = year, leadtime, 'Linear trend', (icenet_acc.values - lintrend_acc.values)[i]

    fontsize=8

    with plt.rc_context({'font.size': fontsize, 'axes.labelsize': fontsize, 'ytick.labelsize': fontsize, 'xtick.labelsize': fontsize}):
        g = sns.catplot(
            data=extreme_improvement_df,
            x='Lead time (months)',
            y='improvement',
            kind='bar',
            col='year',
            order=[4, 3, 2],
            edgecolor='k',
            lw=.5,
            height=2,
            legend=False,
            col_wrap=3,
            aspect=1,
            hue='Model',
            palette={
                'SEAS5': sns.color_palette('tab10')[3],
                'IceNet': sns.color_palette('tab10')[0],
                'Linear trend': 'gray',
            },
        )
        g.add_legend(fontsize=fontsize)
        g.set_ylabels("IceNet's binary\naccuracy improvement (%)")
        g.set_titles(template='Sept {col_name}')

        for ax in g.axes.ravel():
            ax.axhline(0, color='k', linewidth=.5)
            max = np.max(np.abs(g.axes.ravel()[0].get_ylim()))

        axis_labels = 'abcdefghi'
        for i, ax in enumerate(g.axes.ravel()):
            ax.set_title(axis_labels[i], fontsize=fontsize, fontweight='bold', loc='left', x=-.0)

        plt.savefig(os.path.join(fig_folder_png, 'fig4.png'), bbox_inches='tight')
        plt.savefig(os.path.join(fig_folder_pdf, 'fig4.pdf'), bbox_inches='tight')
        plt.close()

    print('Done.')

### Figure 5
####################################################################

if True:
    print('Plotting Figure 5... ', end='', flush=True)

    metric = 'Binary accuracy'

    # Ensemble mean with CMIP
    ensemble_mean_cmip_df = compute_heatmap(results_df, 'IceNet', 'ensemble', metric)

    # Ensemble mean no CMIP
    ensemble_mean_nocmip_df = compute_heatmap(results_df, 'IceNet-noCMIP', 'ensemble', metric)

    # Mean over individual members with CMIP
    icenet_seeds = list(set(results_df.loc['IceNet'].index.get_level_values(0)))
    icenet_seeds.remove('ensemble')
    if 'ensemble_tempscaled' in icenet_seeds:
        icenet_seeds.remove('ensemble_tempscaled')
    ensemble_dfs = []
    for seed in icenet_seeds:
        ensemble_dfs.append(
            compute_heatmap(results_df, 'IceNet', seed, metric)
        )
    mean_member_cmip_df = pd.concat(ensemble_dfs).groupby('Calendar month').\
        mean().reindex(month_names)

    # Mean over individual members no CMIP
    ensemble_dfs_nocmip = []
    for seed in icenet_seeds:
        ensemble_dfs_nocmip.append(
            compute_heatmap(results_df, 'IceNet-noCMIP', seed, metric)
        )
    mean_member_nocmip_df = pd.concat(ensemble_dfs_nocmip).groupby('Calendar month').\
        mean().reindex(month_names)

    min = np.min((ensemble_mean_cmip_df - mean_member_cmip_df).values)
    max = np.max((ensemble_mean_cmip_df - mean_member_cmip_df).values)
    max_ensemble = np.max(np.abs((min, max)))


    min = np.min((ensemble_mean_cmip_df - ensemble_mean_nocmip_df).values)
    max = np.max((ensemble_mean_cmip_df - ensemble_mean_nocmip_df).values)
    max_cmip = np.max(np.abs((min, max)))

    max = np.max([max_ensemble, max_cmip])

    #### Figure

    if True:

        # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(6, 2))

        fig = plt.figure(figsize=(6, 2))

        gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0.3, wspace=0.05)

        # plt.subplots_adjust(left=0.08, right=0.8)
        # cbar_ax = fig.add_axes([0.85, 0.1, 0.025, 0.8])

        # cbar_kws = dict(label='Difference (%)', extend='both')

        cbar_kws = dict(label='Difference (%)', extend='both', fraction=0.05, aspect=8)
        cbar_kws['format'] = FuncFormatter(lambda x, p: format(x, '+.1f'))
        # cbar_kws['cax'] = cbar_ax

        cmap = 'PRGn'

        # Fig. a)
        # ax = axes[0]
        ax = fig.add_subplot(gs[0])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cax.patch.set_visible(False)
        cax.spines['top'].set_visible(False)
        cax.spines['bottom'].set_visible(False)
        cax.spines['right'].set_visible(False)
        cax.spines['left'].set_visible(False)
        cax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        cbar_max = max

        sns.heatmap(
            data=ensemble_mean_cmip_df - ensemble_mean_nocmip_df ,
            cmap=cmap,
            center=0.,
            annot=True,
            annot_kws=dict(alpha=.8, fontsize=6.5),
            fmt='+.1f',
            cbar=False,
            # cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            ax=ax,
            vmin=-cbar_max,
            vmax=cbar_max)
        ax.set_title('a', x=-.05, loc='left', fontweight='bold')
        # ax.set_title('Ensemble-mean binary\naccuracy boost from\nCMIP6 pre-training')
        ax.set_title('Pre-training')

        ax.set_ylabel('Calendar month')
        # ax.set_xlabel('')
        ax.set_xlabel('Lead time (months)')

        # ax.set_xticks([])

        plt.yticks(rotation=0)

        # help(fig.colorbar)
        # cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, **cbar_kws)
        # cb.outline.set_linewidth(0.)
        # cb.outline.set_visible(False)

        ax.tick_params(axis='y', rotation=0)

        # Fig. b)
        # ax = axes[1]
        ax = fig.add_subplot(gs[1])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cax.patch.set_visible(False)
        cax.spines['top'].set_visible(False)
        cax.spines['bottom'].set_visible(False)
        cax.spines['right'].set_visible(False)
        cax.spines['left'].set_visible(False)
        cax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # cbar_kws=dict(label='Difference (%)', extend='both', fraction=0.05, aspect=8),
        # cbar_kws = dict(label='Difference (%)', extend='both', fraction=0.05, aspect=8)
        cbar_kws = dict(label='Binary accuracy change (%)', extend='both', fraction=0.05, aspect=8)
        cbar_kws['format'] = FuncFormatter(lambda x, p: format(x, '+.1f'))

        cbar_max = max

        sns.heatmap(
            data=ensemble_mean_cmip_df - mean_member_cmip_df,
            cmap=cmap,
            center=0.,
            annot=True,
            annot_kws=dict(alpha=.8, fontsize=6.5),
            # fmt='.1f',
            fmt='+.1f',
            cbar=False,
            # cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            ax=ax,
            vmin=-cbar_max,
            vmax=cbar_max)
        ax.set_title('b', x=-.05, loc='left', fontweight='bold')
        # ax.set_title('Binary accuracy boost from\nensembling of CMIP6\npre-trained models')
        ax.set_title('Ensembling')

        # ax.set_ylabel('')
        # ax.set_xlabel('')
        ax.set_xlabel('Lead time (months)')
        # ax.set_ylabel('Calendar month')
        ax.set_ylabel('')
        ax.set_yticks([])

        # ax.set_xticks([])

        # ax.set_ylabel('')
        # ax.set_yticks([])
        # ax.tick_params(axis='x', bottom=False, labelbottom=False)

        # Fig. c)
        # ax = axes[2]
        ax = fig.add_subplot(gs[2])
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', size='5%', pad=0.05)

        sns.heatmap(
            data=ensemble_mean_cmip_df - mean_member_nocmip_df,
            cmap=cmap,
            center=0.,
            annot=True,
            annot_kws=dict(alpha=.8, fontsize=6.5),
            # fmt='.1f',
            fmt='+.1f',
            # cbar=False,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            ax=ax,
            vmin=-cbar_max,
            vmax=cbar_max)
        ax.set_title('c', x=-.05, loc='left', fontweight='bold')
        # ax.set_title('Binary accuracy boost from\nCMIP6 pre-training\nfollowed by ensembling')
        ax.set_title('Pre-training and ensembling')
        # ax.set_title('Binary accuracy boost from ensembling\nof CMIP6 pre-trained models.')

        ax.set_xlabel('Lead time (months)')
        # ax.set_ylabel('Calendar month')
        ax.set_ylabel('')
        ax.set_yticks([])

        # ax.tick_params(axis='y', rotation=0)
        plt.subplots_adjust(left=.09, right=1-.09, top=1-.2, bottom=0.2)

        # plt.tight_layout()
        # plt.subplots_adjust(top=0.9)
        plt.savefig(os.path.join(fig_folder_png, 'fig5.png'))
        plt.savefig(os.path.join(fig_folder_pdf, 'fig5.pdf'))
        plt.close()

    print('Done.')

### Figure 6
####################################################################

if True:
    print('Plotting Figure 6... ', end='', flush=True)

    # Only plot calibration curve over test dates
    uncertainty_test_df = \
        uncertainty_df.loc[pd.IndexSlice[:, slice(test_start, test_end)], :]

    fig, ax = plt.subplots(figsize=(2, 2))
    true = uncertainty_test_df.loc['IceNet']['Ground truth'] >= .15
    prob_icenet = uncertainty_test_df.loc['IceNet'].SIP.values
    prob_icenet[prob_icenet > 1.] = 1.
    prob_seas5 = uncertainty_test_df.loc['SEAS5'].SIP.values

    prob_true_icenet, prob_pred_icenet = calibration.calibration_curve(true, prob_icenet, n_bins=20)
    ax.plot(prob_pred_icenet, prob_true_icenet, 'b--+', markersize=4, linewidth=0.5, label='IceNet')

    prob_true_seas5, prob_pred_seas5 = calibration.calibration_curve(true, prob_seas5, n_bins=20)
    ax.plot(prob_pred_seas5, prob_true_seas5, 'r--+', markersize=4, linewidth=0.5, label='SEAS5')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.5)
    ax.set_xlabel('Predicted SIP')
    ax.set_ylabel('Observed ice frequency')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder_png, 'fig6.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'fig6.pdf'))
    plt.close()

    print('Done.')

### Figure 7
####################################################################

if True:
    print('Plotting Figure 7... ', end='', flush=True)

    fig2_dict = {}

    # Compute binarry errors
    err_da = xr.ufuncs.logical_and(true_sic_da < .15, icenet_ensemble_mean_sip_da > .5) | \
        xr.ufuncs.logical_and(true_sic_da > .15, icenet_ensemble_mean_sip_da < .5)

    # Nested dict of dicts for the figure. Key denotes subplot column index. Value
    #   dicts contain the forecast start month and the leadtime.
    fig2_dict[0] = {'fc_month': pd.Timestamp(2020, 7, 1), 'leadtime': 1}
    fig2_dict[1] = {'fc_month': pd.Timestamp(2020, 8, 1), 'leadtime': 1}
    fig2_dict[2] = {'fc_month': pd.Timestamp(2020, 9, 1), 'leadtime': 1}

    row_list = ('SIP', 'Binary entropy contours')

    nrows = len(row_list)
    ncols = len(fig2_dict)

    # ice_edge_rgba = mpl.cm.YlGn(100)
    # ice_edge_rgba = mpl.cm.bone(200)
    ice_edge_rgba = mpl.cm.binary(255)

    err_cmap = mpl.cm.Oranges
    err_color_num = 170

    figsize = (ncols * 2, nrows * 2)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    gs = gridspec.GridSpec(nrows, ncols)
    gs.update(wspace=0.05, hspace=0.05)  # set the spacing between axes.
    axes = []

    for col_i, col_dict in fig2_dict.items():

        fc_month = col_dict['fc_month']
        leadtime = col_dict['leadtime']

        month_str = '{:02d}'.format(fc_month.month)

        # Determine bounding box based on active grid cell mask (crop out
        #   inactive areas)
        mask = np.load(os.path.join(config.mask_data_folder, config.active_grid_cell_file_format.format(month_str)))
        min_0 = np.min(np.argwhere(mask)[:, 0])
        max_0 = np.max(np.argwhere(mask)[:, 0])
        mid_0 = np.mean((min_0, max_0)).astype(int)

        min_1 = np.min(np.argwhere(mask)[:, 1])
        max_1 = np.max(np.argwhere(mask)[:, 1])
        mid_1 = np.mean((min_1, max_1)).astype(int)

        max_diff = np.max([mid_0-min_0, mid_1-min_1])
        max_diff *= 1.1  # Slightly increase scale for legend space
        max_diff = int(max_diff)
        top = mid_0 - max_diff
        bot = mid_0 + max_diff
        left = mid_1 - max_diff
        right = mid_1 + max_diff

        for row_i, row_plot in enumerate(row_list):

            if col_i == 0:
                if leadtime == 1:
                    ylabel_end = 'month'
                else:
                    ylabel_end = 'months'

            ax = plt.subplot(gs.new_subplotspec((row_i, col_i)))
            axes.append(ax)

            # For colorbar space
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if col_i < ncols - 1:
                # No colorbar - make extra axis invisible but leave the space
                cax.patch.set_visible(False)
                cax.spines['top'].set_visible(False)
                cax.spines['bottom'].set_visible(False)
                cax.spines['right'].set_visible(False)
                cax.spines['left'].set_visible(False)
                cax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            if row_i == 0:
                ax.set_title('Forecast month: {} {:04d}\nLeadtime = {} {}'.
                             format(month_names[fc_month.month-1], fc_month.year, leadtime, ylabel_end))

            if row_plot == 'SIP':

                cmap = getattr(mpl.cm, 'Blues_r')
                clim = (0, 1)

                sip_arr = icenet_ensemble_mean_sip_da.sel(time=fc_month, leadtime=leadtime).data
                sip_arr[land_mask] = np.nan
                sip_arr = sip_arr[top:bot, left:right]

                im_sip = ax.imshow(sip_arr, cmap=cmap, clim=clim)

                ax.contour(sip_arr, linestyles='dashed', levels=[.5], colors='white', linewidths=1.)

                ground_truth_arr = true_sic_da.sel(time=fc_month).data
                ground_truth_arr[land_mask] = np.nan
                ground_truth_arr = ground_truth_arr[top:bot, left:right]
                ax.contour(ground_truth_arr, linestyles='dashed', levels=[0.15], colors=[ice_edge_rgba], linewidths=1.)

                # Overlay binary errors
                clim = (-1.5, 1.5)
                arr_err = err_da.sel(time=fc_month, leadtime=leadtime).data[top:bot, left:right]
                arr_err[np.isnan(arr_err)] = 0.
                arr_err_rgba = np.zeros((*arr_err.shape, 4))  # RGBA array
                color_rgba = np.array(err_cmap(err_color_num)).reshape(1, 1, 4)
                idx_arrs = np.where(np.abs(arr_err) == 1)
                arr_err_rgba[idx_arrs[0], idx_arrs[1], :] = color_rgba
                idx_arrs = np.where(np.abs(arr_err) == 0)
                arr_err_rgba[idx_arrs[0], idx_arrs[1], 3] = 0  # Alpha = 0 where no error is made
                im_err = ax.imshow(arr_err_rgba)

                if col_i == 0:
                    proxy = [plt.Rectangle((0,0),1,1,fc=ice_edge_rgba),
                             plt.Rectangle((0,0),1,1,fc='white')]
                    legend1 = ax.legend(proxy, ['Observed ice edge',
                                                'Predicted ice edge'],
                                        loc='lower left', fontsize=5)

                    # Add the legend manually to the current Axes so that another legend can be added
                    ax.add_artist(legend1)

                    proxy = [plt.Rectangle((0,0),1,1,fc=err_cmap(err_color_num))]

                    ax.legend(proxy, ['Ice edge error region'],
                              loc='upper left', fontsize=5)

                # Land mask
                # ax.contourf(land_mask[top:bot, left:right], levels=[0.5, 1], colors=[mpl.cm.gray(123)])

                if col_i == ncols - 1:
                    cbar = plt.colorbar(im_sip, cax)
                    # cbar.set_label('Sea ice\nprobability', rotation=0, labelpad=21)
                    cbar.set_label('SIP', rotation=0, labelpad=18)

            elif row_plot == 'Binary entropy contours':
                sip = icenet_ensemble_mean_sip_da.sel(time=fc_month, leadtime=leadtime)[top:bot, left:right].data

                ice_edge_region_rgba = mpl.cm.Greens(125)
                confident_ice_rgba = mpl.cm.Blues_r(255)
                confident_sea_rgba = mpl.cm.Blues_r(0)

                cmap = mpl.colors.ListedColormap([confident_sea_rgba, ice_edge_region_rgba, confident_ice_rgba])
                norm = mpl.colors.BoundaryNorm([0, sip_prime_90, 1-sip_prime_90, 1+1e-6], cmap.N)
                im_sip = ax.imshow(sip, cmap=cmap, norm=norm)

                # Ground truth ice edge
                ground_truth_arr = true_sic_da.sel(time=fc_month).data[top:bot, left:right]
                ax.contour(ground_truth_arr, levels=[0.15], colors=[ice_edge_rgba], linewidths=1.)

                # Land mask
                ax.contourf(land_mask[top:bot, left:right], levels=[0.5, 1], colors=[mpl.cm.gray(123)])

                if col_i == ncols - 1:
                    # No colorbar
                    cax.patch.set_visible(False)
                    cax.spines['top'].set_visible(False)
                    cax.spines['bottom'].set_visible(False)
                    cax.spines['right'].set_visible(False)
                    cax.spines['left'].set_visible(False)
                    cax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

                if col_i == ncols - 1:
                    region_list = ['Confident open-\nwater region', 'Ice edge\nregion', 'Confident ice\nregion']
                    formatter = mpl.ticker.FuncFormatter(lambda val, loc: region_list[loc])
                    plt.colorbar(im_sip, cax, ticks=[sip_prime_90/2, 0.5, 1-sip_prime_90/2], format=formatter)

                if col_i == 0:
                    proxy = [plt.Rectangle((0,0),1,1,fc=ice_edge_rgba)]
                    ax.legend(proxy, ['Observed ice edge'], loc='lower left',
                              fontsize=5)

    axes = np.array(axes).reshape(ncols, nrows).T
    axis_labels = 'abcdefghijklmnopqrstuvwxyz'
    for i, ax in enumerate(axes.ravel()):
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        t = ax.text(s=axis_labels[i], fontweight='bold',
                    x=.99, y=.02, transform=ax.transAxes, horizontalalignment='right')
        t.set_bbox(dict(facecolor='white', alpha=.5, edgecolor='none', pad=0.5))

    plt.subplots_adjust(left=0.05, right=0.85)

    plt.savefig(os.path.join(fig_folder_png, 'fig7.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'fig7.pdf'))
    plt.close()

    print('Done.')

### Supplementary Figure 1
####################################################################

if True:
    print('Plotting Supplementary Figure 1... ', end='', flush=True)

    metric = 'Binary accuracy'

    icenet_acc = results_df.loc[pd.IndexSlice['IceNet', icenet_ensemble]].\
        reset_index().set_index(['Calendar month', 'Leadtime'])[metric]
    seas5_acc = results_df.loc[pd.IndexSlice['SEAS5', 'NA']].\
        reset_index().set_index(['Calendar month', 'Leadtime'])[metric]
    lin_acc = results_df.loc[pd.IndexSlice['Linear trend', 'NA']].\
        reset_index().set_index(['Calendar month', 'Leadtime'])[metric]

    icenet_better_than_seas5 = (icenet_acc - seas5_acc) >= 0.
    icenet_better_than_seas5 = \
        icenet_better_than_seas5.groupby(['Calendar month', 'Leadtime']).sum().\
        reset_index().pivot('Calendar month', 'Leadtime', metric).\
        reindex(month_names).astype(int)

    icenet_better_than_lin = (icenet_acc - lin_acc) >= 0.
    icenet_better_than_lin = \
        icenet_better_than_lin.groupby(['Calendar month', 'Leadtime']).sum().\
        reset_index().pivot('Calendar month', 'Leadtime', metric).\
        reindex(month_names).astype(int)

    cbar_kws = dict(fraction=0.1, aspect=16)

    bounds = np.arange(0, 11, 1)

    cbar_kws['label'] = "Number of years IceNet's binary accuracy was greater"
    cbar_kws['format'] = FuncFormatter(lambda x, p: format(x, '.0f'))
    cbar_kws['ticks']=bounds+.499

    fmt = '.0f'

    n_bounds = bounds.size
    cmap = mpl.cm.get_cmap('RdBu')
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'foo', cmap(np.linspace(0, 1, n_bounds)), len(bounds))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(4, 3))

    ax = axes[0]
    sns.heatmap(data=icenet_better_than_seas5,
                cmap=cmap,
                annot=True,
                fmt=fmt,
                cbar=False,
                cbar_kws=cbar_kws,
                norm=norm,
                ax=ax,
                vmin=0,
                vmax=9)
    ax.set_title('Improvement over SEAS5')
    ax.set_ylabel('')
    ax.set_xlabel('Lead time (months)')
    ax.tick_params(axis='y', rotation=0)

    ax = axes[1]
    sns.heatmap(data=icenet_better_than_lin,
                cmap=cmap,
                annot=True,
                fmt=fmt,
                cbar=True,
                norm=norm,
                cbar_kws=cbar_kws,
                ax=ax,
                vmin=0,
                vmax=9)
    ax.set_title('Improvement over linear trend')
    ax.set_ylabel('')
    ax.set_xlabel('Lead time (months)')
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()

    plt.savefig(os.path.join(fig_folder_png, 'supp_fig1.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'supp_fig1.pdf'))
    plt.close()

    print('Done.')

### Supplementary Figure 2
####################################################################

if True:
    print('Plotting Supplementary Figure 2... ', end='', flush=True)

    sio_months = ['June', 'July', 'Aug']

    for year in set(sie_errors_df.reset_index().year):
        for month, leadtime in zip(sio_months, [4, 3, 2]):

            sie_errors_df.loc[year, month, 'IceNet'] = results_df.\
                loc['IceNet', icenet_ensemble, leadtime, pd.Timestamp(year, 9, 1)]['SIE error'] / 1e6

            sie_errors_df.loc[year, month, 'SEAS5'] = results_df.\
                loc['SEAS5', 'NA', leadtime, pd.Timestamp(year, 9, 1)]['SIE error'] / 1e6

    sie_errors_df = sie_errors_df.sort_index().reset_index()
    sie_errors_df['abs_sie_err'] = sie_errors_df['sie_err'].abs()

    ################ SIO MAE with x=init month
    std_abs_err_df = sie_errors_df.groupby(['init_month', 'Model']).std()['abs_sie_err'].to_frame()
    mean_abs_err_df = sie_errors_df.groupby(['init_month', 'Model']).mean()['abs_sie_err'].to_frame()

    std_abs_err_df = std_abs_err_df.reset_index().\
        pivot(index='init_month', columns='Model', values='abs_sie_err')
    mean_abs_err_df = mean_abs_err_df.reset_index().\
        pivot(index='init_month', columns='Model', values='abs_sie_err')

    mean_abs_err_df = mean_abs_err_df.reindex(['June', 'July', 'Aug'])[['SEAS5', 'SIO', 'IceNet']]
    std_abs_err_df = std_abs_err_df.reindex(['June', 'July', 'Aug'])[['SEAS5', 'SIO', 'IceNet']]

    palette = {'SEAS5': sns.color_palette('tab10')[3],
               'SIO': sns.color_palette('tab10')[1],
               'IceNet': sns.color_palette('tab10')[0]}

    with mpl.rc_context({"lines.linewidth": 1}):
        g = sns.catplot(
            data=sie_errors_df,
            x='init_month',
            order=['June', 'July', 'Aug'],
            kind='bar',
            y='abs_sie_err',
            ci=False,
            height=2,
            hue_order=['SEAS5', 'SIO', 'IceNet'],
            palette=palette,
            aspect=2,
            hue='Model',
            legend=False,
            edgecolor='k',
            lw=.5,
        )
    plt.legend(bbox_to_anchor=(1.05, .75), loc='upper left')

    # Plot std dev
    mean_abs_err_df.plot(kind='bar', yerr=std_abs_err_df.values / 2,
                         ax=g.ax, width=.8, legend=False, visible=False)

    g.set_xticklabels([4, 3, 2])
    plt.xticks(rotation=0)
    g.set_axis_labels('Lead time (months)', 'September SIE MAE\n(million km$^2$)')
    plt.title('a', fontweight='bold', loc='left', x=-.1)
    plt.tight_layout()

    plt.savefig(os.path.join(fig_folder_png, 'supp_fig2a.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'supp_fig2a.pdf'))
    plt.close()

    ################ SIO comparison x=init_month, 9x9 for each year
    g = sns.catplot(
        data=sie_errors_df,
        x='init_month',
        order=['June', 'July', 'Aug'],
        kind='bar',
        col='year',
        col_wrap=3,
        y='sie_err',
        height=1.5,
        hue_order=['SEAS5', 'SIO', 'IceNet'],
        palette={'SIO': sns.color_palette('tab10')[1],
                 'SEAS5': sns.color_palette('tab10')[3],
                 'IceNet': sns.color_palette('tab10')[0]},
        aspect=1.33,
        hue='Model',
        edgecolor='k',
        lw=.5,
    )
    for ax in g.axes.ravel():
        ax.axhline(0, color='k', linewidth=.5)
        max = np.max(np.abs(g.axes.ravel()[0].get_ylim()))
        ax.set_ylim(-max, max)

    sns.color_palette('tab10')

    g.set_titles(col_template="{col_name}")
    g.set_axis_labels('Lead time (months)', 'September SIE error\n(million km$^2$)')
    g.set_xticklabels([4, 3, 2])
    axis_labels = 'bcdefghij'
    for i, ax in enumerate(g.axes.ravel()):
        ax.set_title(axis_labels[i], fontweight='bold', loc='left', x=-.0)
    plt.subplots_adjust(left=0.15, bottom=0.11, right=0.85)
    plt.savefig(os.path.join(fig_folder_png, 'supp_fig2b.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'supp_fig2b.pdf'))
    plt.close()

    plt.close()

    print('Done.')

### Supplementary Figure 3
####################################################################

if True:
    print('Plotting Supplementary Figure 3... ', end='', flush=True)

    sip_primes = sip_bounding_df.loc['IceNet']["p'"].values

    fig = plt.figure(figsize=(6, 4))

    gs_outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 1])

    gs1 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=3, subplot_spec=gs_outer[0], wspace=0.7, hspace=0.5)

    gs2 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_outer[1], wspace=0.4)

    gs1_axes = []
    for i in [0, 1, 2]:
        # with sns.axes_style('white'):
        ax = fig.add_subplot(gs1[i])
        gs1_axes.append(ax)

    gs2_axes = []
    for i in [0, 1]:
        # with sns.axes_style('ticks'):
        ax = fig.add_subplot(gs2[i])
        gs2_axes.append(ax)

    for model in ['SEAS5', 'IceNet']:

        if model == 'IceNet':
            color = sns.color_palette('tab10')[0]
        elif model == 'SEAS5':
            color = sns.color_palette('tab10')[3]

        lw = 1

        ax = gs1_axes[0]
        ax.plot(sip_primes, sip_bounding_df.loc[model].frac_ice_edge, lw=lw, color=color, label=model)
        ax.set_ylabel("% of ice edge in $[p', 1-p']$")

        ax = gs1_axes[1]
        ax.plot(sip_primes, sip_bounding_df.loc[model].frac_all, lw=lw, color=color, label=model)
        ax.set_ylabel("% of all grid cells in $[p', 1-p']$")

        ax = gs1_axes[2]
        ax.plot(sip_primes, sip_bounding_df.loc[model].frac_error, lw=lw, color=color, label=model)
        ax.set_ylabel("% of binary errors in $[p', 1-p']$")

        if model == 'IceNet':
            ax = gs2_axes[0]
            ax.plot(sip_bounding_df.loc[model].frac_ice_edge, sip_bounding_df.loc[model].frac_all, lw=lw, color=color, label=model)
            ax.set_xlabel("% of ice edge in $[p', 1-p']$")
            ax.set_ylabel("% of all grid cells in $[p', 1-p']$")

            ax = gs2_axes[1]
            ax.plot(sip_bounding_df.loc[model].frac_ice_edge, sip_bounding_df.loc[model].frac_error, lw=lw, color=color, label=model)
            ax.set_xlabel("% of ice edge in $[p', 1-p']$")
            ax.set_ylabel("% of binary errors in $[p', 1-p']$")

        # gs2_axes[0].plot(sip_primes, frac_error, lw=1., color=color, label=model)
        # gs2_axes[1].plot(sip_primes, frac_all, lw=1., color=color)

        if model == 'IceNet':
            sip_prime_90 = sip_primes[np.argwhere(np.diff(np.sign(np.array(sip_bounding_df.loc[model].frac_ice_edge) - 90)))].ravel()[0]
            frac_all_90 = sip_bounding_df.loc[model].frac_all[np.argwhere(np.diff(np.sign(np.array(sip_primes) - sip_prime_90)))].ravel()[0]

            w = .5  # Arrow width
            arrowprops = dict(shrink=0., width=w, headwidth=6*w, headlength=6*w, color='k', ls='--', lw=0.25)

            gs1_axes[0].annotate("", xytext=(0, 90), xy=(sip_prime_90, 90), arrowprops=arrowprops)
            gs1_axes[0].annotate("", xytext=(sip_prime_90, 90), xy=(sip_prime_90, 0), arrowprops=arrowprops)
            gs1_axes[0].text(s="90% ice\nedge bound\n$p'={:.03f}$".format(sip_prime_90), x=sip_prime_90+.02, y=5)

            # gs2_axes[0].text(s="", x=sip_prime_90+.02, y=.20)
            gs2_axes[0].annotate("", xytext=(90, 0), xy=(90, frac_all_90), arrowprops=arrowprops)
            gs2_axes[0].annotate("", xytext=(90, frac_all_90), xy=(0, frac_all_90), arrowprops=arrowprops)
            gs2_axes[0].text(s="Ice edge region\ncover = {:.1f}%".format(frac_all_90), x=25, y=frac_all_90+5)

    for i in [0, 1, 2]:
        # gs2_axes[i].set_yticklabels(['{}%'.format(perc) for perc in range(0, 120, 20)])
        gs1_axes[i].set_xlabel("$p'$")
        gs1_axes[i].set_ylim([0, 1.])
        gs1_axes[i].set_xlim([-.003, .503])
        gs1_axes[i].set_xticks(np.arange(0, .55, .1))
        gs1_axes[i].set_yticks(np.arange(0, 105, 10))
        # gs1_axes[i].xaxis.grid(True, which='minor')
        # gs1_axes[i].yaxis.grid(True, which='minor')
        gs1_axes[i].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(.05))
        gs1_axes[i].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        gs1_axes[i].set_yticklabels(['{}%'.format(perc) for perc in range(0, 120, 10)])

    for i in [0, 1]:
        gs2_axes[i].set_xticks(np.arange(0, 105, 10))
        gs2_axes[i].set_yticks(np.arange(0, 105, 10))
        gs2_axes[i].set_ylim([0, 100])
        gs2_axes[i].set_xlim([0, 100])
        # gs2_axes[i].xaxis.grid(True, which='minor')
        # gs2_axes[i].yaxis.grid(True, which='minor')
        gs2_axes[i].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        gs2_axes[i].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        gs2_axes[i].set_yticklabels(['{}%'.format(perc) for perc in range(0, 120, 10)])

    # gs2_axes[0].set_ylabel("$\int_{p'}^{1-p'} \hat{P}(p|e=1) \mathrm{d}p$", labelpad=-3)
    # gs2_axes[1].set_ylabel("$\int_{p'}^{1-p'} \hat{P}(p) \mathrm{d}p$", labelpad=-3)

    gs1_axes[0].legend(loc='best')

    gs1_axes[0].set_title("a", fontweight='bold', loc='left')
    gs1_axes[1].set_title("b", fontweight='bold', loc='left')
    gs1_axes[2].set_title("c", fontweight='bold', loc='left')
    gs2_axes[0].set_title("d", fontweight='bold', loc='left')
    gs2_axes[1].set_title("e", fontweight='bold', loc='left')

    handles, labels = gs1_axes[0].get_legend_handles_labels()
    gs1_axes[0].legend(handles[::-1], labels[::-1], loc='best')

    gs_outer.update(hspace=.5)

    plt.savefig(os.path.join(fig_folder_png, 'supp_fig3.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'supp_fig3.pdf'))
    plt.close()

    print('Done.')

### Supplementary Figure 4
####################################################################

if True:
    print('Plotting Supplementary Figure 4... ', end='', flush=True)

    warnings.simplefilter("ignore", FutureWarning)

    # Only plot calibration curve over val dates
    uncertainty_val_df = \
        uncertainty_df.loc[pd.IndexSlice[:, slice(val_start, val_end)], :].\
        reset_index()

    uncertainty_val_df = uncertainty_val_df.set_index(['Model', 'Error?']).sort_index()

    bins = {}
    # 99 bins betw 0 and 1: ensures no bin edge aligns with SEAS5 discrete SIP values
    bins['SEAS5'] = np.linspace(0., 1., 100)
    # 100 bins betw 0 and 1: ensures symmetry for IceNet bins
    bins['IceNet'] = np.linspace(0., 1., 101)

    gs1_axes = []

    fig = plt.figure(figsize=(6, 4))

    gs1 = fig.add_gridspec(nrows=2, ncols=3, wspace=0.4, hspace=0.5)

    for i, model in enumerate(['IceNet', 'SEAS5']):

        if model == 'IceNet':
            color = sns.color_palette('tab10')[0]
        elif model == 'SEAS5':
            color = sns.color_palette('tab10')[3]

        ax = fig.add_subplot(gs1[i, 0])
        gs1_axes.append(ax)

        if model == 'IceNet':
            title = 'a'
        elif model == 'SEAS5':
            title = 'd'

        ax.set_title(title, fontweight='bold', loc='left')

        l = ax.plot([0, .5, 1], [0, .5, 0], 'k', lw=.5, label='Perfect reliability')

        # KDE for IceNet (continuous SIP), histogram for SEASS5 (discrete SIP)
        if model == 'IceNet':
            sns.histplot(
                x='SIP',
                data=uncertainty_val_df.loc[model, True],
                weights=uncertainty_val_df.loc[model, True]['Histogram weight'],
                kde=True,
                kde_kws=dict(bw_method=0.005),
                line_kws=dict(linewidth=0.75),
                fill=False,
                linewidth=0.,
                color=color,
                bins=bins[model],
                ax=ax)
        else:
            sns.histplot(
                x='SIP',
                data=uncertainty_val_df.loc[model, True],
                weights=uncertainty_val_df.loc[model, True]['Histogram weight'],
                kde=False,
                color=color,
                bins=bins[model],
                ax=ax)

        if i == 0:
            ylabel_prefix = r'$\bf{IceNet}$'
        elif i == 1:
            ylabel_prefix = r'$\bf{SEAS5}$'
        ax.set_ylabel(ylabel_prefix + '\n\n' + '$\hat{P}(e=1|p)$')
        ax.set_yticks(np.arange(0.0, .9, 0.1))
        ax.set_ylim([0., None])
        ax.set_xlabel('$p$')
        ax.set_xticks(np.arange(0.0, 1.1, 0.25))
        ax.set_xlim([0., 1.])
        if i == 0:
            ax.legend(l, ['Perfect calibration'], loc='best', fontsize=6.)

        # ax = axes[i, 1]
        ax = fig.add_subplot(gs1[i, 1])
        gs1_axes.append(ax)

        if model == 'IceNet':
            title = 'b'
        elif model == 'SEAS5':
            title = 'e'

        ax.set_title(title, fontweight='bold', loc='left')

        if model == 'IceNet':
            sns.kdeplot(
                x='SIP',
                data=uncertainty_val_df.loc[pd.IndexSlice[model, :]],
                bw_method=0.005,
                color=color,
                linewidth=0.75,
                ax=ax)
        else:
            sns.histplot(
                x='SIP',
                data=uncertainty_val_df.loc[pd.IndexSlice[model, :]],
                kde=False,
                color=color,
                bins=bins[model],
                ax=ax)

        ax.set_yscale('log')
        ax.tick_params(axis='y', which='both', left=True, labelleft=False)
        ax.set_ylabel('$\log(\hat{P}(p))$')
        ax.set_xlabel('$p$')
        ax.set_xticks(np.arange(0.0, 1.1, 0.25))
        ax.set_xlim([0., 1.])

        ax = fig.add_subplot(gs1[i, 2])
        gs1_axes.append(ax)

        if model == 'IceNet':
            title = 'c'
        elif model == 'SEAS5':
            title = 'f'

        ax.set_title(title, fontweight='bold', loc='left')

        if model == 'IceNet':
            sns.kdeplot(
                x='SIP',
                data=uncertainty_val_df.loc[model, True],
                bw_method=0.005,
                color=color,
                linewidth=0.75,
                ax=ax)
        else:
            sns.histplot(
                x='SIP',
                data=uncertainty_val_df.loc[model, True],
                kde=False,
                color=color,
                bins=bins[model],
                ax=ax)

        ax.set_ylabel('$\hat{P}(p|e=1)$')
        ax.set_yticks([])
        ax.set_ylim([0., None])
        ax.set_xticks(np.arange(0.0, 1.1, 0.25))
        ax.set_xlabel('$p$')
        ax.set_xlim([0., 1.])

    for i in [1, 2]:
        gs1_axes[i].axvline(x=sip_prime_90, color='k', ls='--', lw=.5)
        gs1_axes[i].axvline(x=1-sip_prime_90, color='k', ls='--', lw=.5)
        gs1_axes[i].annotate(
            "Ice edge\nregion",
            textcoords=('data', 'axes fraction'),
            xycoords=('data', 'axes fraction'),
            xytext=(.5, .8),
            xy=(sip_prime_90, .8),
            horizontalalignment='center',
            verticalalignment='center',
            arrowprops=arrowprops)
        gs1_axes[i].annotate(
            "Ice edge\nregion",
            textcoords=('data', 'axes fraction'),
            xycoords=('data', 'axes fraction'),
            xytext=(.5, .8),
            xy=(1-sip_prime_90, .8),
            horizontalalignment='center',
            verticalalignment='center',
            arrowprops=arrowprops)

    for i in [0, 1]:
        gs2_axes[i].set_ylim([0, 1.])
        gs2_axes[i].set_xlim([-.005, .505])

    handles, labels = gs2_axes[0].get_legend_handles_labels()
    gs2_axes[0].legend(handles[::-1], labels[::-1], loc='best')

    gs_outer.update(hspace=.3)

    plt.savefig(os.path.join(fig_folder_png, 'supp_fig4.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'supp_fig4.pdf'))
    plt.close()

    print('Done.')

    val_end

### Supplementary Figure 5
####################################################################

if True:
    print('Plotting Supplementary Figure 5... ', end='', flush=True)

    ice_edge_region_test_df = \
        ice_edge_region_df.loc[pd.IndexSlice[:, slice(test_start, test_end)], :]

    fig = plt.figure(figsize=(6, 6))

    gs_outer = fig.add_gridspec(nrows=2, ncols=1, hspace=0.5, height_ratios=[2, 1])

    gs1 = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs_outer[0], wspace=0.5, hspace=0.5)

    ax = fig.add_subplot(gs1[0, :])

    sns.lineplot(
        data=ice_edge_region_test_df.loc[1],
        x='Forecast date',
        y='Area',
        marker='o',
        markerfacecolor='gray',
        color='gray',
        ax=ax)
    sns.lineplot(
        data=ice_edge_region_test_df.loc[6],
        x='Forecast date',
        y='Area',
        marker='o',
        markerfacecolor='k',
        color='k',
        ax=ax)

    ax.legend(['1 month lead time', '6 month lead time'], loc='best')
    # ax.set_ylabel('Active grid cell area covered\nby 95% error region (%)')
    # ax.set_ylabel('Active area covered\nby ice edge region (%)')
    ax.set_ylabel('Area covered by\nice edge region (km$^2$)')
    # ax.ticklabel_format(axis='y', scilimits=(0, 0))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '${:.0f}\\times 10^6$'.format(x/1e6)))
    # ax.set_ylim((0, 50))
    ax.set_xlabel('Forecast date')
    ax.set_title("a", fontweight='bold', loc='left')

    ax.set_xticks([])
    ax.set_xticklabels([''])

    # Plot vertical dashed lines at each year boundary
    years = sorted(list(set(ice_edge_region_test_df.reset_index()['Forecast date'].dt.year)))
    years.append(np.min(years) - 1)
    for year in years:
        ax.axvline(x=pd.Timestamp(year, 12, 15), color='k', linestyle='--', linewidth=0.5)

    ax = fig.add_subplot(gs1[1, :])

    sns.lineplot(
        data=ice_edge_region_test_df.loc[1],
        x='Forecast date',
        y='Coverage',
        marker='o',
        markerfacecolor='gray',
        color='gray',
        ax=ax)
    sns.lineplot(
        data=ice_edge_region_test_df.loc[6],
        x='Forecast date',
        y='Coverage',
        marker='o',
        markerfacecolor='k',
        color='k',
        ax=ax)

    # ax.set_ylabel('Ice-ocean errors covered\nby 95% error region (%)')
    # ax.set_ylabel('Binary errors covered\nby ice edge region (%)')
    ax.set_ylabel('Ice edge covered\nby ice edge region (%)')
    # ax.set_ylim((0, 100))
    # ax.set_ylim((50, 100))
    ax.set_xlabel('Forecast date')
    ax.set_yticklabels(['{}%'.format(num) for num in ax.get_yticks()])
    ax.set_title("b", fontweight='bold', loc='left')

    # Plot vertical dashed lines at each year boundary
    for year in years:
        ax.axvline(x=pd.Timestamp(year, 12, 15), color='k', linestyle='--', linewidth=0.5)

    xticks_timestamps = sorted(list(set(ice_edge_region_test_df.reset_index()['Forecast date'])))
    xticks_datetime = [xtick.date() for xtick in xticks_timestamps]
    xticks_year_month = [xtick.strftime('%Y-%m') for xtick in xticks_datetime]
    ax.set_xticks(xticks_timestamps)
    ax.set_xticklabels(xticks_year_month)
    plt.xticks(rotation=70)

    gs2 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=2, subplot_spec=gs_outer[1], wspace=0.5, hspace=0.5)

    line_kws = dict(err_style='bars', linewidth=1.,
                    err_kws=dict(capsize=3., barsabove=True))

    ax = fig.add_subplot(gs2[0])
    sns.boxplot(
        data=ice_edge_region_test_df.reset_index(),
        x='Leadtime',
        y='Coverage',
        color='gray',
        ax=ax,
        whis=np.inf,  # No outliers
        linewidth=.5
    )
    sns.swarmplot(
        data=ice_edge_region_test_df.reset_index(),
        x='Leadtime',
        y='Coverage',
        color='black',
        ax=ax,
        size=2
    )
    ax.set_title("c", fontweight='bold', loc='left')
    ax.set_xlabel('Lead time (months)')
    ax.set_ylabel('Ice edge covered\nby ice edge region (%)')

    ax = fig.add_subplot(gs2[1])
    sns.boxplot(
        data=ice_edge_region_test_df.reset_index(),
        x='Leadtime',
        y='Area',
        color='gray',
        ax=ax,
        whis=np.inf,  # No outliers
        linewidth=.5
    )
    sns.swarmplot(
        data=ice_edge_region_test_df.reset_index(),
        x='Leadtime',
        y='Area',
        color='black',
        ax=ax,
        size=2
    )
    ax.set_title("d", fontweight='bold', loc='left')
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, pos: '${:.0f}\\times 10^6$'.format(x/1e6)))
    ax.set_ylabel('Area covered by\nice edge region (km$^2$)')
    ax.set_xlabel('Lead time (months)')

    plt.savefig(os.path.join(fig_folder_png, 'supp_fig5.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'supp_fig5.pdf'))
    plt.close()

    print('Done.')

### Supplementary Figure 6
####################################################################

if True:
    print('Plotting Supplementary Figure 6... ', end='', flush=True)

    ice_edge_region_heatmap = ice_edge_region_df.reset_index()[['Leadtime', 'Calendar month', 'Coverage']].\
        groupby(['Calendar month', 'Leadtime']).mean().reset_index().\
        pivot('Calendar month', 'Leadtime', 'Coverage').reindex(month_names)

    area_heatmap = ice_edge_region_df.reset_index()[['Leadtime', 'Calendar month', 'Area']].\
        groupby(['Calendar month', 'Leadtime']).mean().reset_index().\
        pivot('Calendar month', 'Leadtime', 'Area').reindex(month_names)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 2))

    ax = axes[0]
    sns.heatmap(data=ice_edge_region_heatmap,
                annot=False,
                # fmt=df1_fmt,
                cbar_kws=dict(label='Binary error coverage (%)', extend='both', fraction=0.05, aspect=8),
                # cmap=cmap1,
                # cbar_ax=cbar_ax1,
                ax=ax)
    ax.set_title("a", fontweight='bold', x=-.15, loc='left')
    ax.set_ylabel('Calendar month')
    ax.set_xlabel('Lead time (months)')
    ax.set_title('Ice edge covered by ice edge region (%)')

    ax = axes[1]
    sns.heatmap(data=area_heatmap,
                # cmap=cmap2,
                # center=0.,
                annot=False,
                cbar_kws=dict(label='Ice edge region area (km$^2$)', extend='both', fraction=0.05, aspect=8,
                              format= mpl.ticker.FuncFormatter(lambda x, pos: '${:.0f}\\times 10^6$'.format(x/1e6))),
                # fmt=fmt,
                # cbar_kws=cbar_kws,
                # cbar_ax=cbar_ax2,
                ax=ax)
    ax.set_title("b", fontweight='bold', x=-.15, loc='left')
    ax.set_title('Area covered by ice edge region (km$^2$)')
    ax.set_ylabel('')
    ax.set_xlabel('Lead time (months)')

    plt.tight_layout()

    # for ax in axes[:-1]:
    #     ax.tick_params(axis='x', bottom=False, labelbottom=False)

    for ax in axes:
        ax.tick_params(axis='y', rotation=0)

    plt.savefig(os.path.join(fig_folder_png, 'supp_fig6.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'supp_fig6.pdf'))
    plt.close()

    print('Done.')

### Supplementary Figure 7
####################################################################

if True:
    print('Plotting Supplementary Figure 7... ', end='', flush=True)

    all_ordered_variable_names = dataloader.determine_variable_names()

    # Mean over seed and forecast tate
    mean_results_df = pap_results_df.groupby(['Leadtime', 'Variable']).mean()

    mean_results_heatmap = mean_results_df.reset_index().\
        pivot(index='Variable', columns='Leadtime').\
        reindex(all_ordered_variable_names)['Accuracy drop (%)']

    mean_results_df = mean_results_df.reset_index()

    cbar_kws = {}
    cbar_kws['label']='Accuracy change (%)'

    verbose_varnames = []
    short_varnames = mean_results_heatmap.index.values.astype('str')
    for varname in short_varnames:
        verbose_varname = utils.make_varname_verbose_any_leadtime(varname)
        verbose_varnames.append(verbose_varname)
    mean_results_heatmap.index = verbose_varnames

    with plt.rc_context({'font.size': 9, 'axes.labelsize': 9, 'ytick.labelsize': 9, 'xtick.labelsize': 9}):
        fig, ax = plt.subplots(figsize=(6, 9))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='10%', pad=0.1)

        sns.heatmap(
            data=mean_results_heatmap,
            annot=True,
            annot_kws=dict(fontsize=7),
            fmt='.2f',
            ax=ax,
            cmap='RdBu_r',
            center=0.,
            cbar_kws=cbar_kws,
            vmin=-.5,
            cbar_ax=cax,
        )
        ax.set_xlabel('Lead time (months)')
        ax.set_ylabel('Input variable name')
        cax.set_frame_on(True)
        ax.set_frame_on(True)
        plt.tight_layout()

    plt.savefig(os.path.join(fig_folder_png, 'supp_fig7.png'))
    plt.savefig(os.path.join(fig_folder_pdf, 'supp_fig7.pdf'))
    plt.close()

    print('Done.')


### Table 1
####################################################################

if True:
    print('Generating Table 1... ', end='', flush=True)

    # Add a 'calendar month' column
    df_month_numbers = pap_results_df['Forecast date'].dt.month.values
    df_month_names = month_names[df_month_numbers - 1]
    pap_results_df['Calendar month'] = df_month_names

    leadtimes_to_plot = [5, 3, 1]

    table_month_names = ['Sept', 'Mar']

    for month in table_month_names:

        pap_month_results_df = pap_results_df.set_index('Calendar month').loc[month]

        # Mean over seed and forecast tate
        mean_results_df = pap_month_results_df.groupby(['Leadtime', 'Variable']).mean()

        table_df = pd.DataFrame(index=np.arange(1, 6))
        table_df.index.name = 'Rank'

        fc_month_idx = np.where(month_names == month)[0][0]

        mean_results_df_verbose = mean_results_df.copy().reset_index()
        mean_results_df_verbose = \
            mean_results_df_verbose[(mean_results_df_verbose.Variable != 'cos(month)') &
                                    (mean_results_df_verbose.Variable != 'sin(month)') &
                                    (mean_results_df_verbose.Variable != 'land')]

        verbose_varnames = []
        short_varnames = mean_results_df_verbose.Variable.values.astype('str')
        leadtimes = mean_results_df_verbose.Leadtime.values
        for varname, leadtime in zip(short_varnames, leadtimes):
            verbose_varname = utils.make_varname_verbose(varname, leadtime, fc_month_idx)
            verbose_varnames.append(verbose_varname)

        mean_results_df_verbose.Variable = verbose_varnames

        mean_results_df_verbose = mean_results_df_verbose.drop('Seed', axis=1)

        # Top 5
        for i, leadtime in enumerate(leadtimes_to_plot):

            initialisation_month = month_names[(fc_month_idx - leadtime) % 12]

            leadtime_sorted_top5_df = \
                mean_results_df_verbose[mean_results_df_verbose.Leadtime == leadtime].\
                drop('Leadtime', axis=1).set_index('Variable').sort_values('Accuracy drop (%)')[0:5]

            acc_drops = leadtime_sorted_top5_df.values

            labels = []
            for drop_i in range(len(acc_drops)):
                string = leadtime_sorted_top5_df.index[drop_i]
                string += ' ({:.2f}%) '.format(leadtime_sorted_top5_df.values.ravel()[drop_i])
                labels.append(string)

            labels = np.array(labels).reshape(leadtime_sorted_top5_df.values.shape)

            table_df['{} intitialisation ({} month leadtime)'.format(initialisation_month, leadtime)] = labels.ravel()

        if month == 'Sept':
            subtable = 'a'
        elif month == 'Mar':
            subtable = 'b'

        table_df.to_csv(os.path.join(fig_folder_table, f'table_1{subtable}.csv'))

    print('Done.')

from core.trainer import train_record
import torch
import numpy as np
import argparse
import json
import os
import torch
import matplotlib.pyplot as plt


class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__

parser = argparse.ArgumentParser()
parser.add_argument(
    '--save_dir',
    default='_sin_',
    help="Directory to save the data in")
parser.add_argument(
    '--json_file',
    default='sine.json',
    help="Directory containing params.json")
parser.add_argument(
    '--opt',
    default='DPMCL',
    help="The optimization")
parser.add_argument(
    '--kappa',
    default=None,
    help="kappa value")
parser.add_argument(
    '--zeta',
    default=None,
    help="zeta value")
parser.add_argument(
    '--eta',
    default=None,
    help="eta value")
parser.add_argument(
    '--total_runs',
    default=None,
    help="total number of runs value")
parser.add_argument(
    '--total_samples',
    default=None,
    help="total no. of tasks value")
parser.add_argument(
    '--batch_size',
    default=None,
    help="kappa value")

if __name__ == '__main__':
    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join('params/', args.json_file)
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path).dict
    params['opt'] = args.opt
    params['save_file'] = args.save_dir
    print(params['save_file']) 
    if args.total_runs is not None:
        params['total_runs'] = int(args.total_runs)
    
    if args.total_samples is not None:
        params['total_samples'] = int(args.total_samples)

    if args.zeta is not None:
        params['zeta'] = int(args.zeta)
    
    if args.eta is not None:
        params['eta'] = int(args.eta)

    if args.kappa is not None:
        params['kappa'] = int(args.kappa)

    if params['problem'] == 'classification':
        params['criterion'] = torch.nn.CrossEntropyLoss()
    else: 
        params['criterion'] = torch.nn.MSELoss()

    print(params)
    x = input("Lets start here")
    # ################################################
    # # Run this command multiple times for individual runs of the code

    CME = np.zeros([params['total_runs'],\
    params['total_samples']])

    CTE = np.zeros([params['total_runs'],\
    params['total_samples']])

    for i in range(params['total_runs'] ):
        Runner = train_record(params)

        CTE[i,:], CME[i,:] = Runner.main()
        # if params['task_wise']>0:
        #     np.savetxt(params['save_file']+str(i)+'TE.csv', TE, delimiter = ',')

        Runner.show_gpu('after all stuff have been removed')
        Runner.print_gpu_obj()

    print(CTE.shape, CME.shape)

    
    # # print(self.get_gpu_memory_map())
    # # self.show_gpu(f'{0}: Before deleting objects')
    # # self.show_gpu(f'{0}: After deleting objects') 
    # # gc.collect()
    # # self.show_gpu(f'{0}: After gc collect') 
    # # self.show_gpu(f'{0}: After empty cache') 
    # # self.show_gpu('after all stuff have been removed')
    # # self.print_gpu_obj()

    ################################################
    np.savetxt(params['save_file']+'CME.csv', CME, delimiter=',')
    np.savetxt(params['save_file']+'CTE.csv', CTE, delimiter=',')

    ## Lets plot things and see how is the behavior
    Runner = None
    def cm2inch(value):
        return value/2.54

    small = 7
    med = 10
    large = 12
    plt.style.use('seaborn-white')
    COLOR = 'darkslategray'
    params1 = {'axes.titlesize': small,
            'legend.fontsize': small,
            'figure.figsize': (cm2inch(15),cm2inch(8)),
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small, 
            'font.family': "sans-serif",
            'font.sans-serif': "Myriad Hebrew",
                'text.color' : COLOR,
                'axes.labelcolor' : COLOR,
                'axes.linewidth' : 0.3,
                'xtick.color' : COLOR,
                'ytick.color' : COLOR}

    plt.rcParams.update(params1)
    plt.rc('text', usetex = False)
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['mathtext.fontset'] = 'cm'
    from matplotlib.lines import Line2D
    import matplotlib.font_manager as font_manager
    large=24; med=8; small=7
    labels = ['Old', 'New', 'Original']
    titles = ['worst', 'median', 'best']

    # create plots with numpy array
    fig,a=plt.subplots(2,1, sharex=False, dpi=600,\
        gridspec_kw={'wspace':0.7, 'hspace':0.7})


    #CME
    # Some Plot oriented settings 
    a[0].spines["top"].set_visible(False)    
    a[0].spines["bottom"].set_visible(False)    
    a[0].spines["right"].set_visible(False)    
    a[0].spines["left"].set_visible(True)  

    a[0].grid(linestyle=':', linewidth=0.5)
    a[0].get_xaxis().tick_bottom()    
    a[0].get_yaxis().tick_left()  

    
    # Some Plot oriented settings 
    a[1].spines["top"].set_visible(False)    
    a[1].spines["bottom"].set_visible(False)    
    a[1].spines["right"].set_visible(False)    
    a[1].spines["left"].set_visible(True)  
    a[1].grid(linestyle=':', linewidth=0.5)
    a[1].get_xaxis().tick_bottom()    
    a[1].get_yaxis().tick_left()  

    t = np.arange(CME.shape[1])

    mean = np.mean(CME, axis=0)
    yerr = np.std(CME, axis=0)

    print(t.shape, mean.shape, yerr.shape)
    a[0].fill_between(t, (mean + yerr), (mean), alpha=0.4, color = color[3])
    # a[0].set_xlim([0, 500])
    #a[0].set_yscale('log')
    a[0].set_xlabel('Tasks')
    a[0].set_ylabel('CME')

    a[0].set_title('('+params["data_id"]+","+str(params["opt"])+')')
    # a[0].legend(bbox_to_anchor=(0.0008, -0.5, 0.3, 0.1), loc = 'upper left',ncol=3 )
    
    
    t = np.arange(CTE.shape[1])
    # The Final Plots with CME
    mean = np.mean(CTE, axis=0)
    yerr = np.std(CTE, axis=0)
    print(mean, yerr)
    a[1].fill_between(t, (mean + yerr), (mean), alpha=0.4, color = color[3])
    #a[1].set_yscale('log')
    a[1].set_xlabel('Tasks')
    a[1].set_ylabel('CTE')
    plt.savefig( params["data_id"]+"_"+params['opt']+".png", dpi=600)

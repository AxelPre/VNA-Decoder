import copy
import os
import numpy as np
import pandas as pd
from scipy import optimize
from src.rotated_surface_model import RotSurCode
# decoders
from decoders import EWD_alpha, EWD_droplet_alpha
#from src.dilatedrnn import DilatedRNNWavefunction, CustomRNNCell, return_local_energies, runVNA
def get_individual_error_rates(params):
    assert params['noise'] in ['depolarizing', 'alpha', 'biased'], f'{params["noise"]} is not implemented.'
    
    if params['noise'] == 'biased':
        eta = params['eta']
        p = params['p_error']
        p_z = p * eta / (eta + 1)
        p_x = p / (2 * (eta + 1))
        p_y = p_x
    
    if params['noise'] == 'alpha':
        # Calculate pz_tilde from p_error (total error prob.)
        p = params['p_error']
        alpha = params['alpha']
        p_tilde = p / (1 - p)
        pz_tilde = optimize.fsolve(lambda x: x + 2*x**alpha - p_tilde, 0.5)[0]
        
        p_z = pz_tilde*(1 - p)
        p_x = p_y = pz_tilde**alpha * (1 - p)


        #### obs obs, changing p_y and p_z to 0
        p_y = p_z = 0
        
    if params['noise'] == 'depolarizing':
        p_x = p_y = p_z = params['p_error']/3

    return p_x, p_y, p_z


# This function generates training data with help of the MCMC algorithm
def generate(file_path, params, nbr_datapoints=10**6, fixed_errors=None):

    # Creates df
    df = pd.DataFrame()

    # Add parameters as first entry in dataframe
    names = ['data_nr', 'type']
    index_params = pd.MultiIndex.from_product([[-1], np.arange(1)],
                                                names=names)
    df_params = pd.DataFrame([[params]],
                            index=index_params,
                            columns=['data'])
    df = df.append(df_params)

    print('\nDataFrame opened at: ' + str(file_path))

    # If using a fixed number of errors, let the max number of datapoins be huge
    if fixed_errors != None:
        nbr_datapoints = 10000000
    failed_syndroms = 0

    # Initiate temporary list with results (to prevent appending to dataframe each loop)
    df_list = []

    p_x, p_y, p_z = get_individual_error_rates(params)

    # Loop to generate data points
    for i in range(nbr_datapoints):
        print('Starting generation of point nr: ' + str(i + 1), flush=True)

        # Initiate code
        if params['code'] == 'rotated':
            assert params['noise'] in ['depolarizing', 'alpha', 'biased'], f'{params["noise"]}-noise is not compatible with "{params["code"]}"-model.'
            init_code = RotSurCode(params['size'])
            init_code.generate_random_error(p_x=p_x, p_y=p_y, p_z=p_z)

        # Flatten initial qubit matrix to store in dataframe
        df_qubit = copy.deepcopy(init_code.qubit_matrix)
        eq_true = init_code.define_equivalence_class()

        # Create inital error chains for algorithms to start with
        if params['mwpm_init']: #get mwpm starting points
            assert params['code'] == 'planar', 'Can only use eMWPM for planar model.'
            init_code = class_sorted_mwpm(init_code)
            print('Starting in MWPM state')
        else: #randomize input matrix, no trace of seed.
            init_code.qubit_matrix, _ = init_code.apply_random_logical()
            init_code.qubit_matrix = init_code.apply_stabilizers_uniform()
            print('Starting in random state')

        # Generate data for DataFrame storage  OBS now using full bincount, change this
        if params['method'] == "EWD":
            if params['noise'] == 'alpha':
                alpha=params['alpha']
                p_tilde_sampling = params['p_sampling'] / (1 - params['p_sampling'])
                pz_tilde_sampling = optimize.fsolve(lambda x: x + 2*x**alpha - p_tilde_sampling, 0.5)[0]
                p_tilde = params['p_error'] / (1 - params['p_error'])
                pz_tilde = optimize.fsolve(lambda x: x + 2*x**alpha - p_tilde, 0.5)[0]
                df_eq_distr = EWD_alpha(init_code,
                                         pz_tilde,
                                         alpha,
                                         params['steps'],
                                         pz_tilde_sampling=pz_tilde_sampling,
                                         onlyshortest=params['onlyshortest'])
                df_eq_distr = np.array(df_eq_distr)
                print(np.sum(df_eq_distr), df_eq_distr)

            else:
                raise ValueError(f'''EWD does not support "{params['noise']}" noise''')
        """
        if params['method'] == "VNA_1D_Dilated":
            currentn=0
            df_eq_distr = runVNA(init_code, threshold=0.05, nsamples=100, size=params['size'], currentn=currentn)
            df_eq_distr = np.array(df_eq_distr)
            print(np.sum(df_eq_distr), df_eq_distr)
        # Generate data for DataFrame storage  OBS now using full bincount, change this
        """
        # Create indices for generated data
        names = ['data_nr', 'type']
        index_qubit = pd.MultiIndex.from_product([[i], np.arange(1)],
                                                 names=names)
        index_distr = pd.MultiIndex.from_product([[i], np.arange(1)+1], names=names)

        # Add data to Dataframes
        df_qubit = pd.DataFrame([[df_qubit.astype(np.uint8)]], index=index_qubit,
                                columns=['data'])
        df_distr = pd.DataFrame([[df_eq_distr]],
                                index=index_distr, columns=['data'])

        # Add dataframes to temporary list to shorten computation time
        
        df_list.append(df_qubit)
        df_list.append(df_distr)

        # Every x iteration adds data to data file from temporary list
        # and clears temporary list
        
        if (i + 1) % 50 == 0:
            df = df.append(df_list)
            df_list.clear()
            print('Intermediate save point reached (writing over)')
            df.to_pickle(file_path)
            print('Total number of failed syndroms:', failed_syndroms)
        
        # If the desired amount of errors have been achieved, break the loop and finish up
        if failed_syndroms == fixed_errors:
            print('Desired amount of failed syndroms achieved, stopping data generation.')
            break

    # Adds any remaining data from temporary list to data file when run is over
    if len(df_list) > 0:
        df = df.append(df_list)
        print('\nSaving all generated data (writing over)')
        df.to_pickle(file_path)
    
    print('\nCompleted')


if __name__ == '__main__':
    # Get job array id, working directory
    job_id = os.getenv('SLURM_ARRAY_JOB_ID')
    array_id = os.getenv('SLURM_ARRAY_TASK_ID')
    local_dir = os.getenv('TMPDIR')

    # Use environment variables to get parameters
    size = int(os.getenv('CODE_SIZE'))
    code = str(os.getenv('CODE_TYPE'))
    alpha = float(os.getenv('CODE_ALPHA'))

    job_name = str(os.getenv('JOB_NAME'))
    start_p = float(os.getenv('START_P'))
    end_p = float(os.getenv('END_P'))
    num_p = int(os.getenv('NUM_P'))
    mwpm_init = bool(int(os.getenv('MWPM_INIT')))
    p_sampling = float(os.getenv('P_SAMPLE'))

    alg = str(os.getenv('ALGORITHM'))
    only_shortest = bool(int(os.getenv('ONLY_SHORTEST')))

    params = {'code': code,
            'method': alg,
            'size': size,
            'noise': 'alpha',
            'p_error': np.linspace(start_p, end_p, num=num_p)[int(array_id)],
            'eta': 0.5,
            'alpha': alpha,
            'p_sampling': p_sampling,
            'droplets': 1,
            'mwpm_init': mwpm_init,
            'fixed_errors':None,
            'Nc': None,
            'iters': 10,
            'conv_criteria': 'error_based',
            'SEQ': 2,
            'TOPS': 10,
            'eps': 0.01,
            'onlyshortest': only_shortest}
    # Steps is a function of code size L
    params.update({'steps': int(5*params['size']**5)})
    
    print('Nbr of steps to take if applicable:', params['steps'])

    # Build file path
    file_path = os.path.join(local_dir, f'data_paper_{job_name}_{job_id}_{array_id}.xz')
    
    # Generate data
    generate(file_path, params, nbr_datapoints=10000, fixed_errors=params['fixed_errors'])

classdef PRB_System_Constants
    properties (Constant)
        % Network configuration
        N_RB = 200; % Number of PRBs
        f_sc = 30e3; % Subcarrier bandwidth (30 kHz)
        N_sc_RB = 12; % Number of subcarriers per PRB
        B = 30e3 * 12; % Bandwidth per PRB
        T_drop = 0.5e-3; % Simulation step size (0.5ms)
        Fd = 0.5e-3; % TTI duration
        
        % Service types
        I_V2X = 1; % V2X service index
        I_eMBB = 2; % eMBB service index
        I_mMTC = 3; % mMTC service index
        I_VoNR = 4; % VoNR service index (new)
        Nr_S = 4; % Number of service types (updated)
        
        % Traffic parameters
        M_veh = 8; % Number of V2X UEs
        M_eMBB = 4; % Number of eMBB UEs
        M_mMTC = 12; % Number of mMTC UEs
        M_VoNR = 4; % Number of VoNR UEs (new)
        lambda_niu = 200; % V2X packet arrival rate
        lambda_e = 300; % eMBB session arrival rate
        p_mMTC_tx = 0.75; % mMTC transmission probability
        p_VoNR_tx = 0.95; % VoNR transmission probability (new)
        max_mMTC_packet_size = 128; % Maximum mMTC packet size
        VoNR_packet_size = 856; % VoNR packet size in bytes (new)
        Sm = 300; % V2X packet size (bytes)
        Rb_session = 1e6; % eMBB session bit rate (1 Mbps)
        velocity = 80/3.6; % Vehicle velocity in m/s (80 km/h)
        
        % SLA requirements (new)
        max_V2X_latency = 10e-3; % 10ms max latency for V2X
        min_V2X_reliability = 0.999; % 99.9% reliability for V2X
        min_eMBB_rate = 50e6; % 50 Mbps minimum rate for eMBB
        max_mMTC_latency = 50e-3; % 50ms max latency for mMTC
        max_VoNR_latency = 20e-3; % 20ms max latency for VoNR
        min_VoNR_reliability = 0.995; % 99.5% reliability for VoNR
        
        % Simulation parameters
        sim_drops_per_round = 10; % Simulation steps per round
        sim_rounds = 10; % Number of simulation rounds
        
        % DART-PRB parameters (new)
        micro_resolution = 1e-3; % 1ms resolution for V2X/URLLC
        meso_resolution = 10e-3; % 10ms resolution for eMBB
        macro_resolution = 100e-3; % 100ms resolution for mMTC
        traffic_prediction_window = 5; % Number of rounds to predict ahead
        min_v2x_ratio = 0.2; % Minimum resource ratio for V2X
        
        % DQN parameters (new)
        batch_size = 32; % Batch size for DQN training
        replay_buffer_capacity = 10000; % Experience replay buffer size
        target_update_freq = 10; % Target network update frequency
        discount_factor = 0.95; % Discount factor for future rewards
        learning_rate = 0.001; % Learning rate for DQN
        
        % RL parameters
        alpha_learning_rate = 0.1; % RL learning rate
        Ar = 4; % Action space dimension 1
        Ax = 20; % Action space dimension 2
        nr_episode = 100; % Number of RL episodes
        tao = 1; % Temperature parameter
        Avg_T = 20; % Averaging window
        
        % WINNER II channel model parameters
        pathloss_exponent = 3.76; % Path loss exponent for urban environment
        shadow_std = 8; % Shadow fading standard deviation (dB)
        noise_floor = -174; % Noise power density (dBm/Hz)
        tx_power_bs = 43; % Base station transmit power (dBm)
        tx_power_ue = 23; % UE transmit power (dBm)
        antenna_gain_bs = 15; % Base station antenna gain (dBi)
        antenna_gain_ue = 0; % UE antenna gain (dBi)
        frequency = 2.6e9; % Carrier frequency (Hz)
        reference_distance = 1; % Reference distance for path loss model (m)
        
        % Interference management parameters (new)
        inter_slice_interference_factor = 0.1; % Interference between slices
        spatial_reuse_factor = 0.8; % Spatial reuse capability
        intra_slice_interference_factor = 0.05; % Interference within a slice
        interference_threshold = -100; % Interference threshold (dBm)
    end
end
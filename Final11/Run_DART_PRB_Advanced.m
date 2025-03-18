function Run_DART_PRB_Advanced()
    % Run script for Optimized DART-PRB implementation with enhanced PRB utilization
    % This function serves as the entry point for the enhanced DART-PRB simulation
    fprintf('Starting Optimized DART-PRB Simulation with Enhanced PRB Utilization...\n');
    
    % Call the main simulation function with default settings
    DART_PRB_Advanced_update11();
    
    fprintf('Optimized DART-PRB Simulation completed.\n');
    fprintf('Results and performance metrics have been saved.\n');
    
    % Enable scalability testing with different network sizes
    runScalabilityTests = true; % Set to true to run scalability tests
    if runScalabilityTests
        fprintf('\nRunning additional scalability tests with varying network sizes...\n');
        ScalabilityTester();
        fprintf('Scalability testing completed.\n');
        fprintf('Comprehensive complexity report has been generated.\n');
    end
    
    % Display a summary of key enhancements
    fprintf('\nKey enhancements for increased PRB utilization:\n');
    fprintf('1. Improved initial slicing ratios [0.44; 0.36; 0.20] for DL and [0.42; 0.38; 0.20] for UL\n');
    fprintf('2. Reduced PRB reservation safety margins (1.03/0.08 vs 1.05/0.1 for V2X)\n');
    fprintf('3. Increased spatial reuse factors (1.35/1.5/1.7 vs 1.2 for V2X/eMBB/mMTC)\n');
    fprintf('4. More aggressive interference thresholds (0.30/0.45/0.50 vs 0.25/0.40/0.45)\n');
    fprintf('5. Enhanced opportunistic resource sharing with higher borrowing limits\n');
    fprintf('6. Reduced fairness weights (0.15/0.25/0.35 vs 0.25/0.35/0.45) to favor utilization\n');
    fprintf('7. Lower minimum service guarantees (0.10 vs 0.15) for more allocation flexibility\n');
    fprintf('8. Higher maximum service caps (0.55/0.65 vs 0.45/0.55) for better demand adaptation\n');
    fprintf('9. Complexity and scalability analysis has been added to the system.\n');
end


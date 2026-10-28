| lang | acc | direct-answer rate | acc direct | acc w/ reasoning |
|---|---|---|---|---|
| eng | 0.92 | 0.00 | – | 0.92 |
| dan | 0.68 | 0.32 | 0.22 | 0.89 |
| nob | 0.55 | 0.52 | 0.18 | 0.93 |
| swe | 0.69 | 0.09 | 0.06 | 0.76 |
| isl | 0.32 | 0.35 | 0.17 | 0.40 |
| deu | 0.69 | 0.01 | 0.00 | 0.69 |
| fra | 0.73 | 0.00 | – | 0.73 |
| zho | 0.63 | 0.00 | – | 0.63 |

| lang | n | acc | λ mix | λ en-prior | λ da-prior | λ mismatch | AUROC err (λ) | AUROC err (entropy) |
|---|---|---|---|---|---|---|---|---|
| eng | 200 | 0.92 | -0.113 | 0.079 | -0.095 | -0.097 | 0.60 | 0.65 |
| dan | 200 | 0.68 | 0.070 | 0.005 | 0.090 | 0.095 | 0.14 | 0.92 |
| nob | 200 | 0.55 | 0.047 | 0.040 | 0.060 | 0.066 | 0.09 | 0.94 |
| swe | 200 | 0.69 | 0.056 | 0.015 | 0.068 | 0.082 | 0.51 | 0.74 |
| isl | 200 | 0.32 | 0.051 | 0.044 | 0.052 | 0.075 | 0.41 | 0.60 |
| deu | 200 | 0.69 | 0.030 | 0.009 | 0.046 | 0.045 | 0.60 | 0.66 |
| fra | 200 | 0.73 | -0.027 | 0.011 | 0.003 | -0.017 | 0.52 | 0.57 |
| zho | 200 | 0.63 | -0.058 | 0.027 | -0.024 | -0.052 | 0.46 | 0.49 |
{"n": 1600, "accuracy": 0.65, "auroc_error_mean_lambda": 0.5232400412087912, "auroc_error_mean_entropy": 0.7290539148351648}
{"n": 1344, "accuracy": 0.7395833333333334, "auroc_error_mean_lambda": 0.5430698476573728, "auroc_error_mean_entropy": 0.6867893072722047}
eng {'direct_answer_rate': 0.0, 'accuracy_direct': None, 'accuracy_reasoning': 0.915, 'auroc_error_onset_lambda': 0.415, 'auroc_error_mean_lambda_reasoning_only': 0.604, 'auroc_error_mean_entropy_reasoning_only': 0.648, 'lambda_direct': None, 'lambda_reasoning': -0.113}
dan {'direct_answer_rate': 0.32, 'accuracy_direct': 0.219, 'accuracy_reasoning': 0.89, 'auroc_error_onset_lambda': 0.141, 'auroc_error_mean_lambda_reasoning_only': 0.408, 'auroc_error_mean_entropy_reasoning_only': 0.756, 'lambda_direct': 0.014, 'lambda_reasoning': 0.096}
nob {'direct_answer_rate': 0.515, 'accuracy_direct': 0.184, 'accuracy_reasoning': 0.928, 'auroc_error_onset_lambda': 0.114, 'auroc_error_mean_lambda_reasoning_only': 0.387, 'auroc_error_mean_entropy_reasoning_only': 0.632, 'lambda_direct': 0.014, 'lambda_reasoning': 0.082}
swe {'direct_answer_rate': 0.09, 'accuracy_direct': 0.056, 'accuracy_reasoning': 0.758, 'auroc_error_onset_lambda': 0.358, 'auroc_error_mean_lambda_reasoning_only': 0.675, 'auroc_error_mean_entropy_reasoning_only': 0.664, 'lambda_direct': 0.018, 'lambda_reasoning': 0.06}
isl {'direct_answer_rate': 0.35, 'accuracy_direct': 0.171, 'accuracy_reasoning': 0.4, 'auroc_error_onset_lambda': 0.312, 'auroc_error_mean_lambda_reasoning_only': 0.591, 'auroc_error_mean_entropy_reasoning_only': 0.461, 'lambda_direct': 0.017, 'lambda_reasoning': 0.07}
deu {'direct_answer_rate': 0.005, 'accuracy_direct': 0.0, 'accuracy_reasoning': 0.693, 'auroc_error_onset_lambda': 0.321, 'auroc_error_mean_lambda_reasoning_only': 0.61, 'auroc_error_mean_entropy_reasoning_only': 0.652, 'lambda_direct': -0.02, 'lambda_reasoning': 0.03}
fra {'direct_answer_rate': 0.0, 'accuracy_direct': None, 'accuracy_reasoning': 0.73, 'auroc_error_onset_lambda': 0.435, 'auroc_error_mean_lambda_reasoning_only': 0.516, 'auroc_error_mean_entropy_reasoning_only': 0.572, 'lambda_direct': None, 'lambda_reasoning': -0.027}
zho {'direct_answer_rate': 0.0, 'accuracy_direct': None, 'accuracy_reasoning': 0.63, 'auroc_error_onset_lambda': 0.403, 'auroc_error_mean_lambda_reasoning_only': 0.457, 'auroc_error_mean_entropy_reasoning_only': 0.494, 'lambda_direct': None, 'lambda_reasoning': -0.058}
steering lam0 0.004 kl_prior [4.31, 4.14, 4.49, 4.56, 4.56, 4.55, 4.54, 4.51, 4.5, 4.46, 4.39, 4.25, 3.99] kl_random [4.57, 4.56, 4.56, 4.56, 4.56, 4.56, 4.57, 4.57, 4.57, 4.58, 4.58, 4.58, 4.57] p_orig_prior [0.09, 0.51, 0.84, 0.89, 0.9, 0.91, 0.91, 0.91, 0.9, 0.9, 0.88, 0.85, 0.76]
steering_maxent lam0 0.017 kl_prior [5.62, 5.55, 5.49, 5.35, 5.07, 4.65, 4.12, 3.53, 3.24, 3.16, 3.16, 3.19, 3.25] kl_random [5.31, 5.2, 5.12, 5.08, 5.08, 5.11, 5.17, 5.26, 5.38, 5.53, 5.72, 5.95, 6.22] p_orig_prior [0.08, 0.09, 0.1, 0.12, 0.17, 0.2, 0.17, 0.1, 0.06, 0.03, 0.02, 0.02, 0.01]

# , compare_data_statistics, compare_inputfeatures, plot_frequency_tasklabels, compare_stats_across_models
from plots import plot_decisionmaking_data_statistics, induce_condition_llm_generated_data, model_comparison_binz2022
from utils import save_real_data_openML, save_real_data_lichtenberg2017

# extract real world data
save_real_data_openML(k=4, num_points=20)
save_real_data_lichtenberg2017(k=4, num_points=20)
save_real_data_lichtenberg2017(k=4, num_points=20, method='random')
save_real_data_openML(k=4, num_points=20, method='random')

# create psuedo ranked and direction data
induce_condition_llm_generated_data(condition='ranking')
induce_condition_llm_generated_data(condition='direction')

# plot data statistics for 2D data
plot_decisionmaking_data_statistics(0, dim=2, condition='unknown')
plot_decisionmaking_data_statistics(2, dim=2, condition='real')
plot_decisionmaking_data_statistics(1, dim=2, condition='synthetic')

# plot data statistics for 4D data
plot_decisionmaking_data_statistics(0, dim=4, condition='ranked')
plot_decisionmaking_data_statistics(0, dim=4, condition='direction')
plot_decisionmaking_data_statistics(0, dim=4, condition='unknown')
plot_decisionmaking_data_statistics(
    2, dim=4, condition='openML', method='random')
plot_decisionmaking_data_statistics(
    2, dim=4, condition='lichtenberg2017', method='random')

# model simulations

# model comparison
model_comparison_binz2022(experiment_id=1)
model_comparison_binz2022(experiment_id=2)
model_comparison_binz2022(experiment_id=3)
model_comparison_binz2022(experiment_id=4)

# Demographic Inference Results

This table contains one row per X user processed by the demographic inference
pipeline. The pipeline predicts location, age, and gender from user posts,
profile text, network/interactions, and images.

Important fields:

- `user_id`: User identifier from the input CSV.
- `truth_location_raw`, `truth_age_raw`, `truth_gender_raw`: Original ground-truth labels.
- `truth_location_group`, `truth_age_group`, `truth_gender_group`: Normalized labels used for metrics.
- `pred_location_group`, `pred_age_group`, `pred_gender_group`: Normalized predictions.
- `location_correct`, `age_correct`, `gender_correct`: Whether each prediction matches ground truth when evaluable.
- `location_belief`, `age_belief`, `gender_belief`: Confidence or belief assigned to each final prediction.
- `age_abs_error_years`: Absolute error between predicted and true age-bucket midpoints.
- `dst_k`, `dst_non_kanto`, `dst_omega`: Final fused Dempster-Shafer masses for location.
- `max_conflict`: Highest pairwise conflict observed during location DST fusion.
- `posts_confidence`, `network_confidence`, `bio_confidence`, `images_confidence`: LLM confidence by location evidence source.
- `posts_kanto_mass`, `network_kanto_mass`, `bio_kanto_mass`, `images_kanto_mass`: Discounted source masses for Kanto.
- `posts_non_kanto_mass`, `network_non_kanto_mass`, `bio_non_kanto_mass`, `images_non_kanto_mass`: Discounted source masses for Non-Kanto.
- `posts_omega_mass`, `network_omega_mass`, `bio_omega_mass`, `images_omega_mass`: Discounted source uncertainty.
- `age_mass_18_24` through `age_mass_65_plus`, `age_mass_omega`: LLM age mass function.
- `gender_mass_male`, `gender_mass_female`, `gender_mass_omega`: LLM gender mass function.
- `combined_reasoning`: LLM reasoning text for location, age, and gender.

Related generated files:

- `metric_summary.csv`: Accuracy, precision, recall, F1, coverage, and age MAE.
- `confusion_matrix.csv`: Confusion cells for location, age, and gender.
- `structured_inference_output.json`: Structured JSON export used to regenerate these tables.

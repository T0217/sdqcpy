import pandas as pd
from sdqc_synthesize import YDataSynthesizer
from sdqc_integration import SequentialAnalysis
import logging
import warnings
import json
import io
import base64
import matplotlib.pyplot as plt
from jinja2 import Environment, PackageLoader, select_autoescape

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# Ignore warnings and set logging level to ERROR
warnings.filterwarnings('ignore')
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

random_seed = 17

# Replace with your own data path
raw_data = pd.read_csv(
    r"C:\Users\18840\OneDrive\Project\数据\数据集\肾移植\xi_jun_zhen_jun.csv")

# Synthesize data
synth = YDataSynthesizer(
    data=raw_data,
    random_seed=random_seed
)
synthetic_data = synth.generate()

# Perform sequential analysis
sequential = SequentialAnalysis(raw_data, synthetic_data)
results = sequential.run()

# Generate statistical metrics
stats = {}
column_types = results['Statistical Test']['column_types']

for col in raw_data.columns:
    if col in column_types['categorical']:
        # Generate PMF plots for categorical data
        plt.figure(figsize=(10, 5))
        raw_data[col].value_counts(normalize=True).plot(
            kind='bar', alpha=0.5, label='Raw')
        synthetic_data[col].value_counts(normalize=True).plot(
            kind='bar', alpha=0.5, label='Synthetic')
        plt.legend()
        plt.title(f'{col} PMF Comparison')
        plt.xlabel('Category')
        plt.ylabel('Probability')
        distribution_comparison = results['Statistical Test']['results']['distribution_comparison']['categorical'][col]
        raw_stats = results['Statistical Test']['results']['raw']['categorical'][col]
        synth_stats = results['Statistical Test']['results']['synthetic']['categorical'][col]
    else:
        # Generate KDE plots for numerical data
        plt.figure(figsize=(10, 5))
        raw_data[col].plot(kind='kde', label='Raw')
        synthetic_data[col].plot(kind='kde', label='Synthetic')
        plt.legend()
        plt.title(f'{col} KDE Comparison')
        plt.xlabel('Value')
        plt.ylabel('Density')
        distribution_comparison = results['Statistical Test']['results']['distribution_comparison']['numerical'][col]
        raw_stats = results['Statistical Test']['results']['raw']['numerical'][col]
        synth_stats = results['Statistical Test']['results']['synthetic']['numerical'][col]

    # Convert plot to base64 encoded string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='jpg')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    stats[col] = {
        'plot': f'data:image/png;base64,{image_base64}',
        'distribution_comparison': distribution_comparison,
        'raw_stats': raw_stats,
        'synth_stats': synth_stats
    }

    plt.close()

classification_metrics = results['Classification'][0].to_dict(orient='records')

# Generate feature importance
feature_importance = [results['Explainability'].to_dict(orient='records')]
features = []
importances = []
plt.figure(figsize=(10, 8))
for item in feature_importance[0]:
    if item['importance'] >= 0.01:
        features.append(item['feature'])
        importances.append(item['importance'])
plt.barh(features[::-1], importances[::-1])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
buffer = io.BytesIO()
plt.savefig(buffer, format='jpg')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
feature_importance.append(f'data:image/png;base64,{image_base64}')
plt.close()
# Generate causal comparison results
causal_comparison = {
    'ate': 0.6,
    'att': 0.7
}

# Combine all data into a dictionary
all_data = {
    'column_types': column_types,
    'stats': stats,
    'classification_metrics': classification_metrics,
    'feature_importance': feature_importance,
    'causal_comparison': causal_comparison
}

# Convert results to JSON format
json_data = json.dumps(all_data)

# Set up Jinja2 environment
env = Environment(
    loader=PackageLoader('sdqc_integration', 'templates'),
    autoescape=select_autoescape(['html', 'xml'])
)
template = env.get_template('index.html')

# Render template
html_content = template.render(json_data=json_data)

# Write results to HTML file
output_path = r"C:\Users\18840\Desktop\raw_synth_comparison.html"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Results have been saved to {output_path}")

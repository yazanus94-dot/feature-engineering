<h1>Automated Feature Engineering Engine</h1>

<h2>Overview</h2>
<p>The Automated Feature Engineering Engine is an advanced AI-powered system that automatically discovers, creates, and optimizes features for any dataset. This revolutionary framework eliminates the need for manual feature engineering by leveraging cutting-edge machine learning algorithms, statistical analysis, and domain-aware transformations to generate high-quality features that significantly enhance model performance.</p>

<p>Developed by mwasifanwar, this system represents a paradigm shift in machine learning workflows, enabling data scientists and ML engineers to focus on model architecture and business logic while the engine handles the complex task of feature creation and optimization. The framework is designed to work seamlessly with structured and unstructured data across diverse domains including finance, healthcare, e-commerce, and IoT applications.</p>

<img width="559" height="469" alt="image" src="https://github.com/user-attachments/assets/fc10d001-7e61-4c19-ba24-b954be411c69" />


<h2>System Architecture</h2>

<p>The engine follows a sophisticated multi-stage pipeline architecture that ensures robust feature generation and optimization:</p>

<pre><code>
┌─────────────────┐
│   Raw Dataset   │
└─────────────────┘
         ↓
┌─────────────────────────────────┐
│      Data Understanding         │
│  • Data Type Detection         │
│  • Statistical Profiling       │
│  • Missing Pattern Analysis    │
│  • Domain Classification       │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│   Feature Discovery Engine      │
│  • Statistical Transformations  │
│  • Domain-Specific Generators  │
│  • Interaction Detection       │
│  • Temporal Feature Mining     │
│  • Text Feature Extraction     │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  Feature Optimization Pipeline  │
│  • Importance Scoring          │
│  • Stability Analysis          │
│  • Redundancy Elimination      │
│  • Multi-objective Selection   │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│  Feature Validation Framework   │
│  • Cross-validation Performance │
│  • Statistical Significance    │
│  • Business Logic Validation   │
│  • Production Readiness Check  │
└─────────────────────────────────┘
         ↓
┌─────────────────┐
│ Optimized Features │
└─────────────────┘
</code></pre>

<img width="1472" height="569" alt="image" src="https://github.com/user-attachments/assets/b9c7ffba-ff7d-4bff-a49e-f2169b6cb169" />


<h3>Core Processing Stages</h3>
<ul>
  <li><strong>Data Profiling</strong>: Comprehensive analysis of data types, distributions, and quality metrics</li>
  <li><strong>Feature Generation</strong>: Multi-modal feature creation including statistical, temporal, and text-based features</li>
  <li><strong>Intelligent Selection</strong>: Advanced feature selection using multi-criteria optimization</li>
  <li><strong>Quality Assurance</strong>: Rigorous validation ensuring feature stability and performance</li>
</ul>

<h2>Technical Stack</h2>

<h3>Core Frameworks & Libraries</h3>
<ul>
  <li><strong>Python 3.8+</strong>: Primary programming language with type hints and modern syntax</li>
  <li><strong>Pandas & NumPy</strong>: High-performance data manipulation and numerical computing</li>
  <li><strong>Scikit-learn 1.0+</strong>: Machine learning algorithms and model evaluation</li>
  <li><strong>SciPy</strong>: Advanced statistical functions and scientific computing</li>
</ul>

<h3>Specialized Components</h3>
<ul>
  <li><strong>Feature-engine</strong>: Production-ready feature transformers and engineering utilities</li>
  <li><strong>Optuna</strong>: Hyperparameter optimization and automated tuning</li>
  <li><strong>TSFresh</strong>: Automated time series feature extraction</li>
  <li><strong>Category Encoders</strong>: Advanced categorical variable encoding techniques</li>
</ul>

<h3>Development & Deployment</h3>
<ul>
  <li><strong>Jupyter</strong>: Interactive development and experimentation</li>
  <li><strong>Pytest</strong>: Comprehensive testing framework</li>
  <li><strong>Docker</strong>: Containerized deployment and environment management</li>
  <li><strong>MLflow</strong>: Experiment tracking and model management</li>
</ul>

<h2>Mathematical Foundation</h2>

<h3>Feature Importance Scoring</h3>
<p>The engine employs multiple importance metrics to evaluate feature relevance:</p>

<p><strong>Permutation Importance</strong> measures feature significance by evaluating performance degradation when feature values are randomized:</p>
<p>$I_j = \frac{1}{K} \sum_{k=1}^K \left( \mathcal{L} - \mathcal{L}_{\pi_j^{(k)}} \right)$</p>
<p>where $\mathcal{L}$ is the baseline loss and $\mathcal{L}_{\pi_j^{(k)}}$ is the loss with feature $j$ permuted in the $k$-th iteration.</p>

<p><strong>SHAP-based Importance</strong> leverages Shapley values from cooperative game theory:</p>
<p>$\phi_j = \frac{1}{N} \sum_{i=1}^N |\phi_j(x_i)|$</p>
<p>where $\phi_j(x_i)$ represents the SHAP value for feature $j$ on sample $x_i$, providing theoretically grounded feature attribution.</p>

<h3>Multi-objective Feature Selection</h3>
<p>The framework formulates feature selection as a multi-objective optimization problem:</p>
<p>$\max \, F(S) = \left[ f_1(S), f_2(S), -f_3(S) \right]$</p>
<p>where:
<ul>
  <li>$f_1(S)$: Predictive performance of feature subset $S$</li>
  <li>$f_2(S)$: Aggregate feature importance scores</li>
  <li>$f_3(S)$: Cardinality of feature subset (minimization)</li>
</ul>
</p>

<h3>Feature Stability Analysis</h3>
<p>Stability across data splits is quantified using consistency metrics:</p>
<p>$S_j = 1 - \frac{\sigma_{I_j}}{\mu_{I_j}}$</p>
<p>where $\sigma_{I_j}$ and $\mu_{I_j}$ represent the standard deviation and mean of importance scores across cross-validation folds, ensuring robust feature selection.</p>

<h3>Information-theoretic Feature Ranking</h3>
<p>Mutual information based feature relevance:</p>
<p>$I(X_j; Y) = \sum_{x_j \in X_j} \sum_{y \in Y} p(x_j, y) \log \frac{p(x_j, y)}{p(x_j)p(y)}$</p>
<p>This measures the dependency between feature $X_j$ and target variable $Y$, enabling effective feature filtering.</p>

<h2>Features</h2>

<h3>Automated Feature Generation</h3>
<ul>
  <li><strong>Statistical Feature Engineering</strong>: Automated generation of mean, variance, skewness, kurtosis, quantiles, and higher-order moments</li>
  <li><strong>Cross-feature Interactions</strong>: Intelligent detection and creation of product, ratio, polynomial, and combinatorial features</li>
  <li><strong>Temporal Feature Extraction</strong>: Advanced time-series features including lags, rolling statistics, seasonal decomposition, and Fourier transformations</li>
  <li><strong>Text Feature Engineering</strong>: Comprehensive NLP features including TF-IDF, word embeddings, semantic analysis, and sentiment scoring</li>
  <li><strong>Categorical Encoding</strong>: Multiple encoding strategies including target encoding, frequency encoding, and neural embedding-based approaches</li>
</ul>

<h3>Intelligent Feature Selection</h3>
<ul>
  <li><strong>Multi-criteria Optimization</strong>: Simultaneous optimization of importance, stability, and redundancy metrics</li>
  <li><strong>Genetic Algorithm Selection</strong>: Evolutionary computation for optimal feature subset discovery</li>
  <li><strong>Stability-driven Selection</strong>: Cross-validation consistency analysis for robust feature choice</li>
  <li><strong>Domain Adaptation</strong>: Transfer learning techniques for feature relevance across domains</li>
</ul>

<h3>Advanced Capabilities</h3>
<ul>
  <li><strong>AutoML Integration</strong>: Seamless compatibility with popular AutoML frameworks including AutoSklearn and H2O.ai</li>
  <li><strong>Real-time Feature Engineering</strong>: Streaming data support with incremental feature generation</li>
  <li><strong>Feature Store Compatibility</strong>: Native integration with feature stores for production deployment</li>
  <li><strong>Explainable AI</strong>: Transparent feature generation process with comprehensive documentation</li>
  <li><strong>Multi-modal Data Support</strong>: Unified handling of tabular, time-series, text, and image data</li>
</ul>

<img width="747" height="509" alt="image" src="https://github.com/user-attachments/assets/0adc02eb-7400-4448-bd92-efc82755e68c" />


<h2>Installation</h2>

<h3>System Requirements</h3>
<ul>
  <li>Python 3.8 or higher</li>
  <li>4GB RAM minimum (16GB recommended for large datasets)</li>
  <li>1GB free disk space</li>
  <li>Internet connection for package dependencies</li>
</ul>

<h3>Basic Installation</h3>
<pre><code>
# Clone the repository
git clone https://github.com/yazanus94-dot/feature-engineering.git
cd feature-engineering

# Create and activate virtual environment
python -m venv autofe_env
source autofe_env/bin/activate  # On Windows: autofe_env\Scripts\activate

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Verify installation
python -c "from core import AutomatedFeatureEngine; print('Engine successfully installed!')"
</code></pre>

<h3>Development Installation</h3>
<pre><code>
# Development installation with testing tools
pip install -e ".[dev]"
</code></pre>

<h3>Docker Installation</h3>
<pre><code>
# Build the Docker image
docker build -t autofe-engine .

# Run with GPU support
docker run --gpus all -p 8888:8888 -v $(pwd)/data:/app/data autofe-engine

# Run with CPU only
docker run -p 8888:8888 -v $(pwd)/data:/app/data autofe-engine
</code></pre>

<h2>Usage / Running the Project</h2>

<h3>Basic Feature Engineering Pipeline</h3>
<pre><code>
import pandas as pd
import numpy as np
from core import AutomatedFeatureEngine

# Load your dataset
data = pd.read_csv('your_dataset.csv')
target_column = 'price'

# Initialize the feature engine with basic configuration
engine = AutomatedFeatureEngine(
    target_column=target_column,
    task_type='regression',  # 'regression', 'classification', or 'auto'
)

# Generate features automatically
feature_matrix = engine.fit_transform(data)

# Access feature metadata and importance scores
feature_metadata = engine.get_feature_metadata()

print(f"Original features: {len(data.columns)}")
print(f"Selected features: {len(feature_matrix.columns)}")
</code></pre>

<h3>Advanced Configuration Example</h3>
<pre><code>
from core import AutomatedFeatureEngine
from config import FeatureConfig

# Custom configuration for complex use cases
config = FeatureConfig(
    max_features=200,
    feature_interactions=True,
    polynomial_degree=3,
    temporal_features=True,
    text_features=True,
    feature_selection_method='multi_objective',
    validation_strategy='time_series_split',
    stability_threshold=0.85,
    correlation_threshold=0.90
)

# Initialize engine with custom configuration
engine = AutomatedFeatureEngine(
    target_column='sales',
    task_type='regression',
    config=config.to_dict()
)

# Transform data with the fitted engine
transformed_data = engine.transform(data)
</code></pre>

<h3>Command Line Interface</h3>
<pre><code>
# Basic demo execution
python main.py --mode demo

# Process a CSV dataset
python main.py --mode process --input data/sales_data.csv --target revenue --output features/engineered_features.csv

# Show CLI options
python main.py --help
</code></pre>

<h2>Configuration / Parameters</h2>

The implemented configuration is defined by `FeatureConfig` in `config/feature_config.py`.

<h3>Implemented Configuration Fields</h3>
<ul>
  <li><code>max_features</code> - Maximum number of features to retain</li>
  <li><code>correlation_threshold</code> - Maximum allowed correlation between features</li>
  <li><code>importance_threshold</code> - Minimum feature importance score for retention</li>
  <li><code>stability_threshold</code> - Minimum stability score across data splits</li>
  <li><code>cv_folds</code> - Number of cross-validation folds</li>
  <li><code>feature_interactions</code> - Enable interaction feature generation</li>
  <li><code>polynomial_degree</code> - Polynomial feature degree</li>
  <li><code>temporal_features</code> - Enable temporal feature generation</li>
  <li><code>text_features</code> - Enable text feature generation</li>
  <li><code>categorical_features</code> - Enable categorical feature generation</li>
  <li><code>feature_selection_method</code> - Feature selection strategy</li>
  <li><code>validation_strategy</code> - Validation strategy</li>
  <li><code>optimization_metric</code> - Optimization metric</li>
</ul>

<h2>Folder Structure</h2>

<pre><code>
feature-engineering/
├── .github/
│   └── workflows/
│       └── tests.yml
├── ATTRIBUTION.md
├── README.md
├── main.py
├── requirements.txt
├── setup.py
├── config/
│   ├── __init__.py
│   └── feature_config.py
├── core/
│   ├── __init__.py
│   ├── feature_discoverer.py
│   ├── feature_engine.py
│   ├── feature_optimizer.py
│   └── feature_validator.py
├── examples/
│   ├── __init__.py
│   ├── advanced_usage.py
│   └── basic_usage.py
├── pipelines/
│   ├── __init__.py
│   └── feature_pipeline.py
├── tests/
│   └── test_feature_engine.py
├── transformers/
│   ├── __init__.py
│   ├── interaction_transformers.py
│   ├── statistical_transformers.py
│   ├── temporal_transformers.py
│   └── text_transformers.py
└── utils/
    ├── __init__.py
    ├── data_utils.py
    └── validation_utils.py

<h2>Results / Experiments / Evaluation</h2>

<h3>Implemented and Tested Functionality</h3>
<ul>
  <li>Feature engine initialization and configuration.</li>
  <li>Statistical, interaction, transformation, temporal, categorical, and text feature discovery names.</li>
  <li>Feature optimization and validation workflow.</li>
  <li>Basic synthetic-dataset pipeline execution.</li>
</ul>

<h3>Planned or Aspirational Functionality</h3>
<p>The README previously described benchmark results, scalability claims, and advanced deployment features that are not verified by the current test suite. Those claims have been removed.</p>

<h2>References / Citations</h2>

<ol>
  <li>Kanter, J. M., & Veeramachaneni, K. (2015). "Deep Feature Synthesis: Towards Automating Data Science Endeavors." IEEE International Conference on Data Science and Advanced Analytics.</li>
  <li>Chen, J., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.</li>
  <li>Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." Journal of Machine Learning Research.</li>
  <li>Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." Advances in Neural Information Processing Systems.</li>
  <li>Kursa, M. B., & Rudnicki, W. R. (2010). "Feature Selection with the Boruta Package." Journal of Statistical Software.</li>
  <li>Christ, M., et al. (2018). "Time Series Feature Extraction on basis of Scalable Hypothesis tests." Neurocomputing.</li>
  <li>Micci-Barreca, D. (2001). "A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems." ACM SIGKDD Explorations Newsletter.</li>
  <li>Guyon, I., & Elisseeff, A. (2003). "An Introduction to Variable and Feature Selection." Journal of Machine Learning Research.</li>
</ol>

<h2>Acknowledgements</h2>

<p>This project builds upon decades of research in machine learning, feature engineering, and automated machine learning. We extend our gratitude to the open-source community and the following resources that made this project possible:</p>

<ul>
  <li><strong>Scikit-learn Community</strong>: For providing the foundational machine learning algorithms and utilities</li>
  <li><strong>Pandas & NumPy Teams</strong>: For enabling efficient data manipulation and numerical computing</li>
  <li><strong>Academic Researchers</strong>: Whose pioneering work in feature selection and engineering informed our approaches</li>
  <li><strong>Open Source Contributors</strong>: Whose libraries and tools facilitated rapid development and testing</li>
</ul>

<br>

## Attribution

This repository is a fork of the original project by M Wasif Anwar:
https://github.com/mwasifanwar/feature-engineering

See [ATTRIBUTION.md](ATTRIBUTION.md) for authorship and license notes.

<div align="center">

### ⭐ Don't forget to star this repository if you find it helpful!

</div>

<p>This Automated Feature Engineering Engine represents a significant advancement in machine learning automation, empowering organizations to extract maximum value from their data with minimal manual intervention.</p>

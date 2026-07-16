# Exoplanet Transit Detection System

A full-stack web application for automated detection and analysis of exoplanet transits using NASA's Kepler mission data.

## Project Overview

This application implements a machine learning pipeline to detect periodic dimming patterns in stellar light curves—the telltale signature of exoplanets orbiting distant stars. By analyzing photometric data from the Kepler Space Telescope, the system identifies candidate transits, ranks them by scientific merit, and provides actionable insights for astronomical research.

**Live Demo:** [Deploy to Render](#deployment)

## Scientific Methodology

### Algorithm Pipeline

1. **Data Acquisition**
   - Fetches long-cadence (LLC) FITS files from NASA's Kepler archive
   - Supports random sampling across 15+ mission quarters for data diversity
   - Validates file integrity and column structure

2. **Light Curve Detrending**
   - Applies median filtering to remove instrumental noise and stellar variability
   - Preserves authentic transit signals while eliminating false positives
   - Adaptive kernel sizing (configurable 5-25 day windows)

3. **Transit Detection**
   - Identifies periodic dips in detrended light curves using peak detection
   - Configurable prominence thresholds to control sensitivity
   - Filters spurious detections by minimum peak spacing

4. **Period Estimation**
   - Calculates spacing between candidate transit times
   - Computes coefficient of variation (CV) to assess periodicity stability
   - Classifies stability: Stable (CV ≤ 0.15), Moderate (CV ≤ 0.40), Unstable (CV > 0.40)

5. **Candidate Ranking**
   - Scores based on: number of transits, period stability, transit depth consistency, quality metrics
   - Machine-learned weighting to prioritize scientifically viable candidates
   - Generates human-readable explanations for each score

## Technology Stack

### Backend
- **Flask 2.3.3** - Python web framework
- **SQLAlchemy 2.0.51** - ORM for database management
- **Gunicorn 26.0** - Production WSGI server

### Data Science
- **NumPy 2.4.6** - Numerical computing
- **Pandas 3.0.3** - Data manipulation and analysis
- **SciPy 1.17.1** - Scientific algorithms (signal processing, statistics)
- **Astropy 8.0.1** - FITS file handling and astronomical utilities

### Frontend
- **Jinja2** - Template rendering
- **HTML5/CSS3** - Responsive dark-themed UI
- **JavaScript** - Interactive filtering and pagination

### Deployment
- **Render** - Cloud platform with Python 3.11 runtime
- **SQLite** - Ephemeral database (production-ready for multi-database support)
- **GitHub** - Version control and CI/CD integration

## Features

### Web Interface
- **Pipeline Dashboard** - Configure detection parameters in real-time
- **Results Visualization** - Interactive table with filtering by review status
- **CSV Export** - Download full analysis results for external processing
- **Run History** - Track all pipeline executions with timestamps and statistics
- **Progress Tracking** - Real-time status updates for long-running jobs

### Algorithm Configuration
- **Kernel Size**: 5-25 day median filter window
- **Prominence**: Detection sensitivity (0.01-1.0 normalized flux units)
- **Min/Max Peaks**: Transit count constraints (2-50 transits)
- **Quality Thresholds**: Customizable scoring weights
- **Data Source**: Archive or local file upload

### Data Diversity
- Randomized bucket selection across Kepler archive (0001-0013 mission quarters)
- Variable targets per bucket (2-8 random selection)
- Limited files per target (1-3) for KIC ID variety
- Reproducible with seed-based randomization

## Results Example

```
Targets Processed:    10
Success Rate:         100%
Total Candidates:     1,100+
Avg Stability:        38% Moderate+ (stable or moderate period)
```

| Target | Transits | Stability | Period (days) | Mean Depth | Score |
|--------|----------|-----------|---------------|-----------|-------|
| kplr00435612... | 52 | Moderate | 0.061 | 0.0004 | 52 |
| kplr00432662... | 48 | Stable | 0.089 | 0.0005 | 48 |

## Local Development

### Prerequisites
- Python 3.9+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/Immabeatyou/exoplanet-transit-detector-automation.git
cd Exoplanet_Transit_Research

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python -m flask run
# Visit http://127.0.0.1:5000
```

### Running Batch Analysis
```bash
python batch_analyze.py --targets 50 --kernel-size 10 --prominence 0.05
```

## Deployment

### Deploy to Render (Free Tier)

1. **Connect GitHub Repository**
   - Go to https://dashboard.render.com
   - Click "New +" → "Web Service"
   - Connect your GitHub repo

2. **Configure Environment**
   - Python Version: 3.11
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn -w 1 -b 0.0.0.0:10000 web.app:app`
   - Add Environment Variable: `PYTHON_VERSION=3.11`

3. **Deploy**
   - Click "Create Web Service"
   - Monitor logs until startup succeeds
   - Access at provided URL (e.g., `https://exoplanet-transit-detector.onrender.com`)

## Project Structure

```
.
├── web/
│   ├── app.py              # Flask application & routes
│   ├── config.py           # Environment-specific configuration
│   ├── models.py           # SQLAlchemy ORM models
│   ├── pipeline.py         # Core detection algorithm
│   ├── templates/          # HTML templates
│   └── static/             # CSS & JavaScript
├── batch_analyze.py        # Command-line batch processing
├── transit_analyzer.py     # Analysis utilities
├── requirements.txt        # Python dependencies
├── render.yaml            # Render deployment config
└── README.md              # This file
```

## Algorithm Details

### Period Stability Assessment
Stable periods indicate genuine exoplanet signals rather than noise:
- **Stable** (CV ≤ 0.15): High-confidence candidates, ~15% of population
- **Moderate** (CV ≤ 0.40): Viable candidates, ~50% of population  
- **Unstable** (CV > 0.40): Requires manual review, ~35% of population

### Ranking Score Formula
```
Score = num_transits × weight_transits 
       + stability_bonus × weight_stability
       + depth_consistency × weight_depth
       + quality_metrics × weight_quality
```

Weights are optimized to balance statistical significance with astrophysical realism.

## Educational Value

This project demonstrates:
- **Scientific Computing**: Signal processing on real astronomical data
- **Full-Stack Development**: Backend (Python/Flask), database (SQLAlchemy), frontend (HTML/CSS)
- **Software Engineering**: Modular design, error handling, configuration management
- **Data Analysis**: Statistical metrics, filtering, ranking algorithms
- **Cloud Deployment**: Production server configuration, environment management
- **Version Control**: Git workflow, GitHub integration, CI/CD

## Relevant Concepts

- **Signal Processing**: Median filtering, peak detection, time-series analysis
- **Statistics**: Coefficient of variation, mean/median/max calculations
- **Astronomy**: Kepler mission, light curves, transit photometry, exoplanet detection
- **Web Development**: REST APIs, database modeling, form handling
- **DevOps**: Docker containerization, environment variables, production debugging

## References

- [NASA Kepler Mission](https://science.nasa.gov/mission/kepler/)
- [Kepler Archive Data](https://archive.stsci.edu/kepler/data_search/)
- [Transit Method Explanation](https://en.wikipedia.org/wiki/Transit_photometry)
- [Exoplanet Detection Techniques](https://www.nasa.gov/exoplanets/what-are-exoplanets-and-how-do-we-find-them)

## License

This project is provided as-is for educational purposes.

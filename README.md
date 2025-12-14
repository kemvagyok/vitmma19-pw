# Deep Learning Class (VITMMA19) Project Work template

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Benedek Sagi
- **Aiming for +1 Mark**: Yes

### Solution Description

This project focuses on the analysis and classification of foot positions. The problem involves recognizing and categorizing different foot postures based on input data, which can be images or sensor readings. For this task, a convolutional neural network (CNN) architecture was chosen, as CNNs are particularly effective at extracting spatial features from visual data.

The training methodology included standard preprocessing steps such as normalization and resizing of input data, followed by supervised training using a labeled dataset of foot positions. Ordinal distance loss  was used for optimization, and the model was trained for multiple epochs with early stopping to prevent overfitting.

The results show that the model is able to accurately classify foot positions into the predefined categories, demonstrating its potential for applications in biomechanics, sports science, and ergonomic studies.

### Extra Credit Justification

Yes:  
- Implemented data cleansing and preparation functions.  
- Developed incremental model building process.
- I creatd an own loss function (Ordinal distance loss).
- Conducted advanced evaluation using Ordinal MAE, defined evaluation criteria for foot position classification, including metrics and acceptable error thresholds.
- Created a GUI frontend and deployed the ML model as a backend service.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.
[Adjust the commands that show how do build your container and run it with log output.]

#### Build, Run
##### AI Service
Run the following command in the root directory of the repository to build the Docker image of AI service, and run the container:
To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.
**To capture the logs for submission (required), redirect the output to a file :** (log/run.log 2>&1)

```bash
docker-compose run -v  /absolute/path/to/your/local/data:/app/data ai  > log/run.log 2>&1
```
*   Replace `/absolute/path/to/your/local/data` with the actual path to your dataset on your host machine that meets the [Data preparation requirements](#data-preparation).
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).
*   And the inference stage the AI service will be as a backend.
##### GUI Service
Run the following command in the root directory of the repository to build the Docker image of web service, and run the container:
#### Run
```bash
docker-compose run web
```
*    The web service listens on port 5000 inside the container and is mapped to port 8080 on the host.
*    The web service depends on the ai service, which runs in its own container.
*    The ai service can be accessed from the web container at http://ai:5000/.
*    The web service is available from the host at: http://[IP_ADDRESS]:8080/ (usually http://localhost:8080/)


### File Structure and Functions

[Update according to the final file structure.]

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs), settings and paths.
    - `utils.py` – General helper functions used across scripts, e.g.:
        - Logging setup
    
    - `metricsUtils.py` – Functions for model evaluation and metrics, e.g.:
        - Accuracy, precision, recall calculations
        - Confusion matrix generation
        - Custom metric functions
    
    - `dataUtils.py` – Data-related utilities, e.g.:
        - Dataset loading and splitting
        - Feature encoding and normalization
    
    - `models.py` – Model-related utilities, e.g.:
        - Model architectures (CNN, MLP, etc.)

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.
    - `03-incremental_development.ipynb`: Notebook for incremental model development.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- - **`gui/`**: Contains web service files.
    - `Dockerfile`: Configuration file for building the Docker image of web service with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required of AI service for the project.
    - - **`src/`**: Contains (Flask) Python, and HTML/CSS files.
        - `main.py`: It is responsible for running frotend.
        - `templates/`: Contains HTML/CSS files
- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image of AI service with the necessary environment and dependencies.
    - `docker-compose.yaml`: Two services – `web` (ports 8080→5000) and `ai` (GPU-enabled). `web` depends on `ai`. Both build from their respective directories.
    - `requirements.txt`: List of Python dependencies required of AI service for the project.
    - `README.md`: Project documentation and instructions.

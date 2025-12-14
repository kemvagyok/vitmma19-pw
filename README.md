# Deep Learning Class (VITMMA19) Project Work template

[Complete the missing parts and delete the instruction parts before uploading.]

## Submission Instructions

[Delete this entire section after reading and following the instructions.]

### Project Levels

**Basic Level (for signature)**
*   Containerization
*   Data acquisition and analysis
*   Data preparation
*   Baseline (reference) model
*   Model development
*   Basic evaluation

**Outstanding Level (aiming for +1 mark)**
*   Containerization
*   Data acquisition and analysis
*   Data cleansing and preparation
*   Defining evaluation criteria
*   Baseline (reference) model
*   Incremental model development
*   Advanced evaluation
*   ML as a service (backend) with GUI frontend
*   Creative ideas, well-developed solutions, and exceptional performance can also earn an extra grade (+1 mark).

### Data Preparation

**Important:** You must provide a script (or at least a precise description) of how to convert the raw database into a format that can be processed by the scripts.
* The scripts should ideally download the data from there or process it directly from the current sharepoint location.
* Or if you do partly manual preparation, then it is recommended to upload the prepared data format to a shared folder and access from there.

[Describe the data preparation process here]

### Logging Requirements

The training process must produce a log file that captures the following essential information for grading:

1.  **Configuration**: Print the hyperparameters used (e.g., number of epochs, batch size, learning rate).
2.  **Data Processing**: Confirm successful data loading and preprocessing steps.
3.  **Model Architecture**: A summary of the model structure with the number of parameters (trainable and non-trainable).
4.  **Training Progress**: Log the loss and accuracy (or other relevant metrics) for each epoch.
5.  **Validation**: Log validation metrics at the end of each epoch or at specified intervals.
6.  **Final Evaluation**: Result of the evaluation on the test set (e.g., final accuracy, MAE, F1-score, confusion matrix).

The log file must be uploaded to `log/run.log` to the repository. The logs must be easy to understand and self explanatory. 
Ensure that `src/utils.py` is used to configure the logger so that output is directed to stdout (which Docker captures).

### Submission Checklist

Before submitting your project, ensure you have completed the following steps.
**Please note that the submission can only be accepted if these minimum requirements are met.**

- [X ] **Project Information**: Filled out the "Project Information" section (Topic, Name, Extra Credit).
- [ ] **Solution Description**: Provided a clear description of your solution, model, and methodology.
- [ ] **Extra Credit**: If aiming for +1 mark, filled out the justification section.
- [ ] **Data Preparation**: Included a script or precise description for data preparation.
- [ ] **Dependencies**: Updated `requirements.txt` with all necessary packages and specific versions.
- [ ] **Configuration**: Used `src/config.py` for hyperparameters and paths, contains at least the number of epochs configuration variable.
- [ ] **Logging**:
    - [ ] Log uploaded to `log/run.log`
    - [ ] Log contains: Hyperparameters, Data preparation and loading confirmation, Model architecture, Training metrics (loss/acc per epoch), Validation metrics, Final evaluation results, Inference results.
- [ ] **Docker**:
    - [ ] `Dockerfile` is adapted to your project needs.
    - [ ] Image builds successfully (`docker build -t dl-project .`).
    - [ ] Container runs successfully with data mounted (`docker run ...`).
    - [ ] The container executes the full pipeline (preprocessing, training, evaluation).
- [ ] **Cleanup**:
    - [ ] Removed unused files.
    - [ ] **Deleted this "Submission Instructions" section from the README.**

## Project Details

### Project Information

- **Selected Topic**: AnkleAlign
- **Student Name**: Benedek Sag
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
##### AI
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
##### GUI
Run the following command in the root directory of the repository to build the Docker image of web service, and run the container:
#### Run
```bash
docker-compose run ai
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

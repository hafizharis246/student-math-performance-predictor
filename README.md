# Student Math Performance Predictor 📚

A machine learning project that predicts student math scores based on various features using a trained model and provides an interactive web interface through Streamlit.

## Project Overview

This project consists of two main components:
1. A Jupyter notebook for data analysis and model training
2. A Streamlit web application for making predictions

### Features
- Data analysis and visualization of student performance data
- Machine learning model training using scikit-learn
- Interactive web interface for making predictions
- Real-time score analysis and grade prediction

## Project Structure

```
.
├── app.py                      # Streamlit web application
├── Student Math Performance Indicator.ipynb  # Data analysis and model training notebook
├── model.pkl                   # Trained model pickle file
├── preprocessor.pkl            # Preprocessor pickle file
├── stud.csv                    # Dataset
├── requirements.txt            # Project dependencies
├── setup.sh                    # Setup script
├── Procfile                    # Heroku deployment configuration
└── .gitignore                  # Git ignore file
```

## Dataset Information

The dataset (`stud.csv`) contains the following features:
- `gender`: Sex of students (male/female)
- `race_ethnicity`: Ethnicity of students (Group A, B, C, D, E)
- `parental_level_of_education`: Parents' final education level
- `lunch`: Type of lunch (standard/free/reduced)
- `test_preparation_course`: Whether test preparation course was completed
- `math_score`: Target variable (score to predict)
- `reading_score`: Reading test score
- `writing_score`: Writing test score

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Student-Math-Performance-Predictor
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. The web interface will open in your default browser. You can then:
   - Select student information using the dropdown menus
   - Enter reading and writing scores
   - Click "Predict Math Score" to get the prediction
   - View the predicted score and performance analysis

## Model Details

The model is trained using scikit-learn and includes:
- Data preprocessing for categorical variables
- Feature engineering
- Model training and evaluation
- Performance metrics analysis

## Dependencies

- streamlit
- pandas
- numpy
- scikit-learn

## Deployment

The application is configured for deployment on Heroku using the provided `Procfile` and `setup.sh`.

## Contributing

Feel free to submit issues and enhancement requests!

## Author

Hafiz Haris Mehmood - harismehmood948@gmaill.com



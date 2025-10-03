# Build a Reproducible Model Workflow - Exercises

This repo contains the code for demos, exercises, and exercise solutions.

This repository organizes the code by the lessons that they are used in. Each set of code is located in their respective lessons.

Please note that certain instructions for each exercise, as well as any relevant environment setup, are only provided within the Udacity classroom.

## Example:
All lesson 2 files are in `/lesson-2-data-exploration-and-preparation/`.

This directory contains: `demo`, `exercises`, with the `exercises` directory organized by the exercise number, and therein containing an exercise `README.md` file and `starter` and `solution` directories.

## Course Info
This course empowers the students to be more efficient, effective, and productive in modern, real-world ML projects by adopting best practices around reproducible workflows. In particular, it teaches the fundamentals of MLops and how to: a) create a clean, organized, reproducible, end-to-end machine learning pipeline from scratch using MLflow b) clean and validate the data using pytest c) track experiments, code, and results using GitHub and Weights & Biases d) select the best-performing model for production and e) deploy a model using MLflow. Along the way, it also touches on other technologies like Kubernetes, Kubeflow, and Great Expectations and how they relate to the content of the class.

## Exercises 
### Lesson 1 - Machine learning pipelienes
* Exercise 1 - Write a script that uploads an artifact to Weights & Biases
* Exercise 2 - Build first MLflow component
* Exercise 3 - Build first MLflow pipeline by connecting two components

### Lesson 2 - Data exploration and preparation
* Exercise 4 - Perform a simple Exploratory Data Analysis in Jupyter keeping track of your progress with W&B
* Exercise 5 - Create a MLflow component that preprocess the data
* Exercise 6 - Complete a MLflow component that divides the data into training and test sample

### Lesson 3 - Data validation
* Exercise 7 - Apply deterministic tests to the cleaned dataset
* Exercise 8 - Apply non-deterministic tests to the cleaned dataset
* Exercise 9 - Modify the non-deterministic test

### Lesson 4 - Training, validation and experiment tracking
* Exercise 10 - Wite an inference pipeline
* Exercise 11 - Explore hydra_options
* Exercise 12 - Export a model
* Exercise 13 - Build a component that fetches a model and test it on the test dataset

### Lesson 5 - Final pipeline, release and deploy
* Exercise 14 - Bring everything together in a complete ML pipeline that produces a trained Random Forest model
* Exercise 15 - Release your final pipeline as a versioned code artifact on GitHub
* Exercise 16 - Experiment with different ways of deploying the exported model for online and offline inference
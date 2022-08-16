# mlops_projector_test_project
This is a test project for Projector course dedicated to Machine Learning in production course. 
The project is properly deployed to Heroku. Endpoint for testing - https://mlops-projector-test-project.herokuapp.com/docs.

Under the hood there is a dumb model (Ridge regression that is built upon TfIdf text vectorizer) that rates the complexity of reading passages for grade 3-12 classroom use (taken from this competition: https://www.kaggle.com/competitions/commonlitreadabilityprize/overview)  

## Steps to test:
### Stage 1:
1. Clone the project from this repository.
2. Install requirements from requirements.txt (preferrably in a dedicated project virtual env).
3. Uncomment last 2 lines to run locally. 

Note: If you clear up the _results_ folder, it might take some time to preprocess train dataset. 

### Stage 2:
1. Clone the project from this repository.
2. Run the following command: _docker build -t registry.heroku.com/mlops-projector-test-project/web ._
3. Run the following command: _docker run -d --name mycontainer -p 80:80 registry.heroku.com/mlops-projector-test-project/web_
4. Naviate to browser: localhost:80/docs - a convenient FastAPI documentation for /predict method testing will be opened.

### Stage 3 (deploying):
1. Clone the project from this repository.
2. Comment out last line in Dockerfile.
2. Login to Heroku CLI.
3. Run the following command: _docker build -t registry.heroku.com/mlops-projector-test-project/web ._
4. Run the following command: _docker push registry.heroku.com/mlops-projector-test-project_
5. Run the following command: _heroku container:release --app mlops-projector-test-project web_
6. Run the following command: _heroku open --app mlops-projector-test-project_

import os 
import mlflow 
from common_params import model_dir

TAGS = ["NB_CLASS", "NB_IMG_PER_CLASS"]
NAME = ["experiment_name"]
def save_experiment_mlflow(model, model_run_config, run_name, history): 

    experiment_name         =  model_run_config["experiment_name"]

    # Set the experiment name and create an MLflow run
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name = run_name) as mlflow_run:
        
        for tag in TAGS : 
            mlflow.set_tag(tag, model_run_config[tag])
 
        # mlflow.log_param("learning_rate", learning_rate)
        for key in model_run_config.keys() : 
            if key not in TAGS and  key not in NAME: 
                mlflow.log_param(key, model_run_config[key])


        mlflow.log_metric("train_loss", history.history["loss"][-1])
        mlflow.log_metric("train_acc", history.history["accuracy"][-1])
        mlflow.log_metric("val_loss", history.history["val_loss"][-1])
        mlflow.log_metric("val_acc", history.history["val_accuracy"][-1])

        
        mlflow_run_id = mlflow_run.info.run_id
        print("MLFlow Run ID: ", mlflow_run_id)
        
        # log model 
        save_dir = os.path.join(model_dir, mlflow_run_id)
        mlflow.keras.log_model(model, save_dir, keras_module='tensorflow.keras')
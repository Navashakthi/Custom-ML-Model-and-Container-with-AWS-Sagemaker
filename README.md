# Custom-ML-Model-and-Container-with-AWS-Sagemaker
Creating own Custom model and docker image with the train and serve scripts with AWS Sagemaker Notebooks and triggering model training and serving part from gitlab pipeline.

## Architecture
![workflow4](https://github.com/user-attachments/assets/6d50728a-1faf-4f31-8ab6-8dad75afde0f)


The workflow represents training and deploying a custom Convolutional Neural Network (CNN) model using AWS services and Docker, integrated with a CI/CD pipeline managed by GitLab. Here's a breakdown of each step in the workflow:

### **1. Training Custom Model with SageMaker Notebook**
- **Starting Point**: The process begins with developing and training a custom CNN model using a SageMaker notebook. This environment allows you to prototype and fine-tune the model.
- **Building Docker Image**: A Docker image of the training environment is created. This image encapsulates all the dependencies and libraries needed to train the model, ensuring that the training process can be consistently reproduced.

### **2. Data and Model Storage in S3**
- **S3 Training Data**: The training data used by the model is stored in an S3 bucket. This data is pulled into the SageMaker environment during the training process.
- **S3 Model Artifacts**: After training, the model artifacts (including the trained model weights and configurations) are saved back to an S3 bucket. These artifacts are essential for the next steps of the deployment process.

### **3. SageMaker Training Job**
- **Training with Custom Image**: A SageMaker training job is initiated using the custom Docker image and the training data stored in S3. This job runs the model training process in a scalable and managed environment provided by SageMaker.
- **Model Registration**: Once the model is trained, it is registered in the SageMaker Model Registry. This registry serves as a central repository where different versions of models are stored and managed, facilitating model versioning and deployment.

### **4. Docker Image for Prediction**
- **Building Prediction Docker Image**: A new Docker image is built specifically for serving the model predictions. This image includes the trained model and the necessary serving pipeline (e.g., Flask application) to handle incoming inference requests.
- **ECR Registry**: The Docker image for prediction is pushed to the Amazon Elastic Container Registry (ECR). ECR serves as a scalable container image repository where Docker images can be stored securely.

### **5. ECS Deployment**
- **Pulling Docker Image**: The Docker image from ECR is pulled onto an Amazon Elastic Container Service (ECS) instance for deployment. ECS orchestrates the deployment of Docker containers in a scalable manner.
- **Deploying in ECS Cluster**: The model is deployed as a service within an ECS cluster. ECS manages the lifecycle of the containers, ensuring that the model serving application remains available and responsive.

### **6. API Endpoint for Model Serving**
- **Serving the Model**: Once the model is deployed in the ECS cluster, an API endpoint is exposed to the internet. This endpoint allows users or applications to send image data for classification and receive predictions from the CNN model.

### **7. Integration with GitLab**
- **CI/CD Pipeline**: The entire workflow is integrated with a GitLab CI/CD pipeline, which automates the process of training, building Docker images, and deploying the model. This ensures continuous integration and deployment, allowing for rapid iteration and deployment of model updates.

### **Summary**
This workflow demonstrates a comprehensive and automated approach to building, training, and deploying a custom CNN model using AWS services, Docker, and GitLab CI/CD. By leveraging these tools, the workflow ensures scalability, reproducibility, and efficient management of the model lifecycle, from development to production deployment.

## Prerequisites
Before running this pipeline, ensure you have the following prerequisites:

- **AWS Account**: An active AWS account with permissions to use SageMaker, S3, and IAM services.
- **S3 Bucket:** An S3 bucket where training data and model artifacts will be stored.
- **IAM Role:** An IAM role with the necessary permissions for SageMaker to access S3 and ECR.
- **AWS CLI:** AWS CLI configured with your access and secret keys, and the region set.
- **GitLab CI/CD:** A GitLab project with CI/CD enabled and the environment variables set mentioned below.

## Pipeline Stages

The pipeline consists of the following stages:

1. **train_model**: Triggers a SageMaker training job using the algorithm's Docker image and monitors the job's status using create_training_job.py script.
2. **register_model**: Registers the trained model in SageMaker's model registry.
3. **docker_build**: Builds the serving Docker image with the trained model and pushes it to GitLab's image registry.
4. **docker_push**: Pushes the serving Docker image from GitLab to AWS ECR.
5. **deploy_task**: Deploys an ECS task definition using the serving Docker image.
6. **deploy_service**: Deploys an ECS service that runs the task definition created in the previous stage.
7. **inference_endpoint**: Fetches the inference endpoint created by the ECS service for model inference.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/project-name.git
   cd project-name
   ```

2. **Install the necessary Python packages:**
   ```bash
   pip install boto3 sagemaker tensorflow
   ```

3. **Ensure Docker and AWS CLI are installed and configured on your machine:**
   ```bash
   aws configure
   ```
4. **Review the `.gitlab-ci.yml` File**:
   - Verify the pipeline stages and commands in the `.gitlab-ci.yml` file.
   - Make sure the paths and commands match your project structure and requirements.

5. **Push to GitLab**:
   - Commit and push your changes to GitLab:
     ```bash
     git add .
     git commit -m "Initial commit"
     git push origin main
     ```

6. **Monitor the Pipeline**:
   - Navigate to your GitLab project and go to **CI/CD** > **Pipelines**.
   - Monitor the progress of your pipeline stages: preprocess, train, build, config_deploy, deploy, predict, and cleanup.

7. **Verify the Results**:
   - Once the pipeline execution is complete, verify that the ECS endpoint is created and the inferences are successful.
   - Check that the resources are deleted after the cleanup stage to ensure cost savings.

By following these steps, you can execute the entire CI/CD pipeline to automate the training and deployment of a CNN model using Amazon SageMaker with GitLab CI/CD.


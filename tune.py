#!/usr/bin/python3

import os
import yaml

# I need to perform hyperparameter tuning on the model, whose parameters are in the config.yaml file.
# I need to open the file, change the parameters in the file and save the file, then run the script named run.py
def tune():
    # Open the file
    lines = list()
    with open("config.yaml", "r") as fh:
        yfh = yaml.safe_load(fh)

    learning_rates = [10**(-i) for i in range(6)]
    learners = ["adam", "RMSprop"]
    models = ["NeuCMFii", "NeuCMFww", "NeuCMF0i", "NeuCMFi0", "NeuCMF0w", "NeuCMFw0"]
    embedding_sizes = [32, 64, 128, 25, 512]
    weight_decays = [0.0, 0.001, 0.01, 0.1]
    train_batch_sizes = [500, 1000, 2000, 5000, 10000]

    for learning_rate in learning_rates:
        for learner in learners:
            for model in models:
                for embedding_size in embedding_sizes:
                    for weight_decay in weight_decays:
                        for train_batch_size in train_batch_sizes:
                            # Change the parameters
                            print(f"| learning_rate | {learning_rate} |")
                            print(f"| learner | {learner} |")
                            print(f"| model | {model} |")
                            print(f"| embedding_size | {embedding_size} |")
                            print(f"| weight_decay | {weight_decay} |")
                            print(f"| train_batch_size | {train_batch_size} |")
                            print()

                            yfh["learning_rate"] = learning_rate
                            yfh["learner"] = learner
                            yfh["model"] = model
                            yfh["embedding_size"] = embedding_size
                            yfh["weight_decay"] = weight_decay
                            yfh["train_batch_size"] = train_batch_size

                            # Save the file
                            with open("config.yaml", "w") as fh:
                                yaml.safe_dump(yfh, fh)

                            # Run the script
                            os.system("python3 run.py")

if __name__ == "__main__":
    tune()
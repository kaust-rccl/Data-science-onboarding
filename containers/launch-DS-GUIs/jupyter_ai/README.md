### Building the Docker Image

To build the Docker image, use the following command:

```sh
docker build -t $USER/jupyter_ai_custom --platform linux/amd64 --no-cache .
```

This command will create a Docker image named `jupyter_ai_custom` using the specified platform and without using any cached layers.

### Running the Docker Container

To run the Docker container, use the following command:

```sh
docker run --rm -d -p 11434:11434 -p 8888:8888 --platform linux/amd64 --name llama $USER/jupyter_ai_custom
```

This command will start a container named `llama` from the `jupyter_ai_custom` image, mapping ports 11434 and 8888 from the container to the host.

### Accessing the Jupyter Lab Interface

To access the Jupyter Lab interface within the running container, use the following command:

```sh
docker exec -it llama /bin/bash -c "conda run --no-capture-output -n jupyter_ai jupyter lab --ip=0.0.0.0 --allow-root"
```

This command will open a bash shell in the `llama` container and start the Jupyter Lab server, making it accessible from any IP address and allowing root access.


### Stopping the Docker Container
```sh
docker stop llama
```

This command will stop the `llama` container.
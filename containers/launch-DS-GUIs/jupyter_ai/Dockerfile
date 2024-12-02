# Use the Miniforge image
# FROM condaforge/miniforge3
# User ollama image
FROM ollama/ollama

# This is ubuntu so let's update the package list
RUN apt-get update && \
    apt-get -y --no-install-recommends install git curl wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install the conda package manager
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh 
RUN bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy the environment.yml file into the container
COPY environment.yml .

# Install the dependencies from the environment.yml file
RUN conda env create -f environment.yml

# Activate the environment
SHELL ["conda", "run", "-n", "jupyter_ai", "/bin/bash", "-c"]

# Clean up the cache
RUN conda clean --all -f -y

# # Install Ollama locally
# RUN curl -fsSL https://ollama.com/install.sh | sh

# Expose the port for Jupyter Lab
EXPOSE 8888

# # start the Ollama server in the background
# SHELL ["ollama", "serve", "&" ]

# # Create a user and group for Ollama:
# RUN useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
# RUN usermod -a -G ollama $(whoami)

# # Create a service file in /etc/systemd/system/ollama.service
# COPY ollama.service /etc/systemd/system/ollama.service

# # Copy the Ollama service script
# COPY ollama.sh /usr/local/bin/ollama.sh

# # Make the service script executable
# RUN chmod +x /usr/local/bin/ollama.sh

# use ollama to pull the images
# RUN ollama pull qwen2.5:1.5b
# RUN ollama pull llama3.2:3b

# Set the default command to run Jupyter Lab
# CMD ["conda", "run", "--no-capture-output", "-n", "jupyter_ai", "jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
FROM python:3.12

# Set the working directory
WORKDIR /code

# Copy and install Python dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy application files
COPY . .

# Expose required ports
EXPOSE 7860
EXPOSE 8000

# Install supervisor for process management
RUN apt-get update && apt-get install -y supervisor && apt-get clean

# Copy supervisor configuration
COPY ./supervisord.conf /etc/supervisor/supervisord.conf

# Start supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]

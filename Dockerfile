FROM ubuntu:22.04

# ARGs
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG TARGETPLATFORM=linux/amd64,linux/arm64
ARG DEBIAN_FRONTEND=noninteractive

# neo4j 2025.08.0 installation (match GDS v2.21.0) and some cleanup
RUN apt-get update && \
    apt-get install -y wget gnupg software-properties-common && \
    wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add - && \
    echo 'deb https://debian.neo4j.com stable latest' > /etc/apt/sources.list.d/neo4j.list && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y nano unzip neo4j=1:2025.08.0 python3-pip && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* 

ENV uri="your_neo4j_uri"
ENV neo4j_pass="your_neo4j_password"
ENV openai="your_OPENAI_api_key"

WORKDIR /Knowledge Graph/
COPY doc_loader.py .
COPY graph.py .
COPY requirements.txt .
COPY main.py .
COPY AttentionPaper.pdf .
COPY Neo4j-Credentials.txt .
RUN pip install -r requirements.txt && \
    echo "Required packages installed and dataset ready." && \
    wget -P /var/lib/neo4j/plugins/ "https://github.com/neo4j/graph-data-science/releases/download/2.22.0/neo4j-graph-data-science-2.22.0.jar"
RUN echo "dbms.security.allow_csv_import_from_file_urls=true" >> /etc/neo4j/neo4j.conf && \
    echo "dbms.security.procedures.allowlist=apoc.*,gds.*" >> /etc/neo4j/neo4j.conf && \
    echo "dbms.security.procedures.unrestricted=apoc.*,gds.*" >> /etc/neo4j/neo4j.conf && \
    echo "server.default_listen_address=0.0.0.0" >> /etc/neo4j/neo4j.conf

# Run the data loader script
RUN neo4j start & \
    echo "Waiting for Neo4j to start..." 

# Expose neo4j ports
EXPOSE 7474 7687

# Start neo4j service and show the logs on container run
CMD ["/bin/bash", "-c", "neo4j start && tail -f /dev/null"]
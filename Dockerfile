FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y wget gnupg software-properties-common && \
    wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add - && \
    echo 'deb https://debian.neo4j.com stable latest' > /etc/apt/sources.list.d/neo4j.list && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y nano unzip neo4j=1:2025.08.0 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN echo "dbms.security.allow_csv_import_from_file_urls=true" >> /etc/neo4j/neo4j.conf && \
    echo "dbms.security.procedures.allowlist=apoc.*,gds.*" >> /etc/neo4j/neo4j.conf && \
    echo "dbms.security.procedures.unrestricted=apoc.*,gds.*" >> /etc/neo4j/neo4j.conf && \
    echo "server.default_listen_address=0.0.0.0" >> /etc/neo4j/neo4j.conf

RUN wget -P /var/lib/neo4j/plugins/ "https://github.com/neo4j/graph-data-science/releases/download/2026.04.0/neo4j-graph-data-science-2026.04.0.jar"

# Expose neo4j ports
EXPOSE 7474 7687

# Start neo4j service
CMD ["/bin/bash", "-c", "neo4j start && tail -f /dev/null"]
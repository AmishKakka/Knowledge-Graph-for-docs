# Knowledge-Graph-for-Docs

A **Knowledge Graph** is a way to organize information by representing real-world objects (entities) and the relationships between them, often adding semantic meaning and context.

It is also a nice way to visulalize your data as a Graph. Like you can have Uber's trip data represented as a knowledge graph.

**Nodes** --> Drop-off locations, Pickup locations

**Relationships** --> Distance between locations

This way we represent a **large structured dataset**. We can query this graph to get insights like which is the most visited location or the longest distance between any 2 locations.
Queries which would take time processing, if using a SQL query.

## Knowledge Graph for unstructred data (text, images, audio,..)
So, here we use this idea of Knowledge Graph to represent documents which can have text and images. But, for now we focues only on textual data.

1. A graph has nodes and edges connecting them. In our case these nodes are chunks of text taken from the document with some metadata like - page no, chunkId, file-name, and vector embeddings for the chunks of text.

2. The edges (relationships) between these nodes is your way of telling how and why these nodes are connected to each other. 

Now, this can be - 
precedence between nodes based on chunkId, highlighting thematic or contextual connections between chunks or between keywords and chunks, and more.

3. After creating the graph, we can query it. Vectorize the query, do a similarity search using this on our graph. For each similar node, we also look at its neighbours using the relationships we created. 

4. Fetch these results and pass them to the model to generate your response.


## **Installation**
1. Cloning the repository.
```sh 
git clone https://github.com/AmishKakka/Knowledge-Graph-for-docs.git
```

2. We are using [Neo4j AuraDB](https://login.neo4j.com/u/login/identifier?state=hKFo2SBidm1uRlNXaUdLZVdMUmlzRGdiUkJpTGp6c3FhRWJCZ6Fur3VuaXZlcnNhbC1sb2dpbqN0aWTZIHdfYlhSeHBHSVZVa3IyakhOenFSUEJhQW9hSzFjT2Q3o2NpZNkgRXZ2MmNjWFBjOHVPeGV3bzBJalkyMFlJckg3VmtKVzk) (**1 free instance when you create a new account**) for our graph. When creating your AuraDB instance you would have a credentials file to download which contains the URI, username, password you need to connect to your instance through code.

3. Use the dockerfile to create an image of this project and then run it from your terminal or from Docker desktop. First ensure your Docker Desktop is up and running.

But before creating an image, you need to change the values for - 

```sh
    # This is the uri link that you will find in your Neo4j Credentials document downloaded earlier
    ENV uri="your_neo4j_uri"
    # Your unique password to connect to your instance
    ENV neo4j_pass="your_neo4j_password"
    # Your OpenAI API key needed to access an embedding model
    ENV openai="your_OPENAI_api_key"
```

```sh
docker build your_image_name .

docker run -d -p 7474:7474 -p 7687:7687 --name instance_name your_image_name 
```

4. Now, you can run an interactive terminal for this container or just use the Docker Desktop to add, modify and run code.
This lets you **access the instance from your terminal**.

```sh
docker exec -it instance_name /bin/bash  
```

5.  Explore the code

```sh
# Connecting with your Aura graph db
python3 graph.py

# Get you document, split into chunks of text with metadata.
python3 doc_loader.py

# Runs the flow created to load, split documents and add the nodes to the graph, then querying it
python3 main.py
```

**Pointers**

1. Will update code such that you can add multiple documents at a time dynamically at run time.

2. Work on querying the document through terminal.
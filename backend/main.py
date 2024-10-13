from fastapi import FastAPI, HTTPException, logger
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import requests
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from threading import Timer

app = FastAPI()

# CORS setup to allow requests from any origin (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API key
OPENAI_API_KEY = "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"
openai.api_key = OPENAI_API_KEY

# JIRA configuration
JIRA_SERVER = 'https://jiradigi.pxl.be'
JIRA_API_TOKEN = 'NTUzMTk3NDU2MzM4Okg8QixJbKxR+Pxpn4xM0qnOZVch'  # Replace with your actual Bearer token

# Pinecone configuration
PINECONE_API_KEY = '880004dc-1edc-4e4e-bc50-373825e84cdd'  # Replace with your Pinecone API key

# Create an instance of the Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create an index if it doesn't already exist
index_name = "jira-tickets1"

# Check if the index already exists
print(f"Checking if index '{index_name}' exists...")
existing_indexes = pc.list_indexes()  # Get the list of existing indexes

# Extract index names for easier comparison
existing_index_names = [index['name'] for index in existing_indexes]

if index_name in existing_index_names:
    print(f"Index '{index_name}' already exists. No need to create it.")
else:
    print(f"Creating index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,  # Change dimension based on the model you are using
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  # Change to your preferred region
        )
    )

# Get the index for storing ticket embeddings
index = pc.Index(index_name)

# Model to receive the question
class QuestionRequest(BaseModel):
    question: str

# Fetch JIRA tickets
def fetch_jira_tickets():
    headers = {
        "Authorization": f"Bearer {JIRA_API_TOKEN}",
    }
    
    url = f"{JIRA_SERVER}/rest/api/2/search?jql=project=IT2425T05"  # Adjust JQL as necessary
    print(f"Fetching JIRA tickets from: {url}")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("JIRA tickets fetched successfully.")
        tickets = response.json().get('issues', [])
        print(f"Fetched {len(tickets)} tickets: {tickets}")  # Debugging line
        return tickets
    else:
        print(f"Failed to fetch JIRA tickets: {response.status_code} - {response.json()}")
        raise HTTPException(status_code=response.status_code, detail=response.json().get("errors", "Unknown error"))
    
@app.get("/jira-tickets")
async def get_jira_tickets():
    try:
        jira_tickets = fetch_jira_tickets()
        # Limit the response to the first 5 tickets
        limited_tickets = jira_tickets[:5]
        return {"tickets": [{"key": ticket["key"], "summary": ticket["fields"]["summary"]} for ticket in limited_tickets]}
    except Exception as e:
        print(f"Error fetching JIRA tickets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Store tickets in Pinecone if they do not already exist
def store_tickets_in_vector_db(tickets):
    print(f"Storing {len(tickets)} JIRA tickets in Pinecone...")
    
    # Fetch existing ticket keys to avoid duplicates
    existing_ticket_keys = [vector['id'] for vector in index.fetch(ids=[ticket['key'] for ticket in tickets])['vectors']]

    for ticket in tickets:
        # Print the ticket for debugging
        print(f"Processing ticket: {ticket}")  # Debugging line

        # Ensure ticket is a dictionary and has the expected keys
        if isinstance(ticket, dict) and 'key' in ticket and 'fields' in ticket:
            ticket_key = ticket['key']
            summary = ticket['fields'].get("summary", "No summary available")  # Use get to avoid KeyError
            
            if ticket_key not in existing_ticket_keys:
                print(f"Creating embedding for ticket {ticket_key}: {summary}")
                embedding = openai.Embedding.create(input=summary, model="text-embedding-ada-002")["data"][0]["embedding"]
                
                # Store the embedding in Pinecone
                index.upsert(vectors=[(ticket_key, embedding)])
                print(f"Stored ticket {ticket_key} in Pinecone.")
            else:
                print(f"Ticket {ticket_key} already exists in Pinecone. Skipping.")
        else:
            print(f"Invalid ticket format: {ticket}")  # This line will catch any unexpected formats
# Normalize the vector
def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        print("Warning: Attempting to normalize a zero vector.")
        return vec  # Avoid division by zero
    return vec / norm

# Background task to fetch and store JIRA tickets every hour
def fetch_and_store_tickets():
    print("Fetching and storing JIRA tickets...")
    try:
        jira_tickets = fetch_jira_tickets()
        store_tickets_in_vector_db(jira_tickets)
        print("JIRA tickets stored successfully.")
    except Exception as e:
        print(f"Error storing JIRA tickets: {str(e)}")

# Schedule the background task every hour
def schedule_tickets_fetch():
    fetch_and_store_tickets()
    Timer(3600, schedule_tickets_fetch).start()  # Call this function every hour

# Start the scheduled task
schedule_tickets_fetch()

# Endpoint to fetch JIRA tickets and store in Pinecone
@app.get("/store-jira-tickets")
async def store_jira_tickets():
    try:
        jira_tickets = fetch_jira_tickets()
        store_tickets_in_vector_db(jira_tickets)
        return {"message": "JIRA tickets stored successfully."}
    except Exception as e:
        print(f"Error storing JIRA tickets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to handle questions
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    logger.info(f"Received question: {request.question}")
    
    try:
        # Generate embedding for the user question
        question_embedding = openai.Embedding.create(input=request.question, model="text-embedding-ada-002")["data"][0]["embedding"]
        question_embedding = normalize_vector(question_embedding)
        logger.info(f"Normalized question embedding: {question_embedding[:5]}...")

        # Query Pinecone for the most relevant tickets
        logger.info("Querying Pinecone for relevant tickets...")
        query_results = index.query(queries=[question_embedding], top_k=5)

        # Collect ticket keys and summaries
        ticket_keys = [result['id'] for result in query_results['matches']]
        ticket_summaries = []
        
        logger.info(f"Found ticket keys: {ticket_keys}")
        for key in ticket_keys:
            try:
                ticket_details = next(ticket for ticket in fetch_jira_tickets() if ticket['key'] == key)
                ticket_summaries.append(f"Ticket {key}: {ticket_details['fields']['summary']}")
                logger.info(f"Retrieved details for ticket {key}.")
            except StopIteration:
                logger.warning(f"Ticket {key} not found in JIRA.")
                continue

        # Generate a human-like response using ChatGPT
        logger.info("Generating response using ChatGPT...")
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Here are the relevant tickets: {ticket_summaries}. Can you summarize?"}],
            max_tokens=150
        )

        answer = gpt_response['choices'][0]['message']['content']
        logger.info(f"ChatGPT response: {answer}")
        return {"answer": answer}
    
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return {"error": "An error occurred while processing your request."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)










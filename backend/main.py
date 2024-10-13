from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import requests
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from threading import Timer
import logging 

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
index_name = "jira-tickets01"

# Check if the index already exists
print(f"Checking if index '{index_name}' exists...")
existing_indexes = pc.list_indexes()  # Get the list of existing indexes
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

# Fetch all JIRA tickets by project
def fetch_all_jira_tickets():
    headers = {
        "Authorization": f"Bearer {JIRA_API_TOKEN}",
    }
    
    url = f"{JIRA_SERVER}/rest/api/2/search"
    params = {
        "jql": "project = IT2425T05",  # Modify this JQL query as needed
        "fields": "key,summary,status",
        "maxResults": 1000,  # Adjust as needed (max is usually 1000)
    }
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        print("Fetched JIRA tickets successfully.")
        return response.json().get("issues", [])
    else:
        print(f"Failed to fetch JIRA tickets: {response.status_code} - {response.json()}")
        raise HTTPException(status_code=response.status_code, detail=response.json().get("errors", "Unknown error"))

# Fetch a single JIRA ticket by key
def fetch_jira_ticket(ticket_key):
    headers = {
        "Authorization": f"Bearer {JIRA_API_TOKEN}",
    }
    
    url = f"{JIRA_SERVER}/rest/api/2/issue/{ticket_key}"  # Fetch the ticket by its key
    print(f"Fetching JIRA ticket from: {url}")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        print("JIRA ticket fetched successfully.")
        return response.json()  # Return the ticket details
    else:
        print(f"Failed to fetch JIRA ticket: {response.status_code} - {response.json()}")
        raise HTTPException(status_code=response.status_code, detail=response.json().get("errors", "Unknown error"))

# Store a single ticket in Pinecone if it does not already exist
def store_ticket_in_vector_db(ticket):
    ticket_key = ticket['key']
    summary = ticket['fields'].get("summary", "No summary available")

    # Print the ticket for debugging
    print(f"Processing ticket: {ticket}")

    if isinstance(ticket, dict) and 'key' in ticket and 'fields' in ticket:
        if ticket_key not in index.fetch(ids=[ticket_key])['vectors']:
            print(f"Creating embedding for ticket {ticket_key}: {summary}")
            embedding = openai.Embedding.create(input=summary, model="text-embedding-ada-002")["data"][0]["embedding"]
            index.upsert(vectors=[(ticket_key, embedding)])
            print(f"Stored ticket {ticket_key} in Pinecone.")
        else:
            print(f"Ticket {ticket_key} already exists in Pinecone. Skipping.")
    else:
        print(f"Invalid ticket format: {ticket}")

# Fetch and store all JIRA tickets
def fetch_and_store_all_tickets():
    print("Fetching and storing all JIRA tickets...")
    try:
        tickets = fetch_all_jira_tickets()
        for ticket in tickets:
            store_ticket_in_vector_db(ticket)
        print("All JIRA tickets stored successfully.")
    except Exception as e:
        print(f"Error storing JIRA tickets: {str(e)}")

# Background task to fetch and store all JIRA tickets every hour
def schedule_ticket_fetch():
    fetch_and_store_all_tickets()
    Timer(3600, schedule_ticket_fetch).start()  # Call this function every hour

# Start the scheduled task
schedule_ticket_fetch()

# Endpoint to fetch and store all JIRA tickets
@app.get("/store-all-jira-tickets")
async def store_all_jira_tickets():
    try:
        fetch_and_store_all_tickets()
        return {"message": "All JIRA tickets stored successfully."}
    except Exception as e:
        print(f"Error storing JIRA tickets: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to handle questions
class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    print(f"Received question: {request.question}")

    try:
        # Step 1: Generate embedding for the user question
        print("Generating embedding for the question...")
        question_embedding = openai.Embedding.create(
            input=request.question,
            model="text-embedding-ada-002"
        )["data"][0]["embedding"]

        # Normalize the embedding
        print("Normalizing question embedding...")
        question_embedding = normalize_vector(question_embedding)

        # Step 2: Query Pinecone for the most relevant tickets
        print("Querying Pinecone for relevant tickets...")
        query_results = index.query(vector=question_embedding.tolist(), top_k=5)

        # Step 3: Collect ticket keys and summaries
        ticket_keys = [result['id'] for result in query_results['matches']]
        ticket_summaries = []

        print(f"Found ticket keys: {ticket_keys}")
        for key in ticket_keys:
            try:
                print(f"Fetching details for ticket {key}...")
                ticket_details = fetch_jira_ticket(key)  # Fetch details for each ticket
                ticket_summaries.append(f"Ticket {key}: {ticket_details['fields']['summary']}")
                print(f"Retrieved details for ticket {key}.")
            except Exception as e:
                print(f"Could not retrieve details for ticket {key}: {str(e)}")
                continue

        # Step 5: Generate a human-like response using ChatGPT
        print("Generating response using ChatGPT...")
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
             messages=[{"role": "user", "content": f"Hier zijn de relevante tickets: {ticket_summaries}. Kun je samenvatten?"}],
            max_tokens=150
        )

        answer = gpt_response['choices'][0]['message']['content']
        print(f"ChatGPT response: {answer}")
        return {"answer": answer}

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {"error": "An error occurred while processing your request."}

# Normalize the vector
def normalize_vector(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        print("Warning: Attempting to normalize a zero vector.")
        return vec  # Avoid division by zero
    return vec / norm
@app.get("/jira-tickets")
async def get_jira_tickets():
    try:
        # Fetch all JIRA tickets from the JIRA API
        print("Fetching all JIRA tickets from JIRA API...")
        url = f"{JIRA_SERVER}/rest/api/2/search"  # Endpoint for searching issues
        headers = {
            "Authorization": f"Bearer {JIRA_API_TOKEN}",
        }
        
        # You can add JQL or other parameters here to filter tickets if needed
        params = {
            "jql": "project = IT2425T05",  # Modify this JQL query as per your requirements
            "fields": "key,summary,status",  # Specify the fields you want to retrieve
            "maxResults": 5,  # Specify the maximum number of results to fetch
        }

        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            tickets_data = response.json()
            tickets_list = []
            for issue in tickets_data.get("issues", []):
                tickets_list.append({
                    "id": issue["key"],
                    "summary": issue["fields"]["summary"],
                    "status": issue["fields"]["status"]["name"],  # Get the status of the issue
                })

            print("Fetched JIRA tickets successfully.")
            return {"tickets": tickets_list}
        else:
            print(f"Failed to fetch JIRA tickets: {response.status_code} - {response.json()}")
            raise HTTPException(status_code=response.status_code, detail=response.json().get("errors", "Unknown error"))

    except Exception as e:
        print(f"Error fetching JIRA tickets: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch JIRA tickets")


# Endpoint to handle questions
class QuestionRequest(BaseModel):
    question: str


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)











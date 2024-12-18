# BNB-cod--vipasana
 
Revolutionizing PDF Interaction with Gemini 2.0 : Smarter Ways to Comprehend Documents
Introduction

PDFs are indispensable for students, offering a reliable format for study materials that preserve layouts, support multimedia, and enable secure sharing. With features like annotations and bookmarks, they're ideal for eBooks, research papers, and notes. However, their static nature can make lengthy documents harder to navigate, slowing comprehension in today's information-heavy world.
This post will show you how to revolutionize your interaction with PDFs - making it more intuitive and efficient. Say goodbye to tedious searches and static content as we explore smarter, interactive ways to engage with your documents.
Design
Several solutions exist to address this challenge, such as crafting prompts to add document context. However, models often hallucinate when answering complex queries like, "Does the document sufficiently cover topic X to answer question Y?"
To mitigate this, responses can be grounded in the document's context using a Retrieval-Augmented Generation (RAG) approach, significantly reducing the likelihood of inaccuracies.
In this tutorial, we will demonstrate this process using a Google Colab notebook.
here is the full code : 
Prerequisites
Proficiency in Python
Familiarity with the Google Colab environment

Tech stack
Google cloud platform.
Vertex AI Studio.
Gemini 2.0 flash.
Helper functions : https://github.com/lavinigam-gcp/generative-ai/blob/main/gemini/use-cases/retrieval-augmented-generation/utils/intro_multimodal_rag_utils.py
PyPDF

instructions
Setup and requirements
Install Vertex AI SDK for Python and other dependencies
Run the following four cells below before you get to Task 1. Be sure to add your current project ID to the cell titled Define Google Cloud project information.
!pip3 install --upgrade --user google-cloud-aiplatform
!pip3 install --upgrade --user google-cloud-aiplatform pymupdf
pip install colorama
Restart current runtime
You must restart the runtime in order to use the newly installed packages in this Jupyter runtime. You can do this by running the cell below, which will restart the current kernel.
# "RUN THIS CELL AS IS"

import IPython

# Restart the kernet after libraries are loaded.

app = IPython.Application.instance()
app.kernel.do_shutdown(True)
Define Google Cloud project information
# "COMPLETE THE MISSING PART AND RUN THIS CELL"
import sys

# Define project information and update the location if it differs from the one specified in the lab instructions.

PROJECT_ID = "qwiklabs-gcp-00-0c81695f67f8"  # @param {type:"string"}
LOCATION = "us-east4"  # @param {type:"string"}

# Try to get the PROJECT_ID automatically.
if "google.colab" not in sys.modules:
    import subprocess
    PROJECT_ID = subprocess.check_output(
        ["gcloud", "config", "get-value", "project"], text=True
    ).strip()
print(f"Your project ID is: {PROJECT_ID}")

Your project ID is: <bnb-code-vipasana>
Initialize Vertex AI
Initialize the Vertex AI SDK for Python for your project:
# "RUN THIS CELL AS IS"

# Initialize Vertex AI.

import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)

import google.generativeai as genai

from IPython.display import Markdown, display
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    HarmCategory,
    HarmBlockThreshold,
    Image,
    Part,
)

text_model = GenerativeModel("gemini-1.5-pro")
multimodal_model = GenerativeModel("gemini-2.0-flash-exp")
Importing helper functions 
import os
import urllib.request
import sys

if not os.path.exists("utils"):
    os.makedirs("utils")

url_prefix = "https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/use-cases/retrieval-augmented-generation/utils"
files = ["intro_multimodal_rag_utils.py"]

for fname in files:
    urllib.request.urlretrieve(f"{url_prefix}/{fname}", filename=f"utils/{fname}")

# Import helper functions from utils.

from utils.intro_multimodal_rag_utils import get_document_metadata
from utils.intro_multimodal_rag_utils import get_user_query_text_embeddings
from utils.intro_multimodal_rag_utils import get_similar_text_from_query
from utils.intro_multimodal_rag_utils import print_text_to_text_citation
from utils.intro_multimodal_rag_utils import get_gemini_response
Giving PDF as input and creating prompt for images
Here for example a prehistory text book is given as input.
ref : https://web.ung.edu/media/university-press/World%20History%20Textbook-082817.pdf?t=1510261063109

pdf_folder_path = "/content/History"
image_description_prompt = """Explain what is going on in the image.
If it's a table, extract all elements of the table.
If it's a map , extract all elements of the map.
If it's a graph, explain the findings in the graph.
Do not consider any white spaces as the image.
ignore the image if the image is not related to the document.
ignore the image if the image is only containing numbers.
"""
Creating Data frames from PDF 
creating text embeddings and image embeddings

text_metadata_df, image_metadata_df = get_document_metadata(
    multimodal_model, # we are passing gemini 1.0 pro vision model
    pdf_folder_path,
    image_save_dir="images",
    image_description_prompt=image_description_prompt,
    embedding_size=1408,
    add_sleep_after_page = True,
    sleep_time_after_page = 60
)

print("\n\n --- Completed processing. ---")
inspect the dataframes

text_metadata_dfimage_metadata_dfValidation
test_question = "when humans invented fire?"
question_emb = get_user_query_text_embeddings(user_query=test_question)

matching_results_chunks_data = get_similar_text_from_query(
    query=test_question,
    text_metadata_df=text_metadata_df,
    column_name="text_embedding_chunk",
    top_n = 1,
    chunk_text=True,
    print_citation = False)

print_text_to_text_citation(
    matching_results_chunks_data,
    print_top=False,
    chunk_text=True
)
# Create an empty list named "context_text". This list will be used to store the combined chunks of text.
context_text = list()

for key, value in matching_results_chunks_data.items():
    context_text.append(value["chunk_text"])

final_context_text = "\n".join(context_text)

# engineering prompt for contextual answer
prompt = f""" Instructions: Compare the images and the text provided as Context: to answer multiple Question:
Make sure to think thoroughly before answering the question and put the necessary steps to arrive at the answer in bullet points for easy explainability.
If unsure, respond, "Not enough context to answer".

Context:
 - Text Context:
 {final_context_text}

{test_question}

Answer:
"""
Let's get the contextual response now.
Markdown(
    get_gemini_response(
        multimodal_model,
        model_input=[prompt],
        stream=True,
        generation_config=GenerationConfig(temperature=0.2, max_output_tokens=2048),
    )
)
Response from Gemini: 
Okay, let's break down the question and the context to arrive at the answer.
Identify the key information: The question asks when humans invented fire.
Locate relevant information in the text: The text states, "Fire was another important tool, first used by Homo erectus about 1.6 million years ago."
Synthesize the information: The text directly provides the answer to the question.

Answer: According to the text, Homo erectus first used fire about 1.6 million years ago.
What's next?
This is Just the tip of the iceberg for what can be achieved. few of the possibilities are as follows.
Test assessments
Notes and revision assistance
Validation of documents

Suggestions
https://cloud.google.com/vertex-ai/generative-ai/docs/cookbook
https://www.cloudskillsboost.google/paths/183

Call to action
To learn more about Google Cloud services and to create impact for the work you do, get around to these steps right away:
Register for Code Vipassana sessions
Join the meetup group Datapreneur Social
Sign up to become Google Cloud Innovator

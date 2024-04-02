# LLM - RAG with Claude for Snapchat User Review Insights

This is a personal proof-of-concept project that uses a Retrieval-Augmented Generation (RAG) Large Language Model (LLM) implementation to derive insights from Snapchat user reviews. Specifically, this uses Claude models (V2 and Sonnet) via Amazon Bedrock, Amazon Titan Embeddings, and Meta's Facebook AI Similarity Search (FAISS) vector store to classify user reviews (i.e., positive, neutral, or negative), extract named products referenced for product tagging, and succinctly summarizing each review.

Disclaimer: This work is in a personal capacity unrelated to my employer. It relies only on publicly available resources. This package is for illustrative purposes and is not designed for end-to-end productionization as-is. Also consider that this is an extremely small sample size of user reviews and should therefore not be perceived as representative of the average user experience. For example, the reviews publicly available per the methodology below are apparently mostly negative, whereas the Apple App Store rating of 4.6/5.0 demonstrates that most reviews are positive.

## Prerequisites

You will need your own Amazon Web Services (AWS) account with Claude and Titan Amazon Bedrock model access. Your Python environment will also require:
- langchain>=0.1.11
- langchain-community
- faiss-cpu==1.8.0

## Overview

This package will demonstrate how to:
- Import libraries
- Instantiate the LLM and embeddings models
- Load 10 Snapchat reviews as documents
- Split documents into chunks
- Confirm embeddings functionality
- Create vector store
- Embed question and return relevant chunks
- Create RAG prompt template
- Produce RAG outputs with Claude V2
- Define Claude 3 function
- Read in reviews as .csv for iterative prompting
- Define function to prompt reviews
- Generate review sentiments, identify products mentioned, and generate summaries

## Claude V2 RAG Output Examples

When Claude V2 was provided with the vector store of user reviews and prompted "What are some of the features that people like most about Snapchat?", it returned:

![image](https://github.com/blallen22/llm-rag-claude-snapchat-reviews/assets/4731381/90d2df94-c20a-41b3-99e2-946b48d7639f)

When Claude V2 was provided with the vector store of user reviews and prompted "What are the most important issues users are experiencing with Snapchat?", it returned:

![image](https://github.com/blallen22/llm-rag-claude-snapchat-reviews/assets/4731381/f98aa016-f04f-4b31-94e9-fd796346bd79)

## Claude 3 Sonnet Sentiment Classification, Named Product Extraction, and Summary Generation Results
When Claude 3 Sonnet was iteratively prompted to classify the sentiment of the summary, extract any named products, and summarize the three most important words in five words or fewer each, it returned (sample of five results):

![image](https://github.com/blallen22/llm-rag-claude-snapchat-reviews/assets/4731381/23cae392-e966-4969-a93b-3a12a0f71c07)

## User Data

The Snapchat user reviews used in this project are [the 10 publicly-available Apple App Store user reviews found here](https://apps.apple.com/us/app/snapchat/id447188370?see-all=reviews). Although usernames are present at the url provided, these were masked in this application to ensure user privacy. While a larger dataset would be more valuable for generating insights at scale, large Kaggle datasets of Snapchat user reviews were not used, explicitly to avoid any perceived infringement using user data of dubious provenance. Additionally, while changing the region in the above URL (e.g., to 'gb' instead of 'us') would have yielded additional user reviews, those reviews were avoided to ensure regulatory compliance (i.e., GDPR, etc.).

## Next Steps
Future next steps for this project include:
- Incorporating evaluation methodologies to assess the quality of the outputs beyond the current heuristic assessment
- Inspecting the methodological decisions with further granularity (e.g., the chunk size during the chunking process, etc.)
- Applying this approach to additional use cases

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/)

## Resources Referenced

  - [Amazon Bedrock Workshop - Langchain Knowledge Bases and RAG Examples](https://github.com/aws-samples/amazon-bedrock-workshop/blob/main/06_OpenSource_examples/01_Langchain_KnowledgeBases_and_RAG_examples/01_qa_w_rag_claude.ipynb)
  - [Pinecone Langchain RAG Examples](https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-augmentation.ipynb)

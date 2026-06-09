1. make a call to the LLM --> only one call with RAG 
2. LLM decided to invoke search ('params')
3. We invoke the search, we have the results
4. send the results back to the LLM -->(another call with Agentic RAG)
5. LLM process the results
6. LLM gives the answer 
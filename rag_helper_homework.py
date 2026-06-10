from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

files = reader.read()

documents = []
for file in files:
    documents.append(file.parse()) 

from dotenv import load_dotenv
from openai import OpenAI
from minsearch import Index

load_dotenv()

openai_client = OpenAI()

# Build search index from GitHub lesson documents
index = Index(
    text_fields=["content"],
    keyword_fields=["filename"]
)

index.fit(documents)

INSTRUCTIONS = """
Your task is to answer questions using the lesson pages from the LLM Zoomcamp repository.

Use only the provided context.
If the answer is not found in the context, respond with "I don't know".
"""

PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="gpt-5.4-mini"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):

        return self.index.search(
            query,
            num_results=num_results
        )

    def build_context(self, search_results):

        lines = []

        for doc in search_results:
            lines.append(f"Filename: {doc['filename']}")
            lines.append(doc["content"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):

        context = self.build_context(search_results)

        return self.prompt_template.format(
            question=query,
            context=context
        )

    def llm(self, prompt):

        messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=messages
        )

        return response

    def rag(self, query):

        search_results = self.search(query)

        prompt = self.build_prompt(
            query=query,
            search_results=search_results
        )

        response = self.llm(prompt)

        return (
            response.output_text,
            response.usage
        )

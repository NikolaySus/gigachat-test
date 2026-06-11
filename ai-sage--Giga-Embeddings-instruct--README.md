---
license: mit
language:
- ru
- en
pipeline_tag: feature-extraction
tags:
- MTEB
- transformers
library_name: sentence-transformers
---
## Giga-Embeddings-instruct
- Base Decoder-only LLM: GigaChat-3b
- Pooling Type: Latent-Attention
- Embedding Dimension: 2048

Для получения более подробной информации о технических деталях, пожалуйста, обратитесь к нашей [статье](https://aclanthology.org/2025.bsnlp-1.3/).

## Использование

Ниже приведен пример кодирования запросов и текстов.

### Requirements

```bash
pip install -q transformers==4.51.0 sentence-transformers==5.1.1 flash-attn langchain_community langchain_huggingface langchain_gigachat
```

### Transformers

```python
import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = [
    get_detailed_instruct(task, 'What is the capital of Russia?'),
    get_detailed_instruct(task, 'Explain gravity')
]
# No need to add instruction for retrieval documents
documents = [
    "The capital of Russia is Moscow.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
]
input_texts = queries + documents

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
tokenizer = AutoTokenizer.from_pretrained(
    'ai-sage/Giga-Embeddings-instruct',
    trust_remote_code=True
)
model = AutoModel.from_pretrained(
    'ai-sage/Giga-Embeddings-instruct', 
    attn_implementation="flash_attention_2", 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()
model.cuda()

max_length = 4096

# Tokenize the input texts
batch_dict = tokenizer(
    input_texts,
    padding=True,
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)
batch_dict.to(model.device)
embeddings = model(**batch_dict, return_embeddings=True)

scores = (embeddings[:2] @ embeddings[2:].T)
print(scores.tolist())
# [[0.58203125, 0.0712890625], [0.06884765625, 0.62109375]]
```

### Sentence Transformers

```python
import torch

from sentence_transformers import SentenceTransformer

# Load the model
# We recommend enabling flash_attention_2 for better acceleration and memory saving
model = SentenceTransformer(
    "ai-sage/Giga-Embeddings-instruct",
    model_kwargs={
        "attn_implementation": "flash_attention_2", 
        "torch_dtype": torch.bfloat16, 
        "trust_remote_code": "True"
    },
    config_kwargs={
        "trust_remote_code": "True"
    }
)
model.max_seq_length = 4096

# The queries and documents to embed
queries = [
    'What is the capital of Russia?',
    'Explain gravity'
]
# No need to add instruction for retrieval documents
documents = [
    "The capital of Russia is Moscow.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
]

# Encode the queries and documents. Note that queries benefit from using a prompt
query_embeddings = model.encode(queries, prompt='Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ')
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
# tensor([[0.5846, 0.0702],
#         [0.0691, 0.6207]])
```

### LangChain

```python
import torch

from langchain_huggingface import HuggingFaceEmbeddings

# Load model
embeddings = HuggingFaceEmbeddings(
    model_name='ai-sage/Giga-Embeddings-instruct',
    encode_kwargs={},
    model_kwargs={
        'device': 'cuda',
        'trust_remote_code': True,
        'model_kwargs': {'torch_dtype': torch.bfloat16},
        'prompts': {'query': 'Instruct: Given a question, retrieve passages that answer the question\nQuery: '}
    }
)

# Tokenizer
embeddings._client.tokenizer.tokenize("Hello world! I am GigaChat")

# Query embeddings
query_embeddings = embeddings.embed_query("Hello world!")
print(f"Your embeddings: {query_embeddings[0:20]}...")
print(f"Vector size: {len(query_embeddings)}")

# Document embeddings
documents = ["foo bar", "bar foo"]
documents_embeddings = embeddings.embed_documents(documents)
print(f"Vector size: {len(documents_embeddings)} x {len(documents_embeddings[0])}")
```

## Инструктивность

**Использование инструкций для улучшения качества эмбеддингов**  

Для достижения более точных результатов при работе с эмбеддингами, особенно в задачах поиска и извлечения информации (retrieval), рекомендуется добавлять инструкцию на естественном языке перед текстовым запросом (query). Это помогает модели лучше понять контекст и цель запроса, что положительно сказывается на качестве результатов. Важно отметить, что инструкцию нужно добавлять только перед запросом, а не перед документом.  

Для **симметричных задач**, таких как классификация (classification) или семантическое сравнение текстов (semantic text similarity), инструкцию необходимо добавлять перед каждым запросом. Это связано с тем, что такие задачи требуют одинакового контекста для всех входных данных, чтобы модель могла корректно сравнивать или классифицировать их.  

**Примеры инструкций для симметричных задач:**  
- `"Retrieve semantically similar text"`  
- `"Given a text, retrieve semantically similar text"`  
- `"Дано предложение, необходимо найти его парафраз"`  
- `"Классифицируй отзыв на товар как положительный, отрицательный или нейтральный"`  
- `"Классифицируй чувствительную тему по запросу"`  

Для **retrieval-задач** (например, поиск ответа в тексте) можно использовать инструкцию:  
`'Дан вопрос, необходимо найти абзац текста с ответом'`.  

Такой подход особенно эффективен для задач поиска и извлечения информации, таких как поиск релевантных документов или извлечение ответов из текста.

**Примеры инструкций для retrieval-задач:**   
- `'Дан вопрос, необходимо найти абзац текста с ответом'`
- `'Given the question, find a paragraph with the answer'`     

Инструкции необходимо оборачивать в шаблон: `f'Instruct: {task_description}\nQuery: {query}'`. Использование инструкций позволяет значительно улучшить качество поиска и релевантность результатов, что подтверждается тестами на бенчмарках, таких как RuBQ, MIRACL. Для симметричных задач добавление инструкции перед каждым запросом обеспечивает согласованность и повышает точность модели.

## Поддерживаемые языки

Эта модель инициализирована pretrain моделью GigaChat и дополнительно обучена на смеси английских и русских данных.

## FAQ

1. Нужно ли добавлять инструкции к запросу?

Да, именно так модель обучалась, иначе вы увидите снижение качества. Определение задачи должно быть инструкцией в одном предложении, которая описывает задачу. Это способ настройки текстовых эмбеддингов для разных сценариев с помощью инструкций на естественном языке.

С другой стороны, добавлять инструкции на сторону документа не требуется.

2. Почему мои воспроизведённые результаты немного отличаются от указанных в карточке модели?

Разные версии библиотек transformers и pytorch могут вызывать незначительные, но ненулевые различия в результатах.


## Ограничения

Использование этой модели для входных данных, содержащих более 4096 токенов, невозможно.
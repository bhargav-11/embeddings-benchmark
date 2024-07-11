import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import torch
import time
from dotenv import load_dotenv
import os
import voyageai
from mistralai.client import MistralClient

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
voyage_api_key = os.getenv("VOYAGE_API_KEY")

vo = voyageai.Client(api_key=voyage_api_key)
openai_client = OpenAI(api_key=openai_api_key)
mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY"))

# Define the models to evaluate
models = {
    "text-embedding-3-large": {"type": "openai", "model": "text-embedding-3-large"},
    # "voyage-large-2-instruct": {"type": "voyage", "model": "voyage-large-2-instruct"},
    "voyage-multilingual-2": {"type": "voyage", "model": "voyage-multilingual-2"},
    "mistral-embed": {"type": "mistral", "model": "mistral-embed"},
    "multilingual-e5-large": {"type": "huggingface", "model": "intfloat/multilingual-e5-large"}
}

# Define the documents and queries in multiple languages
documents = [
    (
        "El Comité de Representantes Permanentes o Coreper —artículo 16, apartado 7, del Tratado de la Unión Europea (TUE) y artículo 240, apartado 1, del Tratado de Funcionamiento de la Unión Europea (TFUE)— se encarga de preparar los trabajos del Consejo de la Unión Europea. Cada Estado miembro de la Unión Europea (UE) está representado en el Coreper por un representante permanente (Coreper II) y un representante permanente adjunto (Coreper I) con rango de embajador de la UE. El Coreper ocupa un lugar central en el sistema de toma de decisiones de la UE. Coordina y prepara los trabajos de todas las sesiones del Consejo y, a su nivel, trata de llegar a un acuerdo que posteriormente se somete a la aprobación del Consejo. Además, vela por la coherencia de las políticas y acciones de la UE y por que se respeten: los principios de legalidad, subsidiariedad, proporcionalidad y motivación de los actos; las normas por las que se establecen los poderes de las instituciones, órganos y organismos de la UE; las disposiciones presupuestarias; las normas de procedimiento, transparencia y calidad de la redacción. Garantiza una presentación adecuada de cada expediente al Consejo y, en su caso, presenta directrices, opciones o sugerencias. Además, el orden del día de las sesiones del Consejo se elabora en función de la situación en que se hallen los trabajos del Coreper. El Coreper se divide en dos partes: la parte I (o Coreper I) incluye puntos que, en principio, no requieren debate y que normalmente serán puntos «A» del orden del día del Consejo (es decir, puntos que, tal como los prepara el Coreper, podrían ser aprobados por el Consejo sin debate); la parte II (Coreper II) requiere un debate. No obstante, si el Coreper llega a un acuerdo sobre un punto de la parte II de su orden del día, dicho punto se incluye normalmente como punto «A» en el orden del día del Consejo, que consiste en lo siguiente: los puntos A, se aprueban sin debate en el seno del Coreper; los puntos B, que requieren un debate. El Coreper se divide en dos partes: el Coreper I prepara el trabajo de seis formaciones del Consejo, a saber: Empleo, Política Social, Sanidad y Consumidores; Competitividad (Mercado Interior, Industria e Investigación y Espacio); Transporte, Telecomunicaciones y Energía; Agricultura y Pesca; Medio Ambiente; Educación, Juventud, Cultura y Deporte. el Coreper II prepara el trabajo de cuatro formaciones del Consejo, a saber: Asuntos Generales; Asuntos Exteriores; Asuntos Económicos y Financieros; Justicia y Asuntos de Interior. En principio, el Coreper se reúne cada semana. La víspera, los preparativos de su trabajo son llevados a cabo por los colaboradores más cercanos de los miembros del Coreper, que se reúnen con los siguientes nombres: Grupo Mertens para el Coreper I; Grupo Antici para el Coreper II. Estos grupos revisan el orden del día de los Coreper I y II, respectivamente, y establecen los detalles técnicos y organizativos. Esta etapa preparatoria también permite definir una idea inicial de las posiciones que adoptarán las distintas delegaciones en la reunión del Coreper. El Coreper puede adoptar las decisiones de procedimiento enumeradas en el artículo 19, apartado 7, del Reglamento Interno del Consejo (por ejemplo, la decisión de celebrar una sesión del Consejo en un lugar distinto de Bruselas o Luxemburgo, o la decisión de aplicar el procedimiento escrito). Órganos preparatorios del Consejo El Consejo de la Unión Europea tiene el apoyo del Coreper, o Comité de Representantes Permanentes de los Gobiernos de los Estados miembros ante la Unión Europea (UE), así como de grupos de trabajo y comités altamente especializados. Estos órganos preparatorios ayudan a preparar el trabajo del Consejo. Están compuestos por delegados de todos los Estados miembros de la UE y cuentan con la asistencia de la Secretaría general del Consejo.",
        "Coreper in EU Decision-Making", 
        "Role and responsibilities of Coreper in EU decision-making"
    ),
    (
        "3.3 Anonymization For ethical and legal reasons, we decided to anonymize the dataset before publishing it. While the original data does not contain any information from individuals, it does contain (publicly accessible) information from companies, like phone numbers, addresses, and tax IDs. We took multiple steps to remove this data from the contracts in order to make it harder to identify the company that drafted the contract. Companies can and do change their contracts over time, so we want to avoid consumers finding and reading an outdated version of a contract. Additionally, it is not unlikely that one or multiple assessments made by the annotators would not hold up in a court of law, either due to an unconscious mistake or due to the aforementioned bias with regard to the interpretation of the law. Wrongfully claiming a company uses void terms can potentially be harmful to their business and could implicate liabilities. The other way around, wrongfully claiming a clause is valid could potentially harm consumers if they rely on that assessment. Therefore we implemented ten anonymization steps: 1. Remove all clauses with the topic label party from the corpus (these clauses consist only of information about the contracting party, i.e. the company) 2. Replace all email addresses with “hello@example.com” using regular expressions (regex) 3. Replace all URLs with “www.example.com” using regex 4. Replace all international bank account numbers (IBANs) with “DE75512108001245126199” using regex 5. Replace all tax IDs with DE398517849 using regex 6. Replace all phone numbers with 00 00 12345678 using regex 7. Replace all ZIP codes with 00000 using regex 8. Replace all names of companies and organizations with “«NAME»” using Named Entity Recognition (NER) 9. Replace all city names with “«STADT»” (German for city) using NER 10. Replace all street names with “«STRASSE»” (German for street) using NER While the first seven steps turned out to work very well and straight-forward (in total 84 party clauses have been removed and 120 email addresses, 231 URLs, 2 IBANs, 117 phone numbers, and 279 ZIP codes have been replaced), many of the available standard NER libraries turned out to not work very well for the texts. In the end, the FLAIR library (Akbik et al., 2019) with the ner-german-legal model (Leitner et al., 2019) turned out to be most suitable. With the help of the model, we were able to replace 724 names of organizations and companies, 418 city names, and 53 street names. However, a manual inspection revealed that an additional, manual, anonymization round was necessary. In this manual process an additional 1,338 company names, 38 city names, and 85 streets have been removed. In order to avoid over-anonymization which could potentially result in decreased classification performance, a list of organizations and URLs were explicitly excluded from being removed or replaced. The list mainly included political bodies like the European Union and their URLs, shipping companies and their URLs, and payment provider and their URLs. An excerpt from the final corpus is shown in Table 1. Corpus Analysis The corpus consists of 93 contracts with 3,764 clauses (an average of 40 clauses per contract), which contain 11,387 sentences (avg. of 3 sentences per clause) and 250,859 tokens (avg. of 22 per sentence). Out of the 3,764 clauses present in the corpus, 179 (or 4.8%) have been annotated as potentially void. That is comparable, although slightly lower, than the 6% reported by Braun and Matthes (2021) on a much smaller dataset of 24 contracts. While that results in a corpus that is imbalanced, we believe it to be a realistic reflection of reality, where void clauses are also significantly less frequent than void clauses. 5 Automated Legal Assessment In order to present a baseline for the main task for which the corpus was designed and evaluate the difficulty of the task, we compared different language models and an SVM for the classification of clauses into (potentially) void and valid.",
        "Anonymization of Dataset", 
        "Ethical and legal reasons for anonymizing the dataset"
    )
]

queries = [
    ("Role and responsibilities of Coreper in EU decision-making", "English"),
    ("Rolle und Verantwortlichkeiten des Coreper im EU-Entscheidungsprozess", "German"),
    ("Rolle och ansvar för Coreper i EU:s beslutsfattande", "Swedish"),
    ("Coreperin rooli ja vastuut EU:n päätöksenteossa", "Finnish"),
    ("Rôle et responsabilités du Coreper dans la prise de décision de l'UE", "French"),
    ("Papel y responsabilidades del Coreper en la toma de decisiones de la UE", "Spanish"),
    ("Função e responsabilidades do Coreper na tomada de decisões da UE", "Portuguese"),
    ("Rola i obowiązki Coreper w procesie decyzyjnym UE", "Polish"),
    ("Rol en verantwoordelijkheden van Coreper in de besluitvorming van de EU", "Dutch"),
    ("Rolle og ansvar for Coreper i EUs beslutningsprosess", "Norwegian"),
    ("Corepers rolle og ansvar i EU-beslutningsprocessen", "Danish"),
    ("Ethical and legal reasons for anonymizing the dataset", "English"),
    ("Ethische und rechtliche Gründe für die Anonymisierung des Datensatzes", "German"),
    ("Etiska och juridiska skäl för att anonymisera datasetet", "Swedish"),
    ("Eettiset ja oikeudelliset syyt anonymisoida tietojoukko", "Finnish"),
    ("Raisons éthiques et légales pour anonymiser le jeu de données", "French"),
    ("Razones éticas y legales para anonimizar el conjunto de datos", "Spanish"),
    ("Razões éticas e legais para anonimizar o conjunto de dados", "Portuguese"),
    ("Etyczne i prawne powody anonimizacji zbioru danych", "Polish"),
    ("Ethische en juridische redenen om de dataset te anonimiseren", "Dutch"),
    ("Etiske og juridiske grunner for å anonymisere datasettet", "Norwegian"),
    ("Etiske og juridiske grunde til at anonymisere datasættet", "Danish")
]

def get_embedding(text, model_info):
    if model_info["type"] == "openai":
        response = openai_client.embeddings.create(input=text, model=model_info["model"])
        return np.array(response.data[0].embedding)
    elif model_info["type"] == "voyage":
        result = vo.embed([text], model=model_info["model"])
        return np.array(result.embeddings[0])
    elif model_info["type"] == "mistral":
        response = mistral_client.embeddings(model=model_info["model"], input=[text])
        return np.array(response.data[0].embedding)
    elif model_info["type"] == "huggingface":
        tokenizer = AutoTokenizer.from_pretrained(model_info["model"])
        model = AutoModel.from_pretrained(model_info["model"])
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

def evaluate_model(model_name, model_info):
    results = []
    for doc, doc_name, doc_query in documents:
        start_time = time.time()
        
        # Embed document and queries
        doc_embedding = get_embedding(doc, model_info)
        query_embeddings = [get_embedding(query, model_info) for query, _ in queries]
        
        # Calculate cosine similarities
        similarities = cosine_similarity([doc_embedding], query_embeddings).flatten()
        
        # Evaluate performance
        average_similarity = np.mean(similarities)
        
        end_time = time.time()
        speed = end_time - start_time
        
        # Calculate memory usage
        memory_usage = doc_embedding.nbytes + sum(embed.nbytes for embed in query_embeddings)
        
        results.append({
            "document_name": doc_name,
            "average_similarity": average_similarity,
            "speed": speed,
            "memory_usage": memory_usage / (1024 ** 2),  # Convert to MB
            "similarity_scores": similarities
        })
    
    return results

# Evaluate all models
all_results = {}
for model_name, model_info in models.items():
    print(f"Evaluating {model_name}...")
    all_results[model_name] = evaluate_model(model_name, model_info)

# Print and plot results
for model_name, model_results in all_results.items():
    for result in model_results:
        doc_name = result["document_name"]
        print(f"\nResults for {model_name} on document '{doc_name}':")
        print(f"Average Similarity: {result['average_similarity']:.4f}")
        print(f"Speed: {result['speed']:.2f} seconds")
        print(f"Memory Usage: {result['memory_usage']:.2f} MB")
        
        similarity_scores = result["similarity_scores"]
        query_labels = [f"{query} ({language})" for query, language in queries]
        
        plt.figure(figsize=(12, 6))
        plt.title(f'Similarity Scores for {model_name} - {doc_name}')
        plt.barh(query_labels, similarity_scores, color='skyblue')
        
        for index, value in enumerate(similarity_scores):
            plt.text(value, index, f'{value:.4f}', va='center')
        
        plt.xlabel('Similarity Score')
        plt.ylabel('Query (Language)')
        plt.tight_layout()
        plt.show()

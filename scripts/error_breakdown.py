import logging
import time
from collections import Counter, defaultdict
import csv
import json
import nltk
import string
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from heapq import nlargest
import torch
import torch.nn as nn
from langchain.llms import GPT4All
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
from transformers import set_seed

# Download NLTK data if not present
# nltk.download('punkt')
# nltk.download('stopwords')

pdist = nn.PairwiseDistance(p=2.0, eps=1e-06)

# Utility functions
def normalize_and_stem(text):
    stemmer = PorterStemmer()
    tokens = nltk.word_tokenize(text.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens if token not in string.punctuation and token not in stopwords.words('english')]
    return ' '.join(stemmed_tokens)

def calculate_em(predicted, actual):
    return int(predicted == actual)

def calculate_token_f1(predicted, actual):
    predicted_tokens = predicted.split()
    actual_tokens = actual.split()
    common_tokens = Counter(predicted_tokens) & Counter(actual_tokens)
    num_same = sum(common_tokens.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(actual_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def get_question_type(question):
    q = question.lower()
    if q.startswith("what"):
        return "what"
    elif q.startswith("when"):
        return "when"
    elif q.startswith("who"):
        return "who"
    elif q.startswith("why"):
        return "why"
    elif q.startswith("how many"):
        return "how many"
    elif q.startswith("how"):
        return "how"
    elif q.startswith("where"):
        return "where"
    else:
        return "other"


def newsqa_loop(data, llm, output_csv_path, output_log_path, max_stories,
                chunk_sizes, overlap_percentages, top_n_sentences, dist_functions,
                instruct_embedding_model_name, instruct_embedding_model_kwargs, 
                instruct_embedding_encode_kwargs, QA_CHAIN_PROMPT,
                max_question_per_story=100):

    # Clear CSV and log file contents first
    open(output_csv_path, 'w').close()
    open(output_log_path, 'w').close()

    with open(output_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Chunk_size', 'Chunk_Overlap', 'Top_N', 'Distance_Function', 'Time',
            'Story Number', 'Question Number', 'isQuestionBad', 'isAnswerAbsent', 'Question Type',
            'EM', 'Precision', 'Recall', 'F1',
            'Original Question', 'Correct Answer', 'Normalized Actual Answer',
            'Predicted Answer', 'Normalized Predicted Answer',
            'EM Fail', 'Chunk Retrieval Failure', 'Answer Generation Failure', 'Answer Too Long'
        ])

        hf_story_embs = HuggingFaceInstructEmbeddings(
            model_name=instruct_embedding_model_name,
            model_kwargs=instruct_embedding_model_kwargs,
            encode_kwargs=instruct_embedding_encode_kwargs,
            embed_instruction="Use the following pieces of context to answer the question at the end:"
        )

        hf_query_embs = HuggingFaceInstructEmbeddings(
            model_name=instruct_embedding_model_name,
            model_kwargs=instruct_embedding_model_kwargs,
            encode_kwargs=instruct_embedding_encode_kwargs,
            query_instruction="How does this information relate to the question?"
        )

        start_time = time.time()

        def get_question_type(q):
            q_lower = q.lower()
            for key in ["what", "when", "who", "why", "how many", "where", "how"]:
                if key in q_lower:
                    return key
            return "other"

        for chunk_size in chunk_sizes:
            for overlap_percentage in overlap_percentages:
                actual_overlap = int(chunk_size * overlap_percentage)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=actual_overlap)

                for dist_function in dist_functions:
                    for top_n_sentence in top_n_sentences:

                        print(f"\n{time.time()-start_time:.2f}s | Chunk: {chunk_size}, Overlap: {actual_overlap}, TopN: {top_n_sentence}, DistFn: {dist_function}")

                        for i, story in enumerate(data['data']):
                            if i >= max_stories:
                                break
                            print(f"Processing STORY {i}...")

                            with open(output_log_path, 'a') as details_file:
                                details_file.write(f"===== STORY {i} START =====\n")
                                details_file.write(f"Original News Text:\n{story['text']}\n")
                                details_file.write("===== NEWS TEXT END =====\n\n")

                            sentences = sent_tokenize(story['text'])
                            sentence_embs = hf_story_embs.embed_documents(sentences)
                            all_splits = text_splitter.split_text(story['text'])
                            vectorstore = Chroma.from_texts(texts=all_splits, embedding=hf_story_embs)
                            qa_chain = RetrievalQA.from_chain_type(
                                llm,
                                retriever=vectorstore.as_retriever(),
                                chain_type="stuff",
                                verbose=False,
                                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT, "verbose": False},
                                return_source_documents=False
                            )

                            for j, question_data in enumerate(story['questions'][:max_question_per_story]):
                                is_question_bad = question_data.get('isQuestionBad', 0.0)
                                is_answer_absent = question_data.get('isAnswerAbsent', 0.0)
                                question = question_data['q']
                                question_type = get_question_type(question)

                                if is_question_bad == 1 or is_answer_absent == 1:
                                    writer.writerow([
                                        chunk_size, overlap_percentage, top_n_sentence, dist_function,
                                        time.time()-start_time, i, j,
                                        f"{is_question_bad:.2f}", f"{is_answer_absent:.2f}", question_type,
                                        '', '', '', '', question, '', '', '', '', '', '', '', ''
                                    ])

                                    with open(output_log_path, 'a') as details_file:
                                        details_file.write(f"Story: {i}\nQuestion: {j}\n")
                                        details_file.write(f"isQuestionBad: {is_question_bad:.2f}\n")
                                        details_file.write(f"isAnswerAbsent: {is_answer_absent:.2f}\n")
                                        details_file.write("Skipped due to bad/absent flag.\n")
                                        details_file.write("----------------------------------------\n")
                                    continue

                                print(f"\tProcessing QUESTION {j}...")

                                question_emb = hf_query_embs.embed_documents([question])[0]

                                if dist_function == 'pairwise':
                                    scores = [pdist(torch.tensor(se).unsqueeze(0), torch.tensor(question_emb).unsqueeze(0)).item() for se in sentence_embs]
                                else:
                                    scores = [torch.cosine_similarity(torch.tensor(se).unsqueeze(0), torch.tensor(question_emb).unsqueeze(0))[0].item() for se in sentence_embs]

                                top_scores_indices = nlargest(top_n_sentence, range(len(scores)), key=lambda idx: scores[idx])
                                context_for_qa = " ".join([sentences[idx] for idx in top_scores_indices])

                                consensus = question_data['consensus']
                                if 's' in consensus and 'e' in consensus:
                                    actual_answer = story['text'][consensus['s']:consensus['e']]
                                else:
                                    continue

                                result = qa_chain({"context": context_for_qa, "query": question})
                                predicted_answer = result['result'] if isinstance(result['result'], str) else ""
                                norm_pred = normalize_and_stem(predicted_answer)
                                norm_actual = normalize_and_stem(actual_answer)
                                f1, precision, recall = calculate_token_f1(norm_pred, norm_actual)
                                em = calculate_em(norm_pred, norm_actual)

                                em_fail = 1 - em
                                chunk_retrieval_fail = int(em_fail == 1 and actual_answer not in context_for_qa)
                                answer_gen_fail = int(em_fail == 1 and chunk_retrieval_fail == 0 and precision == 0)
                                answer_too_long = int(em_fail == 1 and chunk_retrieval_fail == 0 and precision != 0)

                                writer.writerow([
                                    chunk_size, overlap_percentage, top_n_sentence, dist_function,
                                    time.time()-start_time, i, j,
                                    f"{is_question_bad:.2f}", f"{is_answer_absent:.2f}", question_type,
                                    em, precision, recall, f1,
                                    question, actual_answer, norm_actual,
                                    predicted_answer, norm_pred,
                                    em_fail, chunk_retrieval_fail, answer_gen_fail, answer_too_long
                                ])

                                with open(output_log_path, 'a') as details_file:
                                    details_file.write(f"Chunk Size: {chunk_size}\n")
                                    details_file.write(f"Overlap: {overlap_percentage}\n")
                                    details_file.write(f"TopN: {top_n_sentence}\n")
                                    details_file.write(f"Distance Function: {dist_function}\n")
                                    details_file.write(f"Story: {i}\n")
                                    details_file.write(f"Question: {j}\n")
                                    details_file.write(f"isQuestionBad: {is_question_bad:.2f}\n")
                                    details_file.write(f"isAnswerAbsent: {is_answer_absent:.2f}\n")
                                    details_file.write(f"Question Type: {question_type}\n")
                                    details_file.write(f"Original Question: {question}\n")
                                    details_file.write(f"Correct Answer: {actual_answer}\n")
                                    details_file.write(f"Normalized Actual Answer: {norm_actual}\n")
                                    details_file.write(f"Predicted Answer: {predicted_answer}\n")
                                    details_file.write(f"Normalized Predicted Answer: {norm_pred}\n")
                                    details_file.write(f"Retrieved Chunk: {context_for_qa}\n")
                                    details_file.write(f"Time: {time.time() - start_time}\n")
                                    details_file.write(f"EM Score: {em}\n")
                                    details_file.write(f"Precision: {precision}\n")
                                    details_file.write(f"Recall: {recall}\n")
                                    details_file.write(f"F1: {f1}\n")
                                    details_file.write(f"EM Fail: {em_fail}\n")
                                    details_file.write(f"Chunk Retrieval Failure: {chunk_retrieval_fail}\n")
                                    details_file.write(f"Answer Generation Failure: {answer_gen_fail}\n")
                                    details_file.write(f"Answer Too Long: {answer_too_long}\n")
                                    details_file.write("----------------------------------------\n")

                            del qa_chain
                            del vectorstore
                            del all_splits

                del text_splitter


# Example parameters
if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)
    set_seed(123)

    model_location = "/home/ly364/Models/Mistral-7B-Instruct-v0.3.Q4_K_M.gguf"
    input_file_path = "/home/ly364/NewsQA/combined-newsqa-data-v1.json"
    output_csv_path = "/home/ly364/NewsQA/2025/error_breakdown_20250527.csv"
    output_log_path = "/home/ly364/NewsQA/2025/error_breakdown_20250527.log"

    print("Loading data.")
    data = json.loads(Path(input_file_path).read_text())

    print("Preparing template.")
    template_original = """
    Based on the following information only:
    
    {context}

    {question} Please provide the answer in as few words as possible and please do NOT repeat any word in the question, i.e. \"{question}\".

    Answer:
    """
    QA_CHAIN_PROMPT_ORIGINAL = PromptTemplate.from_template(template_original)

    print("Loading LLM.")
    llm = GPT4All(model=model_location, backend="gptj", max_tokens=2048, seed=123)

    instruct_embedding_model_name = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    instruct_embedding_model_kwargs = {'device': 'cpu'}
    instruct_embedding_encode_kwargs = {'normalize_embeddings': True}

    newsqa_loop(
        data,
        llm,
        output_csv_path,
        output_log_path,
        max_stories=50,
        max_question_per_story=10,
        chunk_sizes=[100],
        overlap_percentages=[0],
        top_n_sentences=[2],
        dist_functions=['cosine'],
        instruct_embedding_model_name=instruct_embedding_model_name,
        instruct_embedding_model_kwargs=instruct_embedding_model_kwargs,
        instruct_embedding_encode_kwargs=instruct_embedding_encode_kwargs,
        QA_CHAIN_PROMPT=QA_CHAIN_PROMPT_ORIGINAL
    )

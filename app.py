import os
import requests
import logging
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
import tiktoken
from flask import Flask, render_template, request, jsonify
from langchain.tools.searx_search import SearxSearch
import schedule
import time
import subprocess
from openai.embeddings_utils import distances_from_embeddings

# Setup logging
logging.basicConfig(level=logging.INFO)

# OpenAI API setup
openai.api_key = 'YOUR_OPENAI_API_KEY'

# SearxNG API setup
searx = SearxSearch(api_base_url='http://localhost:8888', categories=["news"])

# Directory setup
BLOG_DIR = 'blog_posts'
if not os.path.exists(BLOG_DIR):
    os.makedirs(BLOG_DIR)

# User-defined parameters
TONE_STYLE = "professional and informative"
QUALITY_TONE_STYLE = "casual but engaging"
NUM_THREADS = 5
CONTEXT_SIZE = 8192  # Default context size in tokens

app = Flask(__name__)
current_content = ""
current_progress = ""

def fetch_trending_topics():
    logging.info("Fetching trending topics...")
    # Implement logic to fetch trending topics
    topics = ["AI in Healthcare", "AI in Finance", "AI in Education"]
    return topics

def search_and_download(query, output_dir="text"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logging.info(f"Searching and downloading content for query: {query}")
    
    search_results = searx.search(query, num_results=10)
    crawled_urls = set()

    for result in search_results:
        url = result['url']
        if url in crawled_urls:
            continue
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join([p.get_text() for p in soup.find_all('p')])
            filename = os.path.join(output_dir, f"{len(crawled_urls)}.txt")
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(text)
            crawled_urls.add(url)
        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")

def crawl_website_for_topic(topic, num_threads=NUM_THREADS):
    queries = [f"{topic} {section}" for section in generate_blog_structure(topic)]
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(search_and_download, queries), total=len(queries)))

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def process_text_files(input_dir="text", output_csv="processed/scraped.csv"):
    texts = []
    for file in os.listdir(input_dir):
        with open(os.path.join(input_dir, file), 'r', encoding="UTF-8") as f:
            text = f.read()
            texts.append((file[:-4], text))

    df = pd.DataFrame(texts, columns=['fname', 'text'])
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv(output_csv)
    return df

def create_embeddings(df, output_csv="processed/embeddings.csv"):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    max_tokens = 500

    def split_into_many(text, max_tokens=max_tokens):
        sentences = text.split('. ')
        n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

        chunks = []
        tokens_so_far = 0
        chunk = []

        for sentence, token in zip(sentences, n_tokens):
            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            if token > max_tokens:
                continue

            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks

    shortened = []

    for row in df.iterrows():
        if row[1]['text'] is None:
            continue

        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])
        else:
            shortened.append(row[1]['text'])

    df = pd.DataFrame(shortened, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, model='text-embedding-ada-002')['data'][0]['embedding'])
    df.to_csv(output_csv)
    return df

def generate_blog_structure(topic):
    logging.info("Generating blog structure...")
    prompt = f"""
    Create a detailed structure for a blog post about "{topic}". The blog should include the following sections:
    1. Introduction
    2. The Current State of AI in {topic}
    3. Key Applications of AI in {topic}
    4. Benefits of AI in {topic}
    5. Challenges and Ethical Considerations
    6. Future Trends of AI in {topic}
    7. Conclusion
    For each section, provide a brief overview of the content that should be covered.
    """
    structure = query_llm(prompt)
    return [line.strip() for line in structure.split('\n') if line.strip()]

def query_llm(prompt, max_tokens=1500, stop_sequence=None):
    logging.info("Querying LLM...")
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer the question based on the context below, and if the question can't be answered based on the context, say 'I don't know'\n\n"},
                {"role": "user", "content": f"{prompt}"}
            ],
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        return ""

def clean_text(text):
    return ' '.join(text.split())

def split_text_into_chunks(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) * 4 / 3  # Approximating token size
        if current_length >= max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = len(word) * 4 / 3
        current_chunk.append(word)

    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def generate_blog_content(structure, materials, tone_style, context_size=CONTEXT_SIZE):
    global current_content
    global current_progress
    logging.info("Generating blog content...")
    max_prompt_tokens = context_size // 2  # To allow the LLM to generate content within context size

    combined_content = []
    for i, section in enumerate(structure, 1):
        section_prompt = f"Write the section '{section}' of a blog post in a {tone_style} tone using the following materials:\n\n{json.dumps(materials)}"
        section_prompt = clean_text(section_prompt)
        
        chunks = split_text_into_chunks(section_prompt, max_prompt_tokens)
        section_content = []
        for j, chunk in enumerate(chunks):
            chunk_prompt = f"Chunk {j + 1} of {len(chunks)}: {chunk}"
            chunk_content = query_llm(chunk_prompt)
            section_content.append(chunk_content)
            current_progress = f"Generating content for {section} (Chunk {j + 1}/{len(chunks)})"
            update_current_content('\n'.join(section_content))
        
        combined_content.append(f"Section {i} - {section}\n" + ' '.join(section_content))

    final_prompt = f"Combine the following sections into a single cohesive blog post:\n\n{json.dumps(combined_content)}"
    final_content = query_llm(final_prompt, max_tokens=context_size)
    current_progress = "Combining sections into a cohesive blog post"
    return final_content

def quality_control(content, tone_style=QUALITY_TONE_STYLE):
    logging.info("Performing quality control on content...")
    prompt = f"""
    Review and improve the following blog post content to ensure it is engaging and maintains a {tone_style} style. Correct any grammatical errors and enhance the readability:

    {content}
    """
    improved_content = query_llm(prompt)
    return improved_content

def update_current_content(content):
    global current_content
    current_content = content

def save_blog_post(topic, content):
    logging.info(f"Saving blog post for topic: {topic}")
    topic_dir = os.path.join(BLOG_DIR, topic.replace(' ', '_'))
    if not os.path.exists(topic_dir):
        os.makedirs(topic_dir)
    
    with open(os.path.join(topic_dir, 'content.html'), 'w') as f:
        f.write(content)

    # Update blog.html with new post link
    update_blog_index(topic, content)

def update_blog_index(topic, content):
    logging.info(f"Updating blog index for topic: {topic}")
    soup = BeautifulSoup(content, 'html.parser')
    title = soup.find('h1').text if soup.find('h1') else 'No Title'
    summary = ' '.join([p.text for p in soup.find_all('p')[:2]])  # First two paragraphs as summary

    blog_entry = f"""
    <div class="blog-entry">
        <h2><a href="{topic.replace(' ', '_')}/content.html">{title}</a></h2>
        <p>{summary}</p>
    </div>
    """

    with open("blog.html", "r") as file:
        blog_html = file.read()

    soup = BeautifulSoup(blog_html, 'html.parser')
    soup.body.append(BeautifulSoup(blog_entry, 'html.parser'))

    with open("blog.html", "w") as file:
        file.write(str(soup))

    # Commit and push changes to Git
    commit_and_push_changes(topic)

def commit_and_push_changes(topic):
    logging.info(f"Committing and pushing changes for topic: {topic}")
    commit_message = f"Add blog post for topic: {topic} at {time.strftime('%Y-%m-%d %H:%M:%S')}"
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", commit_message])
    subprocess.run(["git", "push", "--force", "origin", "clean-state-branch:master"])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    topic = request.form.get('topic')
    tone_style = request.form.get('tone_style', TONE_STYLE)
    quality_tone_style = request.form.get('quality_tone_style', QUALITY_TONE_STYLE)
    context_size = int(request.form.get('context_size', CONTEXT_SIZE))

    def run():
        global current_progress
        try:
            current_progress = "Crawling website for topic"
            crawl_website_for_topic(topic)
            current_progress = "Processing text files"
            df = process_text_files()
            current_progress = "Creating embeddings"
            df = create_embeddings(df)
            current_progress = "Generating blog structure"
            structure = generate_blog_structure(topic)
            current_progress = "Generating initial blog content"
            initial_content = generate_blog_content(structure, df['text'].tolist(), tone_style, context_size)
            current_progress = "Performing quality control"
            final_content = quality_control(initial_content, quality_tone_style)
            current_progress = "Saving blog post"
            save_blog_post(topic, final_content)
            current_progress = "Completed"
        except Exception as e:
            logging.error(f"Error in generating blog: {e}")
            current_progress = f"Error: {e}"

    Thread(target=run).start()
    return jsonify({"status": "started"})

@app.route('/progress')
def progress():
    return jsonify({"progress": current_progress, "content": current_content})

if __name__ == "__main__":
    app.run(debug=True)

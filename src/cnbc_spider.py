import scrapy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class CNBCSpider(scrapy.Spider):
    name = 'cnbc_spider'
    start_urls = [
        'https://www.cnbc.com/world/',
        'https://www.cnbc.com/markets/',
        'https://www.cnbc.com/technology/',
        'https://www.cnbc.com/investing/',
        'https://www.cnbc.com/finance/',
    ]  # You can add more sections if needed

    def __init__(self, *args, **kwargs):
        super(CNBCSpider, self).__init__(*args, **kwargs)

        # Load FinBERT model
        self.logger.info("Loading FinBERT model...")
        model_name = "yiyanghkust/finbert-tone"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Set device (MPS for Mac, CUDA for Nvidia, else CPU)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else (
                                   "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device)

        self.logger.info(f"Using device: {self.device}")
        self.finbert = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer, device=0 if self.device != torch.device("cpu") else -1)

        # Set to track URLs and avoid duplicates
        self.visited_urls = set()

    def parse(self, response):
        article_links = response.css('a::attr(href)').extract()
        self.logger.info(f"Found {len(article_links)} links on the page")

        followed_count = 0
        for url in article_links:
            # Accept articles from any year (2020–2025), avoid video pages
            if (
                    url.startswith("https://www.cnbc.com/202")  # only articles with dates
                    and "video" not in url
                    and url not in self.visited_urls
            ):
                self.visited_urls.add(url)
                followed_count += 1
                self.logger.info(f"Following article link: {url}")
                yield scrapy.Request(url, callback=self.parse_article)

        self.logger.info(f"Following {followed_count} article links")

    def parse_article(self, response):
        self.logger.info(f"Processing article: {response.url}")

        title = response.css('h1::text, .ArticleHeader-headline::text').get()

        selectors = [
            'div.ArticleBody-articleBody p::text',
            '.group-container p::text',
            'article p::text',
            'p::text'
        ]

        content = []
        for selector in selectors:
            text = response.css(selector).getall()
            if text:
                content = text
                break

        if not content or not title:
            self.logger.warning(f"Skipped empty article: {response.url}")
            return

        full_text = ' '.join(content)
        tokens = self.tokenizer.tokenize(full_text)
        max_tokens = 450  # keep below 512 token limit

        chunks = [self.tokenizer.convert_tokens_to_string(tokens[i:i + max_tokens])
                  for i in range(0, len(tokens), max_tokens)]
        self.logger.info(f"Split into {len(chunks)} chunks")

        sentiment_scores = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        valid_chunks = 0

        for i, chunk in enumerate(chunks):
            if chunk.strip():
                result = self.finbert(chunk)[0]
                sentiment_scores[result['label']] += result['score']
                self.logger.info(f"Chunk {i + 1} → {result['label']} ({result['score']:.4f})")
                valid_chunks += 1

        if valid_chunks == 0:
            self.logger.warning("No valid text chunks for sentiment")
            return

        # Normalize scores
        for label in sentiment_scores:
            sentiment_scores[label] /= valid_chunks

        final_sentiment = max(sentiment_scores.items(), key=lambda x: x[1])
        sentiment_label, confidence = final_sentiment

        self.logger.info(f"Final sentiment: {sentiment_label} | Confidence: {confidence:.4f}")

        yield {
            'title': title,
            'sentiment_label': sentiment_label,
            'confidence': confidence,
            'url': response.url
        }

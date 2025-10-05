from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
from pathlib import Path

def download_patent_urls_csv(driver, query):
    try:
        search_url = f"https://patents.google.com/?q={query.replace(' ', '+')}"
        driver.get(search_url)
        time.sleep(5)
        driver.find_element(By.CSS_SELECTOR, "#count > div.layout.horizontal.style-scope.search-results > span.headerButton.style-scope.search-results > a").click()
        print("Clicked on 'Download CSV' button")
        time.sleep(10)

    except Exception as e:
        print(f"Error navigating to search URL: {e}")
        return []


#=======Main Function=======#
def main():
    queries = [
        "Transformer Language Model",
        "Federated Learning Privacy",
        "Reinforcement Learning Policy",
        "Generative Diffusion Model",
        "Neural Network Compression",
        "Explainable AI System",
        "Computer Vision Segmentation",
        "Anomaly Detection Time Series",
        "Natural Language Understanding",
        "Active Learning Data Selection",
        "Edge AI Accelerator",
        "Transfer Learning Pre-trained",
        "Model Drift Monitoring",
        "Graph Neural Network Embedding",
        "Causal Inference AI",
        "Hyperparameter Optimization Automated",
        "Synthetic Data Generation",
        "Adversarial Machine Learning",
        "Deep Learning Compiler",
        "Model Deployment Pipeline"
    ]

    # Define the desired download directory
    path = Path("data/downloads_patent_urls")
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

    chrome_options = Options()
    prefs = {
        "download.default_directory": str(path.resolve()),      # Set custom download directory
        "download.prompt_for_download": False,                  # Disable download prompt
        "safebrowsing.enabled": False                           # Disable security warnings for downloads
    }
    chrome_options.add_experimental_option("prefs", prefs)
    chrome_options.add_argument("--headless=new")               # Run in headless mode


    # Initialize WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    # Process each query to download CSVs
    for query in queries:
        download_patent_urls_csv(driver, query)
        print(f"Downloaded CSV file for the query {query}")
      
    driver.quit()


#=======Main Execution=======#
if __name__ == "__main__":
    main()
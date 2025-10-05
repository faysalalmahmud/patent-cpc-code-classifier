#import necessary libraries
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


# Adjust based on CPU cores and RAM (4-8 is a good starting point).
MAX_WORKERS = 4

#Each run Specify the input CSV file containing patent URLs
file_name = "gp-search-20251001-102842.csv"


#=======CPC Extractor Function=======#
def cpc_extractor(driver):
    try:
        # Initialize WebDriverWait
        wait = WebDriverWait(driver, 10)

        view_more_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div.more.style-scope.classification-viewer"))
        )
        view_more_button.click()
        # time.sleep(1) 
                     
    except Exception as e:
        print("No 'View more classifications' link found or already expanded")

    # Initialize a list of CPC codes
    cpc_codes_list = [] 
    
    try:
        scoped_selector = 'div.table.classification-viewer a[id="link"]'
        classification_links = driver.find_elements(By.CSS_SELECTOR, scoped_selector)
        
        for link in classification_links:
            code = link.text.strip()
            if code:
                cpc_codes_list.append(code)

    except Exception as e:
        print(f"Error extracting CPC codes: {e}")

    # Remove unwanted entries [Search, Add to query] if they exist
    try:
        cpc_codes_list.remove('Search')
    except ValueError:
        pass

    try:
        cpc_codes_list.remove('Add to query')
    except ValueError:
        pass
    
    return cpc_codes_list


#=======Scrape Patent Data Function=======#
def scrape_patent_data(url, chrome_options):
    # Initialize WebDriver for each thread
    driver = webdriver.Chrome(options=chrome_options) 
    data = None
    try:
        driver.get(url)
        WebDriverWait(driver, 10).until(
             EC.visibility_of_element_located((By.ID, 'pubnum'))
        )

        # Find the data elements
        publication_numbers = driver.find_element(By.ID, "pubnum").text
        title = driver.find_element(By.CLASS_NAME, "scroll-target.style-scope.patent-result").text
        abstract = driver.find_element(By.CLASS_NAME, "abstract.style-scope.patent-text").text
        cpc = cpc_extractor(driver)                    # Uses the cpc_extractor function to get CPC codes
        
        data = {
            "publication_number": publication_numbers,
            "title": title,
            "abstract": abstract,
            "cpc_codes": cpc
        }
        # print(f"Successfully Scraped data for the url: {url}") # Logging is cleaner in main loop

    except Exception as e:
         print(f"Error: Timeout waiting for elements on {url}")
    except Exception as e:
        print(f"Error scraping data for {url}: {e}")
    finally:
        driver.quit() #Quit the driver when the thread finishes
        
    return data


#=======Get Patent URLs Function=======#
def getting_patent_urls(file_name):
    df = pd.read_csv(f"data/downloads_patent_urls/{file_name}")
    df = df[['result link']]
    df = df.rename(columns={'result link': 'Url'})
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.reset_index(drop=True)
    urls_list = df['Url'].tolist()
    return urls_list


#=======Main Function=======#
def main():
    chrome_options = Options()
    # Adding options for stability in headless mode
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")
    
    urls = getting_patent_urls(file_name) 
    patent_details = []
    
    # Adjust the limit to first 2000 URLs; for full run, use all URLs
    urls_to_scrape = urls[: 2500]  
    
    print(f"Starting parallel scraping of {len(urls_to_scrape)} URLs with {MAX_WORKERS} workers...")

    # Use ThreadPoolExecutor for concurrent execution
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map URLs to the scrape_patent_data function
        future_to_url = {executor.submit(scrape_patent_data, url, chrome_options): url for url in urls_to_scrape}
        
        #scraped data with progress bar
        for future in tqdm(as_completed(future_to_url), total=len(urls_to_scrape), desc="Scraping Patent Details"):
            url = future_to_url[future]
            try:
                data = future.result()
                if data:
                    patent_details.append(data)
            except Exception as e:
                # This catches exceptions that happened inside the thread
                print(f"Thread for {url} raised an unhandled exception: {e}")



    # Save the collected data to a CSV file
    if patent_details:
        df = pd.DataFrame(patent_details)
        output_path = Path("data/scraped")
        output_path.mkdir(parents=True, exist_ok=True) # Ensure path exists

        output_file = output_path / f"patent_details_{file_name[10:-4]}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(df)} patent details to {output_file}")


if __name__ == "__main__":
    main()
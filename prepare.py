import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

class SHLScraper:
    def __init__(self, 
                 base_url="https://www.shl.com/solutions/products/product-catalog/",
                 output_file="shl_df.csv",
                 total_data=388, 
                 page_size=12,
                 delay=1):
        
        self.base_url = base_url
        self.headers = {"User-Agent": "Mozilla/5.0"}
        self.output_file = output_file
        self.total_data = total_data
        self.page_size = page_size
        self.delay = delay
        self.results = []

    def parse_dot(self, td):
        """
        Parses a table cell to determine a Yes/No flag based on the presence of a specific span class.
        """
        return "Yes" if td.find("span", class_="catalogue__circle -yes") else "No"

    def parse_test_type(self, td):
        keys = td.find_all("span", class_="product-catalogue__key")
        return ", ".join([k.get_text(strip=True) for k in keys])

    def extract_minutes_from_string(self, duration_str):
        if not duration_str:
            return ""
        match = re.search(r'\d+', duration_str)
        return int(match.group()) if match else ""

    def enrich_detail_page(self, row):
        """
        Fetches the assessment detail page and extracts the description and duration.        
        """
        url = row["URL"]
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            description = ""
            # Loop through all <h4> tags to locate relevant sections
            h4_tags = soup.find_all("h4")
            for h4 in h4_tags:
                header_text = h4.text.strip().lower()
                if header_text == "description":
                    next_p = h4.find_next_sibling("p")
                    if next_p:
                        description = next_p.text.strip()
                elif header_text == "job levels":
                    next_p = h4.find_next_sibling("p")
                    if next_p:
                        description += "\nThe Job levels: " + next_p.text.strip()
                elif header_text == "languages":
                    next_p = h4.find_next_sibling("p")
                    if next_p:
                        description += "\nLanguages allowed: " + next_p.text.strip()
                        break

            # Extract duration by searching for text containing "minute"
            duration = ""
            for tag in soup.find_all(text=True):
                if "minute" in tag.lower():
                    duration = self.extract_minutes_from_string(tag.strip())
                    break

            row["Description"] = description
            row["Duration"] = duration
            row["Embedding text"]= self.create_embedding_text(row)
        except Exception as e:
            print(f"Failed for {url} - {e}")
        return row

    def parse_page(self, start_val):
        """
        Scrapes a single page of the product catalog starting at the given offset.
        """
        print(f"Scraping start={start_val}")
        params = {"type": 1, "start": start_val}
        response = requests.get(self.base_url, params=params, headers=self.headers)
        soup = BeautifulSoup(response.content, "html.parser")

        # Locate the table with "Individual Test Solutions"
        tables = soup.find_all("table")
        target_table = None
        for table in tables:
            header = table.find("th")
            if header and "Individual Test Solutions" in header.text:
                target_table = table
                break

        if not target_table:
            print("Individual Test Solutions table not found.")
            return

        # Process each row in the table, skipping the header
        rows = target_table.find_all("tr")[1:]
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 4:
                continue

            name_tag = cols[0].find("a")
            name = name_tag.text.strip()
            url = name_tag["href"]
            if not url.startswith("http"):
                url = "https://www.shl.com" + url

            remote_testing = self.parse_dot(cols[1])
            adaptive_irt = self.parse_dot(cols[2])
            test_type = self.parse_test_type(cols[3])
            
            row_data = {
                "Assessment Name": name,
                "URL": url,
                "Remote Testing": remote_testing,
                "Adaptive/IRT": adaptive_irt,
                "Test Type": test_type
            }
            # Enrich the row with detail page data
            enriched_row = self.enrich_detail_page(row_data)
            self.results.append(enriched_row)
    
    def create_embedding_text(self, row):
        duration = row['Duration']
        if pd.isna(duration):
            duration_text = "The time limit is not defined."
        else:
            duration_text = f"The typical duration for this assessment is {duration} minutes."

        return (
            f"{row['Assessment Name']} is an SHL assessment designed to evaluate {row['Description']}.\n"
            f"It falls under the category of '{row['Test Type']}', which includes:\n"
            "A = Ability & Aptitude, B = Biodata & Situational Judgement, C = Competencies,\n"
            "D = Development & 360, E = Assessment Exercises, K = Knowledge & Skills, P = Personality & Behavior.\n"
            f"{duration_text}"
        )
        

    def scrape_all(self):
        """
        Iterates through all data and scrapes data based on pagination.
        """
        for start in range(0, self.total_data, self.page_size):
            self.parse_page(start)
            time.sleep(self.delay)

    def to_dataframe(self):
        """
        Converts the scraped results into a pandas DataFrame.
        """
        return pd.DataFrame(self.results)

    def save_to_csv(self):
        
        df = self.to_dataframe()
        df.to_csv(self.output_file, index=False)
        print(f"Data saved to {self.output_file}")
    
        

    def run(self):
        self.scrape_all()
        self.save_to_csv()

if __name__ == "__main__":
    scraper = SHLScraper(total_data=388)  # 388 because the loop is range(0,388,12)
    scraper.run()

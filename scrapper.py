import gzip
import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


# --------------------------------------------------
# 1. GET ALL URLS FROM SITEMAP
# --------------------------------------------------

def get_sitemap_urls(sitemap_url: str, max_depth: int = 3) -> list[str]:

    if max_depth <= 0:
        return []

    try:
        response = requests.get(
            sitemap_url,
            headers=HEADERS,
            timeout=15
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch sitemap: {e}")
        return []

    content = response.content

    # Handle compressed sitemap
    if sitemap_url.endswith(".gz") or content[:2] == b"\x1f\x8b":
        try:
            content = gzip.decompress(content)
        except Exception:
            print("Could not decompress sitemap.")
            return []

    soup = BeautifulSoup(content, "xml")

    # Check whether the response is a valid sitemap
    if not soup.find("urlset") and not soup.find("sitemapindex"):
        print("Invalid sitemap or blocked request.")
        return []

    # Handle nested sitemap index
    if soup.find("sitemapindex"):

        sub_sitemap_urls = [
            loc.get_text(strip=True)
            for loc in soup.find_all("loc")
        ]

        all_page_urls = []

        for sub_url in sub_sitemap_urls:
            all_page_urls.extend(
                get_sitemap_urls(
                    sub_url,
                    max_depth=max_depth - 1
                )
            )

        return sorted(set(all_page_urls))

    # Handle standard sitemap
    page_urls = [
        loc.get_text(strip=True)
        for loc in soup.find_all("loc")
    ]

    return sorted(set(page_urls))


def scrape_input(url: str):

    # Check whether the URL is a sitemap
    if "sitemap" in url.lower() or url.lower().endswith(".xml"):

        print("Sitemap detected. Starting website crawling...")

        return crawl_website(url)

    else:

        print("Normal webpage detected. Scraping single page...")

        return scrape_website(url)


# --------------------------------------------------
# 2. SCRAPE INDIVIDUAL WEBSITE PAGE
# --------------------------------------------------

def scrape_website(url: str) -> dict:

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return {"error": str(e), "url": url}

    soup = BeautifulSoup(response.text, "html.parser")

    page_url = response.url

    # Remove unwanted HTML elements
    for tag in soup([
        "script",
        "style",
        "noscript",
        "nav",
        "footer",
        "aside"
    ]):
        tag.decompose()

    # Extract title
    title = (
        soup.title.get_text(strip=True)
        if soup.title
        else "No Title"
    )

    # Extract readable text
    text = soup.get_text(
        separator="\n",
        strip=True
    )

    return {
        "url": page_url,
        "title": title,
        "text": text
    }


# --------------------------------------------------
# 3. COMBINE SITEMAP + SCRAPER
# --------------------------------------------------

def crawl_website(sitemap_url: str) -> list[dict]:

    # Step 1: Get all webpage URLs
    urls = get_sitemap_urls(sitemap_url)

    print(f"\nTotal URLs discovered: {len(urls)}")

    scraped_pages = []

    # Step 2: Visit every webpage
    for index, url in enumerate(urls, start=1):

        print(f"\nScraping page {index}/{len(urls)}")
        print(f"URL: {url}")

        # Step 3: Extract webpage content
        page_data = scrape_website(url)

        # Step 4: Store successful results
        if "error" not in page_data and page_data["text"].strip():

            scraped_pages.append(page_data)

            print(f"Title: {page_data['title']}")
            print(f"Characters extracted: {len(page_data['text'])}")

        else:
            print("Could not extract page content.")

    return scraped_pages


# --------------------------------------------------
# 4. MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    input_url = "https://en.wikipedia.org/wiki/Python_(programming_language)?utm_source=chatgpt.com"

    website_data = scrape_input(input_url)

    print("\n" + "=" * 50)
    print("SCRAPING COMPLETED")
    print("=" * 50)

    if isinstance(website_data, list):

        print("Successfully scraped pages:", len(website_data))

        for page in website_data:
            print("\nTitle:", page["title"])
            print("URL:", page["url"])

            print("\n" + "="     * 50)
            print("COMPLETE EXTRACTED     TEXT")
            print("=" * 50)

            print(page["text"])

    else:

        if "error" not in website_data:
            print("Title:", website_data["title"])
            print("URL:", website_data["url"])

            print("\n" + "=" * 50)
            print("COMPLETE EXTRACTED TEXT")
            print("=" * 50)

            print(website_data["text"][:300])

        else:
            print("Scraping failed:", website_data["error"])
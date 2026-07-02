import requests

url = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
output_file = "shl_product_catalog.json"

try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()  # Raises an exception for HTTP errors

    with open(output_file, "wb") as f:
        f.write(response.content)

    print(f"Downloaded successfully as {output_file}")

except requests.exceptions.RequestException as e:
    print("Download failed:", e)
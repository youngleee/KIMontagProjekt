import requests

def query_it(question):
    base_url = "https://api.stackexchange.com/2.3/search/advanced"
    params = {
        "order": "desc",
        "sort": "relevance",
        "q": question,
        "site": "stackoverflow"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["items"]:
            return data["items"][0]["title"]
    return "No relevant IT data found."

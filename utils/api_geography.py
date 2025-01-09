import requests

def query_geography(question):
    geonames_username = os.getenv("GEONAMES_USERNAME")  # From .env file
    base_url = "http://api.geonames.org/searchJSON"
    params = {"q": question, "maxRows": 1, "username": geonames_username}
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data["geonames"]:
            return data["geonames"][0]["name"]
    return "No relevant geographic data found."

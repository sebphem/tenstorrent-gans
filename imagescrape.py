import requests
import os
from bs4 import BeautifulSoup

def img_scrape(query, num):
	format_query = query.replace(' ', '+')
	url = f"https://www.google.com/search?q={format_query}&source=lnms&tbm=isch"
	headers = {"User-Agent": "Mozilla/5.0"}
	response = requests.get(url, headers=headers)
	soup = BeautifulSoup(response.text, "html.parser")

	img_urls = []
	for img_tag in soup.find_all("img"):
		if len(img_urls) >= num:
			break
		try:
			img_url = img_tag["src"]
			if img_url.startswith("http"):
				img_urls.append(img_url)
		except KeyError:
			print("No src attribute in img tag")
			pass

	os.makedirs(f"RAW_{query.repalce(' ', "_")}", exist_ok=True)
	for i, img_url in enumerate(img_urls):
		img_data = requests.get(img_url).content
		with open(f"{query}/{i}.jpg", "wb") as img_file:
			img_file.write(img_data)
   
	print(f"Downloaded {len(img_urls)} images of {query}")
 
img_scrape("Dwayne The Rock Johnson", 10)
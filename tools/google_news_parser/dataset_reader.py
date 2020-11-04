import os
from bs4 import BeautifulSoup

output_file = open("categories.csv", "w")

dataset_path = os.getcwd() + "/dataset/google_nesw/"
for filename in os.listdir(dataset_path):
    input_file = open(dataset_path + filename)

    soup = BeautifulSoup(input_file, features="html.parser")
    body = soup.find("body")
    if not body:
        continue

    for script in body.find_all("script"):
        script.extract()

    category = filename[:-13]
    for idx, topic in enumerate(body.find_all("div", {"class": "NiLAwe mi8Lec gAl5If jVwmLb Oc0wGc R7GTQ keNKEd j7vNaf nID9nc"})):
        for article in topic.find_all("article"):
            article_title = article.find("h3")
            if article_title == None:
                article_title = article.find("h4")
            output_file.write(category + "\t" + str(idx) + "\t" + article_title.get_text() + "\n")
        output_file.write("\n")

output_file.close()
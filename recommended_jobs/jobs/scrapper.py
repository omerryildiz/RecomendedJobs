
from bs4 import BeautifulSoup as bts
import requests
import sys
import json
import csv
import string

def getAndResultURL(url):
    result = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = bts(result.content, 'html.parser')
    return soup

    def extract_text(html):
        soup = bts(html, 'html.parser')
        text = ""

        for p in soup.find_all(['p', 'li']):
            text += p.get_text() + "\n"

        return text

    def remove_punctuation(text):
         return text.translate(str.maketrans("", "", string.punctuation))

    def find_matching_skills(text):
        file_path = os.path.join('jobs/documents','skill_list.txt')
        with open(file_path, 'r') as file:
            skills = [line.strip().lower() for line in file]

        matched_skills = set()

        text = remove_punctuation(text).lower()

        for skill in skills:
            if re.search(r'\W' + re.escape(skill) + r'\W', ' ' + text + ' '):
                matched_skills.add(skill)

        result = ', '.join(matched_skills)
        return result

    JOB_PAGE = []
    for pg in range(1, 18):
        JOB_PAGE.append("https://www.kariyer.net/is-ilanlari/yazilim+uzmani-2?pst=2024&pkw=Yazilim%20Uzmani&cp="+str(pg))

    JOB_LINK = []
    for pages in JOB_PAGE[::]:
        html = getAndResultURL(pages)
        for links in html.findAll("div", {"class": "list-items"}):
            JOB_LINK.append("https://www.kariyer.net/" + links.a["href"])

    disctionaryArr = []
    for detail in JOB_LINK[:]:
        html = getAndResultURL(detail)
        job_titles = html.find('a', {'class': 'link position'}).text.strip()
        company_name = html.find('a', {'class': 'link company'}).text.strip()
        requirement = html.find('div', {'class': 'genel-nitelikler'})
        clean_html = extract_text(str(requirement))
        finall = find_matching_skills(clean_html)

        # Skill kısmı boş olanları filtrele
        if finall:
            link1 = detail
            img_url = ''
            img_element = html.find('div', {'class': 'logo-wrapper'})
            if img_element and img_element.find('img'):
                img_src = img_element.find('img')['src']
            match = re.search(r'https://img-kariyer\.mncdn\.com/UploadFiles/Clients/Logolar/.+', img_src)
            if match:
                img_url = match.group(0)

            job_dict = {
                'title': job_titles,
                'description': company_name,
                "skill": finall,
                "link": link1,
                'img url': img_url,
            }
            disctionaryArr.append(job_dict)

    data = disctionaryArr
    output_file = os.path.join('jobs/documents', 'processed_requirement.csv')
    output = csv.writer(open(output_file, "w"))
    output.writerow(data[0].keys())  # header row
    for row in data:
        output.writerow(row.values())





import requests
from bs4 import BeautifulSoup


url = 'https://www.youtube.com/user/PewDiePie'


url2 = 'https://www.youtube.com/user/tseries'

page = requests.get(url)

page2 = requests.get(url2)


soup = BeautifulSoup(page.text, "html.parser")
soup2 = BeautifulSoup(page2.text, "html.parser")


pew = soup.find_all("span", class_=lambda x: x and "yt-subscription-button-subscriber-count-branded-horizontal" in x)


tseries = soup2.find_all("span", class_=lambda x: x and "yt-subscription-button-subscriber-count-branded-horizontal subscribed yt-uix-tooltip" in x)
for subs in pew:
  print(subs.get_text())

for subs1 in tseries:
  print(subs1.get_text())

pewdiepie = subs.get_text().replace(",", "")
pewdiepie

tseries = subs1.get_text().replace(",", "")
tseries

difference = int(tseries) - int(pewdiepie)

print("The sub gap between T-series and PewDiePie is  ==> "'{:,}'.format(difference))
#print(page2)

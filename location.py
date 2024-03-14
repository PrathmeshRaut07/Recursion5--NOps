import json
from urllib.request import urlopen

url='http://ipinfo.io/json'

response = urlopen(url)

data = json.load(response)
loca=data['loc']
loca=loca.split(',')
latitute=loca[0]
longitude=loca[1]
print("Latitude:",latitute)
print("Longitude:",longitude)
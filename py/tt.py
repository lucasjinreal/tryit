
import xml.etree.ElementTree as ET


tree = ET.parse('a.xml')
root = tree.getroot()

to_add = []
for country in root.findall('country'):
    rank = int(country.find('rank').text)
    if rank > 50:
        to_add.append(country)

for a in to_add:
    root.append(a)
    root.append(a)
    root.append(a)
    root.append(a)

tree.write('output.xml')
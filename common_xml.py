# Import the required modules
import xmltodict


def xml_to_dic(xml_path):
    # Open the file and read the contents
    with open(xml_path, 'r', encoding='utf-8') as file:
        xml = file.read()
    return xmltodict.parse(xml)
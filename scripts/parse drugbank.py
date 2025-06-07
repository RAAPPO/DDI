import xml.etree.ElementTree as ET
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to the DrugBank XML file
drugbank_xml_path = "G:/drug/src/full_database.xml"

# Function to parse the DrugBank XML file
def parse_drugbank_xml(xml_path):
    logging.info("Starting to parse the DrugBank XML file.")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    drugs = []
    interactions = []

    # Namespace
    ns = {"db": "http://www.drugbank.ca"}

    # Iterate through each drug in the XML
    for drug in root.findall("db:drug", ns):
        drug_info = {}
        drug_info["drugbank_id"] = drug.find("db:drugbank-id", ns).text
        drug_info["name"] = drug.find("db:name", ns).text
        drug_info["description"] = drug.find("db:description", ns).text if drug.find("db:description", ns) is not None else ""
        drug_info["cas_number"] = drug.find("db:cas-number", ns).text if drug.find("db:cas-number", ns) is not None else ""
        drug_info["unii"] = drug.find("db:unii", ns).text if drug.find("db:unii", ns) is not None else ""
        drug_info["average_mass"] = drug.find("db:average-mass", ns).text if drug.find("db:average-mass", ns) is not None else ""
        drug_info["monoisotopic_mass"] = drug.find("db:monoisotopic-mass", ns).text if drug.find("db:monoisotopic-mass", ns) is not None else ""
        drug_info["state"] = drug.find("db:state", ns).text if drug.find("db:state", ns) is not None else ""

        # Pharmacological Information
        drug_info["indication"] = drug.find("db:indication", ns).text if drug.find("db:indication", ns) is not None else ""
        drug_info["pharmacodynamics"] = drug.find("db:pharmacodynamics", ns).text if drug.find("db:pharmacodynamics", ns) is not None else ""
        drug_info["mechanism_of_action"] = drug.find("db:mechanism-of-action", ns).text if drug.find("db:mechanism-of-action", ns) is not None else ""
        drug_info["toxicity"] = drug.find("db:toxicity", ns).text if drug.find("db:toxicity", ns) is not None else ""
        drug_info["metabolism"] = drug.find("db:metabolism", ns).text if drug.find("db:metabolism", ns) is not None else ""
        drug_info["absorption"] = drug.find("db:absorption", ns).text if drug.find("db:absorption", ns) is not None else ""
        drug_info["half_life"] = drug.find("db:half-life", ns).text if drug.find("db:half-life", ns) is not None else ""
        drug_info["protein_binding"] = drug.find("db:protein-binding", ns).text if drug.find("db:protein-binding", ns) is not None else ""
        drug_info["route_of_elimination"] = drug.find("db:route-of-elimination", ns).text if drug.find("db:route-of-elimination", ns) is not None else ""
        drug_info["volume_of_distribution"] = drug.find("db:volume-of-distribution", ns).text if drug.find("db:volume-of-distribution", ns) is not None else ""
        drug_info["clearance"] = drug.find("db:clearance", ns).text if drug.find("db:clearance", ns) is not None else ""

        # Calculated Properties
        calculated_properties = {}
        for property in drug.findall("db:calculated-properties/db:property", ns):
            kind = property.find("db:kind", ns).text
            value = property.find("db:value", ns).text
            calculated_properties[kind] = value
        drug_info.update(calculated_properties)

        # Collect drug interactions
        for interaction in drug.findall("db:drug-interactions/db:drug-interaction", ns):
            interaction_info = {
                "drugbank_id": drug_info["drugbank_id"],
                "interaction_id": interaction.find("db:drugbank-id", ns).text,
                "interaction_name": interaction.find("db:name", ns).text,
                "interaction_description": interaction.find("db:description", ns).text,
            }
            interactions.append(interaction_info)

        drugs.append(drug_info)
        logging.info(f"Parsed drug: {drug_info['name']}")

    # Convert to DataFrame
    drugs_df = pd.DataFrame(drugs)
    interactions_df = pd.DataFrame(interactions)

    logging.info("Completed parsing the DrugBank XML file.")
    return drugs_df, interactions_df

# Parse the XML file
drugs_df, interactions_df = parse_drugbank_xml(drugbank_xml_path)

# Save to CSV for further processing
drugs_df.to_csv("G:/drug/src/drugs.csv", index=False)
interactions_df.to_csv("G:/drug/src/interactions.csv", index=False)

logging.info("DrugBank XML parsing completed and data saved to CSV files.")
